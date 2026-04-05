"""Simple fraud inference bundle for merchant predictions."""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .dl_models import load_pickle_artifact, model_path
from .preprocessing import FraudPreprocessor

logger = logging.getLogger("merchant.ensemble")

DEFAULT_WEIGHTS = {"xgboost": 0.5, "lightgbm": 0.5}
MODEL_FILENAMES = {"xgboost": "xgb_model.pkl", "lightgbm": "lgb_model.pkl"}


@dataclass
class PredictionResult:
    fraud_score: int
    fraud_label: str
    risk_color: str
    model_scores: dict[str, float]
    model_status: dict[str, bool]
    timestamp: str
    raw_probability: float
    fraud_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraud_score": self.fraud_score,
            "fraud_label": self.fraud_label,
            "risk_color": self.risk_color,
            "model_scores": self.model_scores,
            "model_status": self.model_status,
            "timestamp": self.timestamp,
            "raw_probability": self.raw_probability,
            "fraud_type": self.fraud_type,
        }


class FraudModelBundle:
    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = model_dir or model_path("")
        self.preprocessor = FraudPreprocessor()
        self.models: dict[str, Any] = {"xgboost": None, "lightgbm": None}
        self.model_status = {"xgboost": False, "lightgbm": False}
        self.weights = DEFAULT_WEIGHTS.copy()
        self.loaded = False

    def load_all(self):
        start = time.perf_counter()
        try:
            self.preprocessor = self._load_preprocessor()
            self.weights = self._load_weights()
            self.models["xgboost"], self.model_status["xgboost"] = self._load_model("xgboost")
            self.models["lightgbm"], self.model_status["lightgbm"] = self._load_model("lightgbm")
            self.loaded = True
        except Exception as exc:
            self.loaded = False
            raise RuntimeError(f"Failed to load model bundle from {self.model_dir}: {exc}") from exc
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        logger.info(
            "Model bundle ready in %.2f ms | xgboost=%s lightgbm=%s",
            elapsed_ms,
            self.model_status["xgboost"],
            self.model_status["lightgbm"],
        )
        return self

    def status(self) -> dict[str, bool]:
        return dict(self.model_status)

    def predict_transaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model bundle is not loaded. Call load_ensemble() during startup.")

        start = time.perf_counter()
        cleaned = self._normalize_payload(payload)
        frame = pd.DataFrame([cleaned])

        transformed = self.preprocessor.transform_frame(frame)

        probabilities: dict[str, float | None] = {}
        for name in ("xgboost", "lightgbm"):
            probabilities[name] = self._predict_model(self.models.get(name), transformed)

        missing = [name for name, value in probabilities.items() if value is None]
        if missing:
            raise RuntimeError(f"Model(s) failed to produce predictions: {', '.join(missing)}")

        available = {name: float(value) for name, value in probabilities.items() if value is not None}
        raw_probability = self._weighted_average(available)

        fraud_score = int(round(np.clip(raw_probability, 0.0, 1.0) * 100))
        fraud_label = self._label_from_score(fraud_score)
        risk_color = self._risk_color_from_label(fraud_label)
        fraud_type = self._fraud_type(cleaned, fraud_score)
        model_scores = {name: round(prob * 100.0, 1) for name, prob in available.items()}
        model_scores["ensemble"] = float(fraud_score)

        result = PredictionResult(
            fraud_score=fraud_score,
            fraud_label=fraud_label,
            risk_color=risk_color,
            model_scores=model_scores,
            model_status=self.status(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            raw_probability=float(round(raw_probability, 6)),
            fraud_type=fraud_type,
        ).to_dict()
        result["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        result["amount"] = self._amount(cleaned)
        return result

    def _load_preprocessor(self) -> FraudPreprocessor:
        path = model_path("preprocessor.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact: {path}")
        preprocessor = FraudPreprocessor.load(path)
        if not getattr(preprocessor, "is_fitted_", False):
            raise RuntimeError("Loaded preprocessor is not fitted.")
        return preprocessor

    def _load_weights(self) -> dict[str, float]:
        path = model_path("ensemble.pkl")
        if not os.path.exists(path):
            return DEFAULT_WEIGHTS.copy()
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict):
                weights = DEFAULT_WEIGHTS.copy()
                for key in weights:
                    if key in payload:
                        weights[key] = float(payload[key])
                total = sum(weights.values()) or 1.0
                return {key: value / total for key, value in weights.items()}
        except Exception as exc:
            raise RuntimeError(f"Failed to load ensemble.pkl: {exc}") from exc
        raise RuntimeError("ensemble.pkl does not contain a valid mapping.")

    def _load_model(self, name: str):
        filename = MODEL_FILENAMES[name]
        path = model_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact: {path}")
        try:
            model = load_pickle_artifact(filename)
            return model, True
        except Exception as exc:
            raise RuntimeError(f"Failed to load {filename}: {exc}") from exc

    def _predict_model(self, model: Any, transformed: Any) -> float | None:
        if model is None:
            return None
        try:
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(transformed)[:, 1]
                return float(probability[0])
            if hasattr(model, "predict"):
                raw = model.predict(transformed)
                raw = np.asarray(raw).reshape(-1)[0]
                return float(raw)
        except Exception as exc:
            raise RuntimeError(f"Model prediction failed: {exc}") from exc
        return None

    def _weighted_average(self, probabilities: dict[str, float]) -> float:
        missing = [name for name in DEFAULT_WEIGHTS if name not in probabilities]
        if missing:
            raise RuntimeError(f"Missing model probability for: {', '.join(missing)}")

        total_weight = 0.0
        weighted_sum = 0.0
        for name, probability in probabilities.items():
            weight = float(self.weights.get(name, 0.0))
            if weight <= 0:
                continue
            weighted_sum += probability * weight
            total_weight += weight
        if total_weight <= 0:
            raise RuntimeError("Invalid model weights: total weight must be positive.")
        return float(weighted_sum / total_weight)

    @staticmethod
    def _label_from_score(score: int) -> str:
        if score < 30:
            return "Low"
        if score <= 70:
            return "Medium"
        return "High"

    @staticmethod
    def _risk_color_from_label(label: str) -> str:
        if label == "High":
            return "red"
        if label == "Medium":
            return "yellow"
        return "green"

    def _fraud_type(self, payload: dict[str, Any], score: int) -> str:
        amount = self._amount(payload)
        txn_hour = int((self._number(payload, "TransactionDT", 0.0) // 3600) % 24)
        network = self._text(payload, "card_network")
        payment = self._text(payload, "payment_type")
        email_domain = self._text(payload, "P_emaildomain")

        card_signals = sum([network in {"visa", "mastercard"}, payment in {"credit", "debit", "card"}, amount >= 5000])
        behavior_signals = sum([txn_hour < 6 or txn_hour >= 22, self._is_missing(payload.get("billing_transaction_distance")), self._is_missing(payload.get("dist2")), amount >= 10000])
        channel_signals = sum([email_domain in {"gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com", "msn.com", "icloud.com", "aol.com"}, amount >= 1000])

        if score > 70:
            if behavior_signals >= card_signals and behavior_signals >= channel_signals:
                return "Behavioral"
            if card_signals >= channel_signals:
                return "Card"
            return "Suspicious"
        if score >= 30:
            if channel_signals >= card_signals:
                return "Channel"
            return "Suspicious"
        return "Suspicious"

    @staticmethod
    def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str):
                value = value.strip()
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value in (None, "", "null", "NULL", "None")

    @staticmethod
    def _number(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
        value = payload.get(key, default)
        if FraudModelBundle._is_missing(value):
            return float(default)
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _text(payload: dict[str, Any], key: str, default: str = "") -> str:
        value = payload.get(key, default)
        if FraudModelBundle._is_missing(value):
            return default
        return str(value).strip().lower()

    @classmethod
    def _amount(cls, payload: dict[str, Any]) -> float:
        for key in ("TransactionAmt", "amount", "Amount"):
            if key in payload:
                return cls._number(payload, key, 0.0)
        return 0.0


_BUNDLE: FraudModelBundle | None = None


def get_model_bundle() -> FraudModelBundle:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = FraudModelBundle()
    return _BUNDLE


def load_ensemble() -> FraudModelBundle:
    return get_model_bundle().load_all()


def predict_transaction(payload: dict[str, Any]) -> dict[str, Any]:
    return get_model_bundle().predict_transaction(payload)