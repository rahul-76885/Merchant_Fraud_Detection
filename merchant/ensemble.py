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

from .dl_models import (
    DNNFraudModel,
    FTTransformerFraudModel,
    load_pickle_artifact,
    load_torch_checkpoint,
    model_path,
    torch_available,
)
from .models import ANNModelAdapter, LightGBMModelAdapter, TransformerModelAdapter, XGBoostModelAdapter
from .preprocessing import FraudPreprocessor

if torch_available():
    import torch

logger = logging.getLogger("merchant.ensemble")

DEFAULT_WEIGHTS = {"xgboost": 0.35, "lightgbm": 0.35, "ann": 0.15, "transformer": 0.15, "anomaly": 0.0}
MODEL_FILENAMES = {
    "xgboost": ("xgb_model.pkl", "xgboost.pkl"),
    "lightgbm": ("lgb_model.pkl", "lightgbm.pkl"),
    "anomaly": ("anomaly.pkl",),
}
TORCH_MODEL_FILENAMES = {"ann": "dnn_model.pt", "transformer": "transformer_model.pt"}


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
        self.models: dict[str, Any] = {
            "xgboost": None,
            "lightgbm": None,
            "ann": None,
            "transformer": None,
            "anomaly": None,
        }
        self.model_status = {key: False for key in self.models}
        self.weights = DEFAULT_WEIGHTS.copy()
        self.loaded = False
        self.device = torch.device("cuda" if torch_available() and torch.cuda.is_available() else "cpu") if torch_available() else None

    def load_all(self):
        start = time.perf_counter()
        try:
            self.preprocessor = self._load_preprocessor()
            self.weights = self._load_weights()
            self.models["xgboost"], self.model_status["xgboost"] = self._load_model("xgboost")
            self.models["lightgbm"], self.model_status["lightgbm"] = self._load_model("lightgbm")
            self.models["anomaly"], self.model_status["anomaly"] = self._load_model("anomaly", required=False)
            self.models["ann"], self.model_status["ann"] = self._load_torch_model("ann")
            self.models["transformer"], self.model_status["transformer"] = self._load_torch_model("transformer")
            self.loaded = True
        except Exception as exc:
            self.loaded = False
            raise RuntimeError(f"Failed to load model bundle from {self.model_dir}: {exc}") from exc
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        logger.warning(
            "Model bundle ready in %.2f ms | xgboost=%s lightgbm=%s ann=%s transformer=%s anomaly=%s",
            elapsed_ms,
            self.model_status["xgboost"],
            self.model_status["lightgbm"],
            self.model_status["ann"],
            self.model_status["transformer"],
            self.model_status["anomaly"],
        )
        return self

    def status(self) -> dict[str, bool]:
        return dict(self.model_status)

    def predict_transaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model bundle is not loaded. Call load_ensemble() during startup.")

        start = time.perf_counter()
        cleaned = self._normalize_payload(payload)
        model_probabilities = self.get_all_model_predictions(cleaned)
        base_available = {
            name: value
            for name, value in model_probabilities.items()
            if name not in {"ensemble", "autoencoder", "isolation_forest"}
        }
        if not base_available:
            raise RuntimeError("No predictive model produced an output.")

        weighted_probability = self._weighted_average(base_available)
        soft_probability = self._soft_average(base_available)
        raw_probability = weighted_probability
        if "isolation_forest" in model_probabilities:
            anomaly_weight = float(np.clip(self.weights.get("anomaly", 0.0), 0.0, 0.3))
            raw_probability = (1.0 - anomaly_weight) * weighted_probability + anomaly_weight * model_probabilities["isolation_forest"]
        raw_probability = float(np.clip(raw_probability, 0.0, 1.0))

        fraud_score = int(round(raw_probability * 100))
        fraud_label = self._label_from_score(fraud_score)
        risk_color = self._risk_color_from_label(fraud_label)
        fraud_type = self._fraud_type(cleaned, fraud_score)
        models = dict(model_probabilities)
        models["soft_voting"] = float(round(soft_probability, 6))
        models["weighted_voting"] = float(round(weighted_probability, 6))
        models["ensemble"] = float(round(raw_probability, 6))

        model_scores = {name: round(value * 100.0, 1) for name, value in models.items()}
        for name in (
            "xgboost",
            "lightgbm",
            "transformer",
            "ann",
            "autoencoder",
            "isolation_forest",
            "soft_voting",
            "weighted_voting",
            "ensemble",
        ):
            if name not in model_scores:
                model_scores[name] = 0.0

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
        result["models"] = models
        result["final_prediction"] = int(raw_probability >= 0.5)
        result["confidence"] = float(round(raw_probability, 6))
        result["model_outputs"] = {
            "xgboost": float(round(models.get("xgboost", 0.0), 6)),
            "lightgbm": float(round(models.get("lightgbm", 0.0), 6)),
            "ann": float(round(models.get("ann", 0.0), 6)),
            "transformer": float(round(models.get("transformer", 0.0), 6)),
        }
        result["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        result["amount"] = self._amount(cleaned)
        logger.warning(
            "Prediction summary | final=%s confidence=%.4f weighted=%.4f soft=%.4f",
            result["final_prediction"],
            result["confidence"],
            weighted_probability,
            soft_probability,
        )
        return result

    def get_all_model_predictions(self, input_data: dict[str, Any]) -> dict[str, float]:
        frame = pd.DataFrame([input_data])
        transformed = self.preprocessor.transform_frame(frame)
        deep_num, deep_cat = self.preprocessor.transform_deep_tensors(frame)

        predictions: dict[str, float | None] = {}
        predictions["xgboost"] = self._safe_model_predict("xgboost", lambda: self._predict_xgboost(transformed))
        predictions["lightgbm"] = self._safe_model_predict("lightgbm", lambda: self._predict_lightgbm(transformed))
        predictions["transformer"] = self._safe_model_predict(
            "transformer", lambda: self._predict_transformer(deep_num, deep_cat)
        )
        predictions["ann"] = self._safe_model_predict("ann", lambda: self._predict_ann(deep_num, deep_cat))
        predictions["isolation_forest"] = self._safe_model_predict(
            "isolation_forest", lambda: self._predict_isolation_forest(transformed)
        )

        available: dict[str, float] = {}
        for name, value in predictions.items():
            if value is None:
                continue
            available[name] = float(np.clip(value, 0.0, 1.0))
        return available

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
                if "dnn" in payload and "ann" not in payload:
                    payload["ann"] = payload["dnn"]
                for key, value in payload.items():
                    if key in weights:
                        weights[key] = float(value)
                positive = {key: max(value, 0.0) for key, value in weights.items()}
                total = sum(positive.values())
                if total <= 0:
                    return DEFAULT_WEIGHTS.copy()
                return {key: value / total for key, value in positive.items()}
        except Exception as exc:
            raise RuntimeError(f"Failed to load ensemble.pkl: {exc}") from exc
        raise RuntimeError("ensemble.pkl does not contain a valid mapping.")

    def _load_model(self, name: str, required: bool = True):
        candidates = MODEL_FILENAMES[name]
        selected = None
        for filename in candidates:
            path = model_path(filename)
            if os.path.exists(path):
                selected = filename
                break
        if selected is None:
            if required:
                expected = ", ".join(model_path(filename) for filename in candidates)
                raise FileNotFoundError(f"Missing required artifact. Tried: {expected}")
            return None, False
        try:
            model = load_pickle_artifact(selected)
            return model, True
        except Exception as exc:
            raise RuntimeError(f"Failed to load {selected}: {exc}") from exc

    def _load_torch_model(self, name: str):
        if not torch_available():
            return None, False
        filename = TORCH_MODEL_FILENAMES[name]
        path = model_path(filename)
        if not os.path.exists(path):
            return None, False
        checkpoint = load_torch_checkpoint(filename, map_location=str(self.device))
        num_features = int(checkpoint.get("num_features", 0))
        cat_cardinalities = [int(v) for v in checkpoint.get("cat_cardinalities", [])]

        if name == "ann":
            model = DNNFraudModel(
                num_features=num_features,
                cat_cardinalities=cat_cardinalities,
                hidden_dims=checkpoint.get("hidden_dims", [256, 128, 64]),
                dropout=float(checkpoint.get("dropout", 0.25)),
            )
        else:
            model = FTTransformerFraudModel(
                num_features=num_features,
                cat_cardinalities=cat_cardinalities,
                d_model=int(checkpoint.get("d_model", 64)),
                n_heads=int(checkpoint.get("n_heads", 4)),
                n_layers=int(checkpoint.get("n_layers", 2)),
                dropout=float(checkpoint.get("dropout", 0.2)),
            )

        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model, True

    def _predict_xgboost(self, transformed: Any) -> float | None:
        model = self.models.get("xgboost")
        if model is None:
            return None
        return XGBoostModelAdapter(model).predict_proba(transformed)

    def _predict_lightgbm(self, transformed: Any) -> float | None:
        model = self.models.get("lightgbm")
        if model is None:
            return None
        return LightGBMModelAdapter(model).predict_proba(transformed)

    def _predict_ann(self, x_num: np.ndarray, x_cat: np.ndarray) -> float | None:
        model = self.models.get("ann")
        if model is None or not torch_available():
            return None
        return ANNModelAdapter(model, self.device).predict_proba(x_num, x_cat)

    def _predict_transformer(self, x_num: np.ndarray, x_cat: np.ndarray) -> float | None:
        model = self.models.get("transformer")
        if model is None or not torch_available():
            return None
        return TransformerModelAdapter(model, self.device).predict_proba(x_num, x_cat)

    def _predict_isolation_forest(self, transformed: Any) -> float | None:
        return self._predict_anomaly(self.models.get("anomaly"), transformed)

    def _safe_model_predict(self, model_name: str, fn) -> float | None:
        try:
            return fn()
        except Exception as exc:
            logger.warning("%s prediction failed and was skipped: %s", model_name, exc)
            return None

    def _predict_anomaly(self, model: Any, transformed: Any) -> float | None:
        if model is None:
            return None
        try:
            decision = float(model.decision_function(transformed)[0])
            return float(1.0 / (1.0 + np.exp(4.0 * decision)))
        except Exception as exc:
            raise RuntimeError(f"Anomaly prediction failed: {exc}") from exc

    def _weighted_average(self, probabilities: dict[str, float]) -> float:
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
    def _soft_average(probabilities: dict[str, float]) -> float:
        if not probabilities:
            raise RuntimeError("No probabilities available for soft voting")
        values = np.asarray(list(probabilities.values()), dtype=np.float32)
        return float(np.clip(values.mean(), 0.0, 1.0))

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


def get_all_model_predictions(input_data: dict[str, Any]) -> dict[str, float]:
    return get_model_bundle().get_all_model_predictions(input_data)