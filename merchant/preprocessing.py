"""Preprocessing and feature engineering for merchant fraud inference."""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("merchant.preprocessing")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DROP_COLUMNS = {"isFraud", "TransactionID", "Unnamed: 0", "Unnamed: 0.1"}
USER_ID_CANDIDATES = ("card_id", "Card User", "Customer_1", "Customer_2")
TIME_CANDIDATES = ("TransactionDT", "transaction_time", "timestamp")
AMOUNT_CANDIDATES = ("TransactionAmt", "amount", "Amount")
BALANCE_OLD_CANDIDATES = ("oldbalanceOrg", "oldbalanceDest")
BALANCE_NEW_CANDIDATES = ("newbalanceOrig", "newbalanceDest")


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _coerce_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame().T
    if isinstance(data, dict):
        return pd.DataFrame([data])
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        if isinstance(data[0], dict):
            return pd.DataFrame(data)
    return pd.DataFrame(data)


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


@dataclass
class PreprocessorSchema:
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


class FraudPreprocessor:
    """Schema-driven preprocessing pipeline used by training and inference."""

    def __init__(self) -> None:
        self.schema_ = PreprocessorSchema([], [], [])
        self.pipeline_: ColumnTransformer | None = None
        self.is_fitted_ = False
        self.deep_numeric_columns_: list[str] = []
        self.deep_categorical_columns_: list[str] = []
        self.deep_category_maps_: dict[str, dict[str, int]] = {}
        self.deep_num_mean_: dict[str, float] = {}
        self.deep_num_std_: dict[str, float] = {}
        self.user_stats_: dict[str, dict[str, float]] = {}
        self.global_stats_: dict[str, float] = {}

    def _ensure_runtime_attrs(self) -> None:
        if not hasattr(self, "deep_numeric_columns_"):
            self.deep_numeric_columns_ = []
        if not hasattr(self, "deep_categorical_columns_"):
            self.deep_categorical_columns_ = []
        if not hasattr(self, "deep_category_maps_"):
            self.deep_category_maps_ = {}
        if not hasattr(self, "deep_num_mean_"):
            self.deep_num_mean_ = {}
        if not hasattr(self, "deep_num_std_"):
            self.deep_num_std_ = {}
        if not hasattr(self, "user_stats_"):
            self.user_stats_ = {}
        if not hasattr(self, "global_stats_"):
            self.global_stats_ = {}

    def fit(self, X: Any, y: Any = None):
        self._ensure_runtime_attrs()
        frame = self._prepare_raw_frame(X)
        if frame.empty:
            raise ValueError("Cannot fit preprocessor on an empty dataset.")

        frame = self._engineer_features(frame, fit=True)

        feature_columns = list(frame.columns)
        numeric_columns, categorical_columns = self._infer_column_types(frame)
        self.schema_ = PreprocessorSchema(feature_columns, numeric_columns, categorical_columns)

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _make_one_hot_encoder()),
            ]
        )
        self.pipeline_ = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        self.pipeline_.fit(frame)

        self.deep_numeric_columns_ = list(numeric_columns)
        self.deep_categorical_columns_ = list(categorical_columns)
        self.deep_category_maps_ = {}
        for column in self.deep_categorical_columns_:
            values = frame[column].astype("string").fillna("missing").str.lower()
            uniques = sorted(set(values.tolist()))
            self.deep_category_maps_[column] = {value: idx + 1 for idx, value in enumerate(uniques)}

        self.deep_num_mean_ = {}
        self.deep_num_std_ = {}
        for column in self.deep_numeric_columns_:
            values = _safe_numeric(frame[column]).fillna(0.0)
            mean = float(values.mean())
            std = float(values.std())
            self.deep_num_mean_[column] = mean
            self.deep_num_std_[column] = std if std > 1e-6 else 1.0

        self.is_fitted_ = True
        return self

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def transform(self, X: Any) -> np.ndarray:
        self._ensure_runtime_attrs()
        if not self.is_fitted_ or self.pipeline_ is None:
            self.fit(X)
        frame = self._prepare_frame(X)
        if frame.empty:
            return np.zeros((0, self.n_features_), dtype=np.float32)
        transformed = self.pipeline_.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=np.float32)

    def transform_frame(self, X: Any) -> pd.DataFrame:
        transformed = self.transform(X)
        if hasattr(self.pipeline_, "get_feature_names_out"):
            names = self.pipeline_.get_feature_names_out().tolist()
        else:
            names = [f"feature_{index}" for index in range(transformed.shape[1])]
        return pd.DataFrame(transformed, columns=names)

    def transform_deep_tensors(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_runtime_attrs()
        if not self.is_fitted_:
            self.fit(X)
        frame = self._prepare_frame(X)
        if frame.empty:
            return np.zeros((0, len(self.deep_numeric_columns_)), dtype=np.float32), np.zeros(
                (0, len(self.deep_categorical_columns_)), dtype=np.int64
            )

        num = np.zeros((len(frame), len(self.deep_numeric_columns_)), dtype=np.float32)
        for idx, column in enumerate(self.deep_numeric_columns_):
            values = _safe_numeric(frame[column]).fillna(0.0).to_numpy(dtype=np.float32)
            mean = self.deep_num_mean_.get(column, 0.0)
            std = self.deep_num_std_.get(column, 1.0)
            num[:, idx] = (values - mean) / std

        cat = np.zeros((len(frame), len(self.deep_categorical_columns_)), dtype=np.int64)
        for idx, column in enumerate(self.deep_categorical_columns_):
            mapping = self.deep_category_maps_.get(column, {})
            values = frame[column].astype("string").fillna("missing").str.lower().tolist()
            cat[:, idx] = np.asarray([mapping.get(value, 0) for value in values], dtype=np.int64)
        return num, cat

    @property
    def deep_category_cardinalities_(self) -> list[int]:
        cardinalities = []
        for column in self.deep_categorical_columns_:
            cardinalities.append(max(len(self.deep_category_maps_.get(column, {})) + 1, 2))
        return cardinalities

    def save(self, path: str | None = None):
        path = path or os.path.join(MODEL_DIR, "preprocessor.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)
        logger.info("Saved preprocessor -> %s", path)

    @classmethod
    def load(cls, path: str | None = None):
        path = path or os.path.join(MODEL_DIR, "preprocessor.pkl")
        if not os.path.exists(path):
            logger.warning("preprocessor.pkl not found, using empty preprocessor")
            return cls()
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        logger.info("Loaded preprocessor <- %s", path)
        return obj

    @property
    def feature_names_(self) -> list[str]:
        return list(self.schema_.feature_columns)

    @property
    def n_features_(self) -> int:
        if self.pipeline_ is None:
            return 0
        try:
            return len(self.pipeline_.get_feature_names_out())
        except Exception:
            return 0

    def _infer_column_types(self, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric_columns.append(column)
                continue
            numeric_try = pd.to_numeric(series, errors="coerce")
            coverage = float(numeric_try.notna().mean())
            if coverage >= 0.95:
                frame[column] = numeric_try
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)
        return numeric_columns, categorical_columns

    def _prepare_raw_frame(self, X: Any) -> pd.DataFrame:
        frame = _coerce_dataframe(X)
        if frame.empty:
            return frame
        for column in DROP_COLUMNS:
            if column in frame.columns:
                frame = frame.drop(columns=[column])
        frame = frame.replace({"": np.nan, "null": np.nan, "NULL": np.nan, "None": np.nan})
        frame = frame.loc[:, ~frame.columns.duplicated()]
        return frame.copy()

    def _prepare_frame(self, X: Any) -> pd.DataFrame:
        frame = self._prepare_raw_frame(X)
        if frame.empty:
            return frame

        frame = self._engineer_features(frame, fit=False)

        if self.schema_.feature_columns:
            missing_columns = [column for column in self.schema_.feature_columns if column not in frame.columns]
            if missing_columns:
                missing_frame = pd.DataFrame(index=frame.index, columns=missing_columns)
                frame = pd.concat([frame, missing_frame], axis=1)
            frame = frame.reindex(columns=self.schema_.feature_columns)

            for column in self.schema_.numeric_columns:
                if column in frame.columns:
                    frame[column] = _safe_numeric(frame[column]).fillna(0.0)

            for column in self.schema_.categorical_columns:
                if column in frame.columns:
                    series = frame[column].astype("string").fillna("missing")
                    frame[column] = series.replace({"<NA>": "missing"}).astype(str)
        return frame

    def _engineer_features(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        self._ensure_runtime_attrs()
        out = frame.copy()
        amount_col = _pick_column(out, AMOUNT_CANDIDATES)
        time_col = _pick_column(out, TIME_CANDIDATES)
        user_col = _pick_column(out, USER_ID_CANDIDATES)
        old_col = _pick_column(out, BALANCE_OLD_CANDIDATES)
        new_col = _pick_column(out, BALANCE_NEW_CANDIDATES)

        amount = _safe_numeric(out[amount_col]).fillna(0.0) if amount_col else pd.Series(0.0, index=out.index)
        out["fe_amount_log1p"] = np.log1p(np.clip(amount, a_min=0.0, a_max=None))

        if old_col and new_col:
            old_balance = _safe_numeric(out[old_col]).fillna(0.0)
            new_balance = _safe_numeric(out[new_col]).fillna(0.0)
        else:
            old_balance = amount
            new_balance = amount * 0.0
        out["fe_balance_delta"] = old_balance - new_balance
        out["fe_balance_delta_ratio"] = out["fe_balance_delta"] / (np.abs(old_balance) + 1.0)
        out["fe_amount_to_balance_ratio"] = amount / (np.abs(old_balance) + 1.0)

        if time_col:
            txn_dt = _safe_numeric(out[time_col]).fillna(0.0)
            out["fe_txn_hour"] = ((txn_dt // 3600) % 24).astype(float)
            out["fe_txn_day"] = (txn_dt // 86400).astype(float)
        else:
            txn_dt = pd.Series(0.0, index=out.index)
            out["fe_txn_hour"] = 0.0
            out["fe_txn_day"] = 0.0

        if user_col and user_col in out.columns:
            user_values = out[user_col].astype("string").fillna("missing").str.lower()
        else:
            user_values = pd.Series("global", index=out.index, dtype="string")
            out["fe_user_proxy"] = user_values

        agg = pd.DataFrame({"user": user_values, "amount": amount, "txn_dt": txn_dt}, index=out.index)
        if fit:
            grouped = agg.groupby("user", dropna=False)
            user_count = grouped["amount"].count().astype(float)
            user_mean = grouped["amount"].mean().astype(float)
            user_std = grouped["amount"].std().fillna(0.0).astype(float)
            user_last = grouped["txn_dt"].max().astype(float)
            self.user_stats_ = {
                user: {
                    "count": float(user_count.get(user, 0.0)),
                    "mean": float(user_mean.get(user, 0.0)),
                    "std": float(user_std.get(user, 0.0)),
                    "last_dt": float(user_last.get(user, 0.0)),
                }
                for user in user_count.index.tolist()
            }
            self.global_stats_ = {
                "amount_mean": float(amount.mean()),
                "amount_std": float(amount.std() if float(amount.std()) > 1e-6 else 1.0),
                "txn_dt_mean": float(txn_dt.mean()),
            }

        stats = getattr(self, "user_stats_", {})
        out["fe_user_txn_count"] = user_values.map(lambda key: stats.get(key, {}).get("count", 0.0)).astype(float)
        out["fe_user_amount_mean"] = user_values.map(lambda key: stats.get(key, {}).get("mean", self.global_stats_.get("amount_mean", 0.0))).astype(float)
        out["fe_user_amount_std"] = user_values.map(lambda key: stats.get(key, {}).get("std", self.global_stats_.get("amount_std", 1.0))).astype(float)
        out["fe_user_amount_z"] = (amount - out["fe_user_amount_mean"]) / (out["fe_user_amount_std"].abs() + 1.0)

        user_last_dt = user_values.map(lambda key: stats.get(key, {}).get("last_dt", self.global_stats_.get("txn_dt_mean", 0.0))).astype(float)
        gap = np.clip(txn_dt - user_last_dt, a_min=0.0, a_max=None)
        out["fe_time_since_last_user_txn"] = gap
        out["fe_txn_velocity_1h"] = 3600.0 / (gap + 1.0)
        out["fe_txn_velocity_24h"] = 86400.0 / (gap + 1.0)

        rolling_frame = pd.DataFrame({"user": user_values, "amount": amount, "txn_dt": txn_dt}, index=out.index)
        rolling_frame = rolling_frame.sort_values("txn_dt")
        out["fe_global_roll_mean_20"] = rolling_frame["amount"].rolling(window=20, min_periods=1).mean().reindex(out.index).fillna(amount)
        out["fe_global_roll_std_20"] = (
            rolling_frame["amount"].rolling(window=20, min_periods=1).std().reindex(out.index).fillna(0.0)
        )
        grouped_roll = rolling_frame.groupby("user")["amount"]
        out["fe_user_roll_mean_5"] = grouped_roll.rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True).reindex(out.index).fillna(amount)
        out["fe_user_roll_std_5"] = grouped_roll.rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True).reindex(out.index).fillna(0.0)

        out = out.replace([np.inf, -np.inf], np.nan)
        return out


def load_and_prepare_csv(csv_path: str):
    """Load the training CSV and return features/target."""

    df = pd.read_csv(csv_path)
    for column in list(df.columns):
        if column in {"Unnamed: 0", "Unnamed: 0.1"}:
            df = df.drop(columns=[column])

    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])

    if "isFraud" not in df.columns:
        raise ValueError("The dataset must contain an isFraud column.")

    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud"])
    logger.info("Loaded dataset with %s rows and %s feature columns", len(df), X.shape[1])
    return X, y