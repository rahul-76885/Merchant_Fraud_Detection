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

    def fit(self, X: Any, y: Any = None):
        frame = self._prepare_frame(X)
        if frame.empty:
            raise ValueError("Cannot fit preprocessor on an empty dataset.")

        feature_columns = list(frame.columns)
        numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
        categorical_columns = [column for column in feature_columns if column not in numeric_columns]

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
        self.is_fitted_ = True
        return self

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def transform(self, X: Any) -> np.ndarray:
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

    def _prepare_frame(self, X: Any) -> pd.DataFrame:
        frame = _coerce_dataframe(X)
        if frame.empty:
            return frame

        for column in DROP_COLUMNS:
            if column in frame.columns:
                frame = frame.drop(columns=[column])

        frame = frame.replace({"": np.nan, "null": np.nan, "NULL": np.nan, "None": np.nan})
        frame = frame.loc[:, ~frame.columns.duplicated()]
        frame = frame.copy()

        for column in frame.columns:
            if frame[column].dtype == object:
                try:
                    frame[column] = pd.to_numeric(frame[column])
                except (TypeError, ValueError):
                    pass

        if self.schema_.feature_columns:
            missing_columns = [column for column in self.schema_.feature_columns if column not in frame.columns]
            if missing_columns:
                missing_frame = pd.DataFrame(index=frame.index, columns=missing_columns)
                frame = pd.concat([frame, missing_frame], axis=1)
            frame = frame.reindex(columns=self.schema_.feature_columns)

            for column in self.schema_.numeric_columns:
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

            for column in self.schema_.categorical_columns:
                if column in frame.columns:
                    series = frame[column].astype("string").fillna("missing")
                    frame[column] = series.replace({"<NA>": "missing"}).astype(str)

            return frame

        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                if series.dropna().empty:
                    frame[column] = 0.0
            else:
                if series.dropna().empty:
                    frame[column] = "missing"

        return frame


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