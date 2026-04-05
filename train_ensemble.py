"""Train fraud models and preprocessing artifacts from dataset/new_Df.csv."""

from __future__ import annotations

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from merchant.preprocessing import FraudPreprocessor

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "merchant", "model")
logger = logging.getLogger("merchant.train")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


TARGET_CANDIDATES = (
    "isFraud",
    "fraud",
    "target",
    "label",
    "class",
    "is_fraud",
)
DROP_COLUMNS = {"TransactionID", "Unnamed: 0", "Unnamed: 0.1"}


def maybe_sample(X: pd.DataFrame, y: pd.Series, max_rows: int | None) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y
    sampled = X.sample(n=max_rows, random_state=42)
    return sampled, y.loc[sampled.index]


def metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    logger.info(
        "%s | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )


def detect_target_column(df: pd.DataFrame) -> str:
    by_name = {column.lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in by_name:
            return by_name[candidate.lower()]

    binary_candidates = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        unique_count = int(series.nunique())
        if unique_count == 2:
            binary_candidates.append(column)

    if len(binary_candidates) == 1:
        logger.warning("Target column not explicitly named. Using inferred binary target: %s", binary_candidates[0])
        return binary_candidates[0]

    raise ValueError(
        "Could not detect target column. Expected one of "
        f"{TARGET_CANDIDATES} or a single binary column."
    )


def coerce_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        full_values = pd.to_numeric(series, errors="coerce")
    else:
        mapped = (
            series.astype("string")
            .str.strip()
            .str.lower()
            .replace(
                {
                    "true": "1",
                    "yes": "1",
                    "fraud": "1",
                    "false": "0",
                    "no": "0",
                    "legit": "0",
                    "non-fraud": "0",
                    "nonfraud": "0",
                }
            )
        )
        full_values = pd.to_numeric(mapped, errors="coerce")

    values = full_values.dropna()
    unique = sorted(values.unique().tolist())
    if len(unique) != 2:
        raise ValueError(f"Target must be binary after coercion, got values={unique}")

    lo, hi = unique[0], unique[1]
    binary = full_values.map({lo: 0, hi: 1})
    if binary.isna().any():
        raise ValueError("Target contains values that could not be mapped to binary classes.")
    return binary.astype(int)


def load_training_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    for column in list(df.columns):
        if column in DROP_COLUMNS:
            df = df.drop(columns=[column])

    target_column = detect_target_column(df)
    y = coerce_binary_target(df[target_column])
    X = df.drop(columns=[target_column])
    logger.info("Loaded dataset with %s rows, %s features, target=%s", len(df), X.shape[1], target_column)
    return X, y


def log_feature_importance(model_name: str, model: object, top_k: int = 10) -> None:
    scores = getattr(model, "feature_importances_", None)
    if scores is None:
        logger.info("%s has no feature_importances_ attribute", model_name)
        return
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        logger.info("%s feature importances are empty", model_name)
        return
    top_idx = np.argsort(arr)[::-1][:top_k]
    formatted = ", ".join([f"f{idx}={arr[idx]:.6f}" for idx in top_idx])
    logger.info("%s top %d feature importances: %s", model_name, top_k, formatted)


def save_weights() -> None:
    weights_path = os.path.join(MODEL_DIR, "ensemble.pkl")
    with open(weights_path, "wb") as handle:
        pickle.dump({"xgboost": 0.5, "lightgbm": 0.5}, handle)
    logger.info("Saved ensemble weights -> %s", weights_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the simplified merchant fraud stack")
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, "dataset", "new_Df.csv"), help="Training CSV path")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for quicker local validation")
    args = parser.parse_args()

    configure_logging()
    ensure_model_dir()

    logger.info("Loading dataset from %s", args.csv)
    X, y = load_training_data(args.csv)
    X, y = maybe_sample(X, y, args.max_rows if args.max_rows > 0 else None)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info("Fitting preprocessor")
    preprocessor = FraudPreprocessor().fit(X_train, y_train)
    X_train_proc = preprocessor.transform_frame(X_train)
    X_test_proc = preprocessor.transform_frame(X_test)
    preprocessor.save(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    logger.info("Preprocessed feature count: %d", X_train_proc.shape[1])

    positive = float(max(int((y_train == 1).sum()), 1))
    negative = float(max(int((y_train == 0).sum()), 1))
    scale_pos_weight = negative / positive

    logger.info("Training XGBoost")
    xgb_model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_weight=1,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_proc, y_train)
    xgb_prob = xgb_model.predict_proba(X_test_proc)[:, 1]
    print_metrics("XGBoost", metrics_dict(y_test.to_numpy(), xgb_prob))
    log_feature_importance("XGBoost", xgb_model)

    logger.info("Training LightGBM")
    lgb_model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train_proc, y_train)
    lgb_prob = lgb_model.predict_proba(X_test_proc)[:, 1]
    print_metrics("LightGBM", metrics_dict(y_test.to_numpy(), lgb_prob))
    log_feature_importance("LightGBM", lgb_model)

    ensemble_prob = 0.5 * xgb_prob + 0.5 * lgb_prob
    ensemble_metrics = metrics_dict(y_test.to_numpy(), ensemble_prob)
    print_metrics("Ensemble", ensemble_metrics)

    xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    lgb_path = os.path.join(MODEL_DIR, "lgb_model.pkl")
    with open(xgb_path, "wb") as handle:
        pickle.dump(xgb_model, handle)
    with open(lgb_path, "wb") as handle:
        pickle.dump(lgb_model, handle)

    save_weights()

    logger.info("Saved XGBoost model -> %s", xgb_path)
    logger.info("Saved LightGBM model -> %s", lgb_path)
    logger.info(
        "Done | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        ensemble_metrics["accuracy"],
        ensemble_metrics["precision"],
        ensemble_metrics["recall"],
        ensemble_metrics["f1"],
        ensemble_metrics["roc_auc"],
    )


if __name__ == "__main__":
    main()