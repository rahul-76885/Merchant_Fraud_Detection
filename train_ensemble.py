"""Train advanced fraud ensemble models and preprocessing artifacts."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from merchant.dl_models import (
    DNNFraudModel,
    FTTransformerFraudModel,
    save_torch_checkpoint,
    torch_available,
)
from merchant.preprocessing import FraudPreprocessor

try:
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover - optional dependency
    SMOTE = None

if torch_available():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "merchant", "model")
logger = logging.getLogger("merchant.train")

TARGET_CANDIDATES = (
    "isFraud",
    "fraud",
    "target",
    "label",
    "class",
    "is_fraud",
)
DROP_COLUMNS = {"TransactionID", "Unnamed: 0", "Unnamed: 0.1"}


if torch_available():
    class FocalLoss(torch.nn.Module):
        def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, logits, targets):
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            pt = torch.where(targets == 1, probs, 1 - probs)
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * ((1 - pt) ** self.gamma) * bce
            return loss.mean()
else:
    class FocalLoss:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FocalLoss requires torch")


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def maybe_sample(X: pd.DataFrame, y: pd.Series, max_rows: int | None) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y
    sampled = X.sample(n=max_rows, random_state=42)
    return sampled, y.loc[sampled.index]


def detect_target_column(df: pd.DataFrame) -> str:
    by_name = {column.lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in by_name:
            return by_name[candidate.lower()]
    raise ValueError(f"Could not detect target column. Expected one of {TARGET_CANDIDATES}")


def coerce_binary_target(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    uniq = sorted(values.dropna().unique().tolist())
    if len(uniq) != 2:
        raise ValueError(f"Target must be binary, got {uniq}")
    lo, hi = uniq[0], uniq[1]
    return values.map({lo: 0, hi: 1}).astype(int)


def load_training_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    for column in list(df.columns):
        if column in DROP_COLUMNS:
            df = df.drop(columns=[column])
    target_column = detect_target_column(df)
    y = coerce_binary_target(df[target_column])
    X = df.drop(columns=[target_column])
    logger.warning("Loaded dataset with %s rows, %s features, target=%s", len(df), X.shape[1], target_column)
    return X, y


def metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    logger.warning(
        "%s | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f",
        name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )


def print_classification_details(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = metrics_dict(y_true, y_prob, threshold=threshold)
    print_metrics(name, metrics)
    logger.warning("%s classification report\n%s", name, classification_report(y_true, y_pred, digits=4))
    logger.warning("%s confusion matrix\n%s", name, confusion_matrix(y_true, y_pred))
    return metrics


def print_summary_metrics_table(summary_metrics: dict[str, dict[str, float]]) -> None:
    if not summary_metrics:
        return

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    ordered_rows = []
    for model_name, metrics in summary_metrics.items():
        ordered_rows.append(
            [
                model_name,
                f"{metrics.get('accuracy', 0.0):.4f}",
                f"{metrics.get('precision', 0.0):.4f}",
                f"{metrics.get('recall', 0.0):.4f}",
                f"{metrics.get('f1', 0.0):.4f}",
                f"{metrics.get('roc_auc', 0.0):.4f}",
            ]
        )

    widths = [len(col) for col in headers]
    for row in ordered_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _fmt(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_fmt(headers), separator]
    lines.extend(_fmt(row) for row in ordered_rows)
    logger.warning("Classification Summary Table\n%s", "\n".join(lines))


def build_xgb(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        min_child_weight=2,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )


def build_lgb() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def maybe_smote(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if SMOTE is None:
        return X, y
    fraud_rate = float(np.mean(y))
    if 0.35 <= fraud_rate <= 0.65:
        return X, y
    smote = SMOTE(random_state=42)
    x_res, y_res = smote.fit_resample(X, y)
    x_res = pd.DataFrame(x_res, columns=X.columns)
    return x_res, y_res


def train_torch_binary_model(
    model,
    x_num_train: np.ndarray,
    x_cat_train: np.ndarray,
    y_train: np.ndarray,
    x_num_val: np.ndarray,
    x_cat_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    use_focal: bool,
    device,
) -> np.ndarray:
    train_ds = TensorDataset(
        torch.from_numpy(x_num_train.astype(np.float32)),
        torch.from_numpy(x_cat_train.astype(np.int64)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    if use_focal:
        criterion = FocalLoss(gamma=2.0, alpha=0.75)
    else:
        pos = max(float((y_train == 1).sum()), 1.0)
        neg = max(float((y_train == 0).sum()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_auc = -1.0
    x_num_val_t = torch.from_numpy(x_num_val.astype(np.float32)).to(device)
    x_cat_val_t = torch.from_numpy(x_cat_val.astype(np.int64)).to(device)

    for _ in range(epochs):
        model.train()
        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_num, xb_cat)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(x_num_val_t, x_cat_val_t)).detach().cpu().numpy()
        try:
            val_auc = roc_auc_score(y_val, val_prob)
        except Exception:
            val_auc = 0.0
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(x_num_val_t, x_cat_val_t)).detach().cpu().numpy()
    return val_prob


def train_torch_final(
    model,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    use_focal: bool,
    device,
) -> Any:
    ds = TensorDataset(
        torch.from_numpy(x_num.astype(np.float32)),
        torch.from_numpy(x_cat.astype(np.int64)),
        torch.from_numpy(y.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    if use_focal:
        criterion = FocalLoss(gamma=2.0, alpha=0.75)
    else:
        pos = max(float((y == 1).sum()), 1.0)
        neg = max(float((y == 0).sum()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(epochs):
        model.train()
        for xb_num, xb_cat, yb in loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_num, xb_cat)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_torch_model(model, x_num: np.ndarray, x_cat: np.ndarray, device) -> np.ndarray:
    x_num_t = torch.from_numpy(x_num.astype(np.float32)).to(device)
    x_cat_t = torch.from_numpy(x_cat.astype(np.int64)).to(device)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x_num_t, x_cat_t)).detach().cpu().numpy()


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    positive = {k: max(float(v), 0.0) for k, v in raw.items()}
    total = sum(positive.values())
    if total <= 0:
        return {k: 1.0 / max(len(positive), 1) for k in positive}
    return {k: v / total for k, v in positive.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train advanced merchant fraud stack")
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, "dataset", "new_Df.csv"), help="Training CSV path")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for quicker local validation")
    parser.add_argument("--folds", type=int, default=5, help="Stratified K-Fold count")
    parser.add_argument("--epochs", type=int, default=8, help="Neural model epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Neural model batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Neural model learning rate")
    parser.add_argument("--use-focal-loss", action="store_true", help="Use focal loss for neural models")
    parser.add_argument("--disable-anomaly", action="store_true", help="Disable IsolationForest anomaly model")
    args = parser.parse_args()

    configure_logging()
    ensure_model_dir()

    X, y = load_training_data(args.csv)
    X, y = maybe_sample(X, y, args.max_rows if args.max_rows > 0 else None)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    base_preprocessor = FraudPreprocessor().fit(X_train, y_train)
    X_train_tree = base_preprocessor.transform_frame(X_train)
    X_test_tree = base_preprocessor.transform_frame(X_test)
    X_train_num, X_train_cat = base_preprocessor.transform_deep_tensors(X_train)
    X_test_num, X_test_cat = base_preprocessor.transform_deep_tensors(X_test)

    logger.warning("Tree features=%d | deep_num=%d deep_cat=%d", X_train_tree.shape[1], X_train_num.shape[1], X_train_cat.shape[1])

    device = torch.device("cuda" if torch_available() and torch.cuda.is_available() else "cpu") if torch_available() else None
    if torch_available():
        logger.warning("PyTorch device: %s", device)
    else:
        logger.warning("PyTorch not available. DNN/Transformer stages will be skipped.")

    skf = StratifiedKFold(n_splits=max(args.folds, 3), shuffle=True, random_state=42)
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    model_names = ["xgboost", "lightgbm"]
    if torch_available():
        model_names.extend(["ann", "transformer"])
    oof = {name: np.zeros(len(X_train), dtype=np.float32) for name in model_names}
    cv_scores: dict[str, list[float]] = {name: [] for name in model_names}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train_np), start=1):
        logger.warning("CV fold %d/%d", fold_idx, skf.n_splits)
        X_tr, y_tr = X_train.iloc[tr_idx], y_train_np[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train_np[va_idx]

        fold_pre = FraudPreprocessor().fit(X_tr, y_tr)
        x_tr_tree = fold_pre.transform_frame(X_tr)
        x_va_tree = fold_pre.transform_frame(X_va)
        x_tr_tree_res, y_tr_res = maybe_smote(x_tr_tree, y_tr)

        pos = max(float((y_tr_res == 1).sum()), 1.0)
        neg = max(float((y_tr_res == 0).sum()), 1.0)
        spw = neg / pos

        xgb = build_xgb(scale_pos_weight=spw)
        xgb.fit(x_tr_tree_res, y_tr_res)
        xgb_va = xgb.predict_proba(x_va_tree)[:, 1]
        oof["xgboost"][va_idx] = xgb_va
        cv_scores["xgboost"].append(average_precision_score(y_va, xgb_va))

        lgb = build_lgb()
        lgb.fit(x_tr_tree_res, y_tr_res)
        lgb_va = lgb.predict_proba(x_va_tree)[:, 1]
        oof["lightgbm"][va_idx] = lgb_va
        cv_scores["lightgbm"].append(average_precision_score(y_va, lgb_va))

        if torch_available():
            x_tr_num, x_tr_cat = fold_pre.transform_deep_tensors(X_tr)
            x_va_num, x_va_cat = fold_pre.transform_deep_tensors(X_va)
            cat_cards = fold_pre.deep_category_cardinalities_

            dnn = DNNFraudModel(
                num_features=x_tr_num.shape[1],
                cat_cardinalities=cat_cards,
                hidden_dims=[256, 128, 64],
                dropout=0.25,
            )
            dnn_va = train_torch_binary_model(
                dnn,
                x_tr_num,
                x_tr_cat,
                y_tr,
                x_va_num,
                x_va_cat,
                y_va,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                use_focal=args.use_focal_loss,
                device=device,
            )
            oof["ann"][va_idx] = dnn_va
            cv_scores["ann"].append(average_precision_score(y_va, dnn_va))

            transformer = FTTransformerFraudModel(
                num_features=x_tr_num.shape[1],
                cat_cardinalities=cat_cards,
                d_model=64,
                n_heads=4,
                n_layers=2,
                dropout=0.2,
            )
            tfm_va = train_torch_binary_model(
                transformer,
                x_tr_num,
                x_tr_cat,
                y_tr,
                x_va_num,
                x_va_cat,
                y_va,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                use_focal=args.use_focal_loss,
                device=device,
            )
            oof["transformer"][va_idx] = tfm_va
            cv_scores["transformer"].append(average_precision_score(y_va, tfm_va))

    logger.warning("Cross-validation PR-AUC summary")
    for name in model_names:
        mean_pr = float(np.mean(cv_scores[name])) if cv_scores[name] else 0.0
        oof_metrics = metrics_dict(y_train_np, oof[name])
        logger.warning("%s CV mean_pr_auc=%.4f", name, mean_pr)
        print_metrics(f"{name} OOF", oof_metrics)

    raw_weights = {name: max(float(np.mean(cv_scores[name])) if cv_scores[name] else 0.0, 1e-4) for name in model_names}
    weights = normalize_weights(raw_weights)
    logger.warning("Ensemble weights from CV PR-AUC: %s", {k: round(v, 4) for k, v in weights.items()})

    pos = max(float((y_train_np == 1).sum()), 1.0)
    neg = max(float((y_train_np == 0).sum()), 1.0)
    spw = neg / pos

    xgb_final = build_xgb(scale_pos_weight=spw)
    xgb_final.fit(*maybe_smote(X_train_tree, y_train_np))
    lgb_final = build_lgb()
    lgb_final.fit(*maybe_smote(X_train_tree, y_train_np))

    probs_test: dict[str, np.ndarray] = {
        "xgboost": xgb_final.predict_proba(X_test_tree)[:, 1],
        "lightgbm": lgb_final.predict_proba(X_test_tree)[:, 1],
    }

    if torch_available():
        cat_cards = base_preprocessor.deep_category_cardinalities_
        dnn_final = train_torch_final(
            DNNFraudModel(X_train_num.shape[1], cat_cards, hidden_dims=[256, 128, 64], dropout=0.25),
            X_train_num,
            X_train_cat,
            y_train_np,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_focal=args.use_focal_loss,
            device=device,
        )
        tfm_final = train_torch_final(
            FTTransformerFraudModel(X_train_num.shape[1], cat_cards, d_model=64, n_heads=4, n_layers=2, dropout=0.2),
            X_train_num,
            X_train_cat,
            y_train_np,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_focal=args.use_focal_loss,
            device=device,
        )
        probs_test["ann"] = predict_torch_model(dnn_final, X_test_num, X_test_cat, device)
        probs_test["transformer"] = predict_torch_model(tfm_final, X_test_num, X_test_cat, device)
    else:
        dnn_final = None
        tfm_final = None

    anomaly_model = None
    anomaly_prob = np.zeros_like(y_test_np, dtype=np.float32)
    if not args.disable_anomaly:
        anomaly_model = IsolationForest(
            n_estimators=300,
            contamination=0.03,
            random_state=42,
            n_jobs=-1,
        )
        anomaly_model.fit(X_train_tree)
        anomaly_score = anomaly_model.decision_function(X_test_tree)
        anomaly_prob = 1.0 / (1.0 + np.exp(4.0 * anomaly_score))
        probs_test["anomaly"] = anomaly_prob

    summary_metrics: dict[str, dict[str, float]] = {}
    for name, prob in probs_test.items():
        model_metrics = print_classification_details(name, y_test_np, prob)
        if name in model_names:
            summary_metrics[name] = model_metrics

    base_model_keys = [key for key in probs_test if key != "anomaly"]
    soft_voting_prob = np.mean(np.column_stack([probs_test[key] for key in base_model_keys]), axis=1)
    print_classification_details("Soft Voting Ensemble", y_test_np, soft_voting_prob)

    weighted_prob = np.zeros(len(y_test_np), dtype=np.float32)
    weight_sum = 0.0
    for name, prob in probs_test.items():
        if name == "anomaly":
            continue
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        weighted_prob += prob * float(w)
        weight_sum += float(w)
    if weight_sum <= 0:
        weighted_prob = soft_voting_prob
    else:
        weighted_prob = weighted_prob / weight_sum

    print_classification_details("Weighted Voting Ensemble", y_test_np, weighted_prob)

    ensemble_prob = weighted_prob
    if "anomaly" in probs_test:
        ensemble_prob = 0.9 * weighted_prob + 0.1 * anomaly_prob

    ensemble_metrics = print_classification_details("Final Ensemble", y_test_np, ensemble_prob)
    summary_metrics["overall_ensemble"] = ensemble_metrics
    print_summary_metrics_table(summary_metrics)

    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "wb") as handle:
        pickle.dump(xgb_final, handle)
    with open(os.path.join(MODEL_DIR, "xgboost.pkl"), "wb") as handle:
        pickle.dump(xgb_final, handle)
    with open(os.path.join(MODEL_DIR, "lgb_model.pkl"), "wb") as handle:
        pickle.dump(lgb_final, handle)
    with open(os.path.join(MODEL_DIR, "lightgbm.pkl"), "wb") as handle:
        pickle.dump(lgb_final, handle)
    if anomaly_model is not None:
        with open(os.path.join(MODEL_DIR, "anomaly.pkl"), "wb") as handle:
            pickle.dump(anomaly_model, handle)

    if torch_available() and dnn_final is not None and tfm_final is not None:
        save_torch_checkpoint(
            "dnn_model.pt",
            {
                "state_dict": dnn_final.cpu().state_dict(),
                "num_features": int(X_train_num.shape[1]),
                "cat_cardinalities": base_preprocessor.deep_category_cardinalities_,
                "hidden_dims": [256, 128, 64],
                "dropout": 0.25,
            },
        )
        save_torch_checkpoint(
            "transformer_model.pt",
            {
                "state_dict": tfm_final.cpu().state_dict(),
                "num_features": int(X_train_num.shape[1]),
                "cat_cardinalities": base_preprocessor.deep_category_cardinalities_,
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "dropout": 0.2,
            },
        )

    weights_to_save = {
        "xgboost": float(weights.get("xgboost", 0.0)),
        "lightgbm": float(weights.get("lightgbm", 0.0)),
        "ann": float(weights.get("ann", 0.0)),
        "transformer": float(weights.get("transformer", 0.0)),
        "anomaly": 0.1 if anomaly_model is not None else 0.0,
    }
    with open(os.path.join(MODEL_DIR, "ensemble.pkl"), "wb") as handle:
        pickle.dump(normalize_weights(weights_to_save), handle)

    base_preprocessor.save(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    logger.warning("Training completed. Final ensemble ROC-AUC=%.4f PR-AUC=%.4f", ensemble_metrics["roc_auc"], ensemble_metrics["pr_auc"])


if __name__ == "__main__":
    main()