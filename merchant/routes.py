"""Flask routes for the merchant fraud dashboard."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Blueprint, abort, current_app, flash, g, jsonify, redirect, render_template, request, session, url_for

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from merchant.ensemble import get_model_bundle, load_ensemble
else:
    from .ensemble import get_model_bundle, load_ensemble

bp = Blueprint("main", __name__)
logger = logging.getLogger("merchant.routes")

DEFAULT_USERS = {
    "admin@fraudshield.ai": {
        "password": "admin123",
        "name": "Fraud Shield Admin",
        "role": "Analyst",
        "initials": "FA",
    }
}


def initialize_runtime(app):
    """Create storage and load inference artefacts during app startup."""

    with app.app_context():
        _ensure_database()
        dataset = _load_dataset()
        load_ensemble()
        logger.info("Runtime initialized | dataset_rows=%d", len(dataset))


@bp.app_context_processor
def inject_user():
    return {"current_user": _current_user()}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please sign in to continue.", "error")
            return redirect(url_for("main.index"))
        return view(*args, **kwargs)

    return wrapped


@bp.route("/", methods=["GET"])
def index():
    if session.get("logged_in"):
        return redirect(url_for("main.dashboard"))
    return render_template("login.html")


@bp.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()
    user = DEFAULT_USERS.get(email)

    if not user or user["password"] != password:
        flash("Invalid email or password.", "error")
        return redirect(url_for("main.index"))

    session.clear()
    session["logged_in"] = True
    session["email"] = email
    session["name"] = user["name"]
    session["role"] = user["role"]
    session["initials"] = user["initials"]
    session.permanent = True
    flash("Signed in successfully.", "success")
    return redirect(url_for("main.dashboard"))


@bp.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    flash("Signed out.", "success")
    return redirect(url_for("main.index"))


@bp.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    bundle = get_model_bundle()
    dataset = _load_dataset()
    recent_transactions = _fetch_recent_transactions(limit=10, bundle=bundle, dataset=dataset)
    summary = _dashboard_summary(dataset, recent_transactions)
    return render_template(
        "dashboard.html",
        recent_transactions=recent_transactions,
        dashboard_summary=summary,
        model_status=bundle.status(),
    )


@bp.route("/transaction/<txn_id>", methods=["GET"])
@login_required
def transaction_page(txn_id: str):
    bundle = get_model_bundle()
    context = _ensure_transaction_context(txn_id, bundle=bundle)
    if not context:
        abort(404, description=f"TransactionID {txn_id} not found")

    logger.info("transaction_row=%s", json.dumps(context.get("payload", {}), default=str)[:1500])
    logger.info("fraud_score=%s", context.get("fraud_score", 0))

    return render_template(
        "transaction.html",
        transaction=context,
        fraud_score=int(context.get("fraud_score", 0)),
        risk_tier=context.get("risk_tier", "SAFE"),
        merchant=context.get("merchant_stats", {}),
        model_status=bundle.status(),
    )


@bp.route("/predict", methods=["POST"])
@login_required
def predict():
    payload = _read_payload()
    if not payload:
        return jsonify({"error": "No transaction data provided."}), 400

    try:
        transaction_id = _extract_transaction_id(payload)
        if not transaction_id:
            return jsonify({"error": "TransactionID is required."}), 400

        result = _predict_transaction(payload)
        history_id = _store_prediction(transaction_id, payload, result)
        status = _ensure_transaction_status(transaction_id)
        response = _build_transaction_response(transaction_id, payload, result, history_id=history_id, status=status)
        return jsonify(response)
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


@bp.route("/history", methods=["GET"])
@login_required
def history():
    rows = _fetch_recent_transactions(limit=5)
    return jsonify(rows)


@bp.route("/search", methods=["POST"])
@login_required
def search_transaction():
    payload = _read_payload()
    transaction_id = str(payload.get("transaction_id", "")).strip()
    if not transaction_id:
        return jsonify({"error": "transaction_id is required."}), 400

    record = _ensure_transaction_context(transaction_id)
    if not record:
        return jsonify({"error": "Transaction not found"}), 404
    return redirect(url_for("main.transaction_page", txn_id=transaction_id))


@bp.route("/update_status", methods=["POST"])
@login_required
def update_status():
    payload = _read_payload()
    transaction_id = str(payload.get("transaction_id", "")).strip()
    action = str(payload.get("action", "")).strip().upper()

    if not transaction_id:
        return jsonify({"error": "transaction_id is required."}), 400

    action_map = {"BLOCK": "BLOCKED", "FLAG": "FLAGGED", "REVIEW": "REVIEW", "SAFE": "ACTIVE"}
    if action not in action_map:
        return jsonify({"error": "action must be BLOCK, FLAG, REVIEW, or SAFE."}), 400

    updated = _set_transaction_status(transaction_id, action_map[action])
    if not updated:
        return jsonify({"error": "Transaction not found"}), 404
    return jsonify(updated)


@bp.route("/health", methods=["GET"])
def health():
    bundle = get_model_bundle()
    return jsonify({"status": "ok", "models": bundle.status(), "timestamp": _utc_now()})


def _current_user() -> dict[str, Any]:
    return {
        "name": session.get("name", "Fraud Analyst"),
        "role": session.get("role", "Analyst"),
        "initials": session.get("initials", "FA"),
        "email": session.get("email", ""),
    }


def _database_path() -> str:
    return current_app.config.get(
        "SQLITE_PATH",
        os.path.join(os.path.dirname(__file__), "model", "fraud_history.sqlite3"),
    )


def _get_db() -> sqlite3.Connection:
    if "db" not in g:
        db_path = _database_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        g.db = connection
    return g.db


@bp.teardown_app_request
def _close_db(exception):
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def _ensure_database():
    connection = sqlite3.connect(_database_path())
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fraud_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                amount REAL NOT NULL,
                timestamp TEXT NOT NULL,
                fraud_score INTEGER NOT NULL,
                fraud_label TEXT NOT NULL,
                fraud_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transaction_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'REVIEW', 'FLAGGED', 'BLOCKED')),
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_fraud_history_timestamp ON fraud_history(timestamp DESC)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_fraud_history_transaction_id ON fraud_history(transaction_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_transaction_status_transaction_id ON transaction_status(transaction_id)")

        status_columns = {row[1] for row in connection.execute("PRAGMA table_info(transaction_status)").fetchall()}
        
        # Handle migration from last_updated to updated_at
        if "last_updated" in status_columns and "updated_at" not in status_columns:
            # Create a new table with correct schema
            connection.execute("""
                CREATE TABLE transaction_status_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'REVIEW', 'FLAGGED', 'BLOCKED')),
                    updated_at TEXT NOT NULL
                )
            """)
            # Copy data from old table
            connection.execute("""
                INSERT INTO transaction_status_new (id, transaction_id, status, updated_at)
                SELECT id, transaction_id, status, last_updated FROM transaction_status
            """)
            # Drop old table and rename new one
            connection.execute("DROP TABLE transaction_status")
            connection.execute("ALTER TABLE transaction_status_new RENAME TO transaction_status")
        elif "updated_at" not in status_columns:
            connection.execute("ALTER TABLE transaction_status ADD COLUMN updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP")

        connection.commit()
    finally:
        connection.close()


def _store_prediction(transaction_id: str, payload: dict[str, Any], result: dict[str, Any]) -> int:
    connection = _get_db()
    timestamp = _utc_now()
    amount = _extract_amount(payload)
    connection.execute("DELETE FROM fraud_history WHERE transaction_id = ?", (transaction_id,))
    cursor = connection.execute(
        """
        INSERT INTO fraud_history (
            transaction_id, amount, timestamp, fraud_score, fraud_label, fraud_type, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            transaction_id,
            amount,
            timestamp,
            int(result.get("fraud_score", 0)),
            str(result.get("fraud_label", "Low")),
            str(result.get("fraud_type", "Suspicious")),
            json.dumps(payload, default=str),
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def _fetch_history(limit: int = 5) -> list[dict[str, Any]]:
    connection = _get_db()
    cursor = connection.execute(
        """
        SELECT
            fh.id,
            fh.transaction_id,
            fh.amount,
            fh.timestamp,
            fh.fraud_score,
            fh.fraud_label,
            fh.fraud_type,
            COALESCE(ts.status, 'ACTIVE') AS status
        FROM fraud_history fh
        LEFT JOIN transaction_status ts ON ts.transaction_id = fh.transaction_id
        ORDER BY fh.timestamp DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = []
    for row in cursor.fetchall():
        rows.append(
            {
                "id": row["id"],
                "transaction_id": row["transaction_id"] or "",
                "amount": float(row["amount"]),
                "timestamp": row["timestamp"],
                "fraud_score": int(row["fraud_score"]),
                "fraud_label": row["fraud_label"],
                "fraud_type": row["fraud_type"],
                "score": int(row["fraud_score"]),
                "label": row["fraud_label"],
                "type": row["fraud_type"],
                "status": row["status"],
            }
        )
    return rows


def _read_payload() -> dict[str, Any]:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        return payload if isinstance(payload, dict) else {}

    form_payload = request.form.to_dict(flat=True)
    if form_payload:
        return form_payload

    raw = request.data.decode("utf-8", errors="ignore").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _extract_amount(payload: dict[str, Any]) -> float:
    for key in ("TransactionAmt", "amount", "Amount"):
        value = payload.get(key)
        if value not in (None, "", "null", "NULL", "None"):
            try:
                return float(value)
            except Exception:
                continue
    return 0.0


def _ensure_transaction_status(transaction_id: str) -> str:
    connection = _get_db()
    timestamp = _utc_now()
    connection.execute(
        """
        INSERT INTO transaction_status (transaction_id, status, updated_at)
        VALUES (?, 'ACTIVE', ?)
        ON CONFLICT(transaction_id) DO NOTHING
        """,
        (transaction_id, timestamp),
    )
    connection.commit()
    row = connection.execute(
        "SELECT status FROM transaction_status WHERE transaction_id = ?",
        (transaction_id,),
    ).fetchone()
    return (row["status"] if row else "ACTIVE")


def _set_transaction_status(
    transaction_id: str,
    status: str,
) -> dict[str, Any] | None:
    connection = _get_db()
    payload = _find_transaction_payload(transaction_id)
    if not payload:
        return None

    result = _predict_transaction(payload)
    timestamp = _utc_now()
    connection.execute(
        """
        INSERT INTO transaction_status (transaction_id, status, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(transaction_id) DO UPDATE SET
            status = excluded.status,
            updated_at = excluded.updated_at
        """,
        (transaction_id, str(status), timestamp),
    )
    connection.commit()
    return {
        "transaction_id": transaction_id,
        "amount": _extract_amount(payload),
        "fraud_score": int(result.get("fraud_score", 0)),
        "fraud_label": result.get("fraud_label", "Low"),
        "fraud_type": result.get("fraud_type", "Suspicious"),
        "status": status,
        "updated_at": timestamp,
    }


def _ensure_transaction_context(transaction_id: str, bundle=None) -> dict[str, Any] | None:
    transaction_id = str(transaction_id).strip()
    if not transaction_id:
        return None
    payload = _find_transaction_payload(transaction_id)
    if payload is None:
        return None
    result = _predict_transaction(payload, bundle=bundle)
    status = _ensure_transaction_status(transaction_id)
    return _build_transaction_context(transaction_id, payload, result, status=status)


def _build_transaction_context(
    transaction_id: str,
    payload: dict[str, Any],
    result: dict[str, Any],
    status: str = "ACTIVE",
) -> dict[str, Any]:
    safe_payload = _json_safe_dict(payload)
    amount = _extract_amount(safe_payload)
    dataset = _load_dataset()
    merchant_stats = _compute_merchant_stats(safe_payload, dataset)
    trend = _fraud_trend_indicator(safe_payload, dataset)
    risk_signals = _risky_features(safe_payload, dataset)
    risk_tier = _risk_tier(int(result.get("fraud_score", 0)))
    response = dict(result)
    response.update(safe_payload)
    response.update(
        {
            "transaction_id": transaction_id,
            "amount": amount,
            "status": status,
            "payload": safe_payload,
            "risk_tier": risk_tier,
            "merchant_stats": merchant_stats,
            "trend": trend,
            "risky_features": risk_signals,
        }
    )
    return response


def _build_transaction_response(
    transaction_id: str,
    payload: dict[str, Any],
    result: dict[str, Any],
    history_id: int | None = None,
    status: str = "ACTIVE",
) -> dict[str, Any]:
    response = _build_transaction_context(transaction_id, payload, result, status=status)
    if history_id is not None:
        response["history_id"] = history_id
    response["status"] = status
    return response


def _history_row_to_dashboard_record(row: dict[str, Any], status: str = "ACTIVE") -> dict[str, Any]:
    fraud_score = int(row.get("fraud_score", 0))
    return {
        "transaction_id": str(row.get("TransactionID", "")),
        "amount": float(row.get("TransactionAmt") or 0.0),
        "timestamp": row.get("TransactionDT", "-"),
        "fraud_score": fraud_score,
        "fraud_label": _risk_label(fraud_score),
        "fraud_type": row.get("fraud_type", "Suspicious"),
        "status": status,
        "TransactionAmt": float(row.get("TransactionAmt") or 0.0),
        "ProductCD": row.get("ProductCD", "-"),
        "card_network": row.get("card_network", "-"),
        "isFraud": int(row.get("isFraud", 0) or 0),
        "fraud_prob": fraud_score,
        "risk_tier": _risk_tier(fraud_score),
        "trend": row.get("trend", "Stable"),
    }


def _predict_transaction(payload: dict[str, Any], bundle=None) -> dict[str, Any]:
    model_bundle = bundle or get_model_bundle()
    # Keep preprocessing stable by replacing null-like values before transformation.
    frame = pd.DataFrame([payload]).fillna(0)
    normalized = frame.to_dict(orient="records")[0]
    result = model_bundle.predict_transaction(normalized)

    score = int(result.get("fraud_score", 0))
    risk_tier = _risk_tier(score)
    result["fraud_label"] = _risk_label(score)
    result["risk_tier"] = risk_tier
    result["risk_color"] = "red" if risk_tier == "FRAUD" else "yellow" if risk_tier == "SUSPICIOUS" else "green"
    return result


def _fetch_recent_transactions(
    limit: int = 10,
    bundle=None,
    dataset: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    df = dataset.copy() if dataset is not None else _load_dataset().copy()
    if "TransactionDT" in df.columns:
        df = df.sort_values(by="TransactionDT", ascending=False)
    else:
        df = df.iloc[::-1]
    sample = df.head(int(limit))

    status_map = _fetch_status_map([str(value) for value in sample["TransactionID"].astype(str).tolist()])
    rows: list[dict[str, Any]] = []
    for _, raw_row in sample.iterrows():
        payload = raw_row.to_dict()
        payload["TransactionID"] = str(payload.get("TransactionID", ""))
        prediction = _predict_transaction(payload, bundle=bundle)
        score = int(prediction.get("fraud_score", 0))

        record = dict(payload)
        record["fraud_score"] = score
        record["trend"] = _fraud_trend_indicator(payload, _load_dataset()).get("direction", "Stable")
        tx_id = str(payload.get("TransactionID", ""))
        rows.append(_history_row_to_dashboard_record(record, status=status_map.get(tx_id, "ACTIVE")))
    return rows


def _fetch_status_map(transaction_ids: list[str]) -> dict[str, str]:
    if not transaction_ids:
        return {}
    placeholders = ",".join(["?"] * len(transaction_ids))
    connection = _get_db()
    cursor = connection.execute(
        f"SELECT transaction_id, status FROM transaction_status WHERE transaction_id IN ({placeholders})",
        transaction_ids,
    )
    return {str(row["transaction_id"]): str(row["status"]) for row in cursor.fetchall()}


def _dashboard_summary(dataset: pd.DataFrame, recent_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if dataset.empty:
        return {
            "total_transactions": 0,
            "fraud_count": 0,
            "suspicious_count": 0,
            "average_score": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "fraud_percent": 0,
        }

    total = int(len(dataset))
    fraud_count = int(pd.to_numeric(dataset.get("isFraud", 0), errors="coerce").fillna(0).astype(int).eq(1).sum())
    scores = [float(row.get("fraud_score", 0)) for row in recent_rows] or [0.0]
    suspicious_count = sum(1 for score in scores if 40 <= int(score) <= 70)

    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "suspicious_count": suspicious_count,
        "average_score": round(sum(scores) / len(scores), 1),
        "high_risk": sum(1 for score in scores if int(score) > 70),
        "medium_risk": suspicious_count,
        "low_risk": sum(1 for score in scores if int(score) < 40),
        "fraud_percent": round((fraud_count / total) * 100, 2) if total else 0,
    }


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    dataset_path = Path(__file__).resolve().parents[1] / "dataset" / "new_Df.csv"
    df = pd.read_csv(dataset_path)
    if "TransactionID" not in df.columns:
        raise ValueError("Dataset must include TransactionID")
    if "isFraud" in df.columns:
        df["isFraud"] = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)
    return df


def _find_transaction_payload(transaction_id: str) -> dict[str, Any] | None:
    dataset = _load_dataset()
    candidate = dataset[dataset["TransactionID"].astype(str) == str(transaction_id)]
    if candidate.empty:
        return None
    row = candidate.iloc[0].drop(labels=["TransactionID"], errors="ignore")
    payload = row.to_dict()
    payload["TransactionID"] = str(transaction_id)
    return _json_safe_dict(payload)


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _json_safe_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(value) for key, value in payload.items()}


def _risk_tier(score: int) -> str:
    if score > 70:
        return "FRAUD"
    if score >= 40:
        return "SUSPICIOUS"
    return "SAFE"


def _risk_label(score: int) -> str:
    tier = _risk_tier(score)
    if tier == "FRAUD":
        return "High"
    if tier == "SUSPICIOUS":
        return "Medium"
    return "Low"


def _compute_merchant_stats(payload: dict[str, Any], dataset: pd.DataFrame) -> dict[str, Any]:
    card_id = str(payload.get("card_id", "")).strip()
    if not card_id:
        return {
            "card_id": "-",
            "total_transactions": 0,
            "fraud_transactions": 0,
            "fraud_rate_pct": 0.0,
            "avg_transaction_amount": 0.0,
            "most_used_product": "-",
            "most_used_network": "-",
            "historical_risk_score": 0,
        }

    group = dataset[dataset["card_id"].astype(str) == card_id]
    total = int(len(group))
    fraud_tx = int(pd.to_numeric(group.get("isFraud", 0), errors="coerce").fillna(0).astype(int).eq(1).sum())
    fraud_rate = round((fraud_tx / total) * 100, 2) if total else 0.0

    amt_series = pd.to_numeric(group.get("TransactionAmt", 0), errors="coerce").fillna(0.0)
    avg_amount = round(float(amt_series.mean()), 2) if total else 0.0

    product_mode = "-"
    if "ProductCD" in group.columns and not group["ProductCD"].dropna().empty:
        product_mode = str(group["ProductCD"].mode().iloc[0])

    network_mode = "-"
    if "card_network" in group.columns and not group["card_network"].dropna().empty:
        network_mode = str(group["card_network"].mode().iloc[0])

    risk_score = int(round(min(100.0, fraud_rate * 1.2 + (10 if total < 5 and fraud_tx > 0 else 0))))

    return {
        "card_id": card_id,
        "total_transactions": total,
        "fraud_transactions": fraud_tx,
        "fraud_rate_pct": fraud_rate,
        "avg_transaction_amount": avg_amount,
        "most_used_product": product_mode,
        "most_used_network": network_mode,
        "historical_risk_score": risk_score,
    }


def _fraud_trend_indicator(payload: dict[str, Any], dataset: pd.DataFrame) -> dict[str, Any]:
    card_id = str(payload.get("card_id", "")).strip()
    if not card_id:
        return {"direction": "Stable", "delta_pct": 0.0}

    group = dataset[dataset["card_id"].astype(str) == card_id].copy()
    if group.empty or "TransactionDT" not in group.columns:
        return {"direction": "Stable", "delta_pct": 0.0}

    group["TransactionDT"] = pd.to_numeric(group["TransactionDT"], errors="coerce").fillna(0)
    group = group.sort_values("TransactionDT")
    cutoff = max(int(len(group) * 0.5), 1)
    early = group.iloc[:cutoff]
    recent = group.iloc[cutoff:]
    if recent.empty:
        recent = early

    early_rate = pd.to_numeric(early.get("isFraud", 0), errors="coerce").fillna(0).astype(int).mean() * 100
    recent_rate = pd.to_numeric(recent.get("isFraud", 0), errors="coerce").fillna(0).astype(int).mean() * 100
    delta = round(float(recent_rate - early_rate), 2)

    if delta > 3:
        direction = "Rising"
    elif delta < -3:
        direction = "Improving"
    else:
        direction = "Stable"
    return {"direction": direction, "delta_pct": delta}


def _risky_features(payload: dict[str, Any], dataset: pd.DataFrame) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []

    amount = float(pd.to_numeric(pd.Series([payload.get("TransactionAmt", 0)]), errors="coerce").fillna(0).iloc[0])
    amount_series = pd.to_numeric(dataset.get("TransactionAmt", 0), errors="coerce").fillna(0)
    amount_threshold = float(amount_series.quantile(0.95)) if not amount_series.empty else 0.0
    if amount >= amount_threshold and amount_threshold > 0:
        signals.append({"key": "high_amount", "label": "High amount", "value": round(amount, 2), "severity": "high"})

    product = str(payload.get("ProductCD", "")).strip()
    if product and "ProductCD" in dataset.columns:
        product_freq = dataset["ProductCD"].astype(str).value_counts(normalize=True)
        freq = float(product_freq.get(product, 0.0))
        if freq < 0.05:
            signals.append({"key": "rare_product", "label": "Rare ProductCD", "value": product, "severity": "medium"})

    country = str(payload.get("addr1") or payload.get("country") or "").strip()
    if country:
        country_col = None
        if "country" in dataset.columns:
            country_col = "country"
        elif "addr1" in dataset.columns:
            country_col = "addr1"
        if country_col:
            country_freq = dataset[country_col].astype(str).value_counts(normalize=True)
            if float(country_freq.get(country, 0.0)) < 0.03:
                signals.append({"key": "unusual_country", "label": "Unusual country/region", "value": country, "severity": "medium"})

    if not signals:
        signals.append({"key": "none", "label": "No high-risk outlier detected", "value": "Baseline behavior", "severity": "low"})
    return signals


def _extract_transaction_id(payload: dict[str, Any]) -> str:
    for key in ("TransactionID", "transaction_id", "txn_id"):
        value = payload.get(key)
        if value not in (None, "", "null", "NULL", "None"):
            return str(value).strip()
    return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    from merchant import create_app

    app = create_app()
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")