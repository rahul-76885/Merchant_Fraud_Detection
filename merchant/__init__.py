"""Merchant fraud application factory."""

from __future__ import annotations

import logging
import os
import sys

from flask import Flask

from .routes import bp, initialize_runtime


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        root.addHandler(handler)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("merchant").setLevel(level)


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder="template", static_folder="static")
    package_dir = os.path.dirname(__file__)
    default_config = {
        "SECRET_KEY": os.environ.get("SECRET_KEY", "merchant-ai-dev-secret"),
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "PERMANENT_SESSION_LIFETIME": 60 * 60 * 24 * 7,
        "JSON_SORT_KEYS": False,
        "MODEL_DIR": os.path.join(package_dir, "model"),
        "SQLITE_PATH": os.path.join(package_dir, "model", "fraud_history.sqlite3"),
        "MAX_CONTENT_LENGTH": 8 * 1024 * 1024,
    }
    app.config.update(default_config)
    if config:
        app.config.update(config)

    configure_logging(debug=app.debug)
    initialize_runtime(app)
    app.register_blueprint(bp)
    return app
