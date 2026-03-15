# =============================================================
#  FraudShield AI  |  __init__.py
#  merchant/__init__.py
#
#  This is the Flask app factory.
#  Run with:   flask run   OR   python -m flask run
#  Set env:    FLASK_APP=merchant   FLASK_DEBUG=1
# =============================================================

from flask import Flask
import os


def create_app():
    app = Flask(__name__)

    # ── Secret key (change this to a real random string in production) ──
    app.secret_key = os.environ.get('SECRET_KEY', 'fraudshield-dev-secret-change-me')

    # ── Load the .pkl model ONCE at startup ──
    from merchant.routes import load_model
    load_model()

    # ── Register routes Blueprint ──
    from merchant.routes import bp
    app.register_blueprint(bp)

    return app