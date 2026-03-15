# =============================================================
#  FraudShield AI  |  routes.py
#  merchant/routes.py
#
#  HOW IT WORKS:
#  1. App starts → load_model() loads your .pkl file once
#  2. User visits / → sees login.html (landing page)
#  3. POST /login → checks credentials → redirect to /dashboard
#  4. /dashboard → shows KPIs, recent transactions, fraud alerts
#  5. /search?txn_id=X → fetches transaction, runs .pkl model,
#                         returns result back into dashboard.html
#  6. /merchant/<id> → fetches merchant + transactions,
#                       runs .pkl model, renders merchant.html
#  7. POST /merchant/<id>/action → block / refund / alert
#  8. /logout → clears session → back to login
# =============================================================

from flask import (
    Blueprint, render_template, request,
    redirect, url_for, session, flash, jsonify
)
import pickle
import os
import pandas as pd
import numpy as np
from functools import wraps
from datetime import datetime

# ── Blueprint ──────────────────────────────────────────────────
bp = Blueprint('main', __name__)

# ── Hardcoded login (replace with DB later) ────────────────────
ANALYST_EMAIL    = "analyst@fraudshield.ai"
ANALYST_PASSWORD = "FraudShield2025"

# ── .pkl model — loaded once when app starts ───────────────────
MODEL = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'fraud_model.pkl')

def load_model():
    """
    Call this from your app factory (create_app) ONCE at startup.
    Example in __init__.py:
        from merchant.routes import load_model
        load_model()
    """
    global MODEL
    try:
        with open(MODEL_PATH, 'rb') as f:
            MODEL = pickle.load(f)
        print(f"[FraudShield] .pkl model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[FraudShield] WARNING: model not found at {MODEL_PATH}. Using dummy scores.")
        MODEL = None

def predict_fraud(features_dict: dict) -> dict:
    """
    Run the loaded .pkl model on a single transaction.

    Args:
        features_dict: dict of column_name → value
                       matching your training columns

    Returns:
        {
          'is_fraud':    0 or 1,
          'fraud_prob':  float 0–100 (e.g. 87.4),
          'risk_tier':   'SAFE' | 'SUSPICIOUS' | 'FRAUD'
        }
    """
    if MODEL is None:
        # Dummy output until your .pkl is in place
        import random
        prob = round(random.uniform(20, 95), 1)
        return {
            'is_fraud':  1 if prob >= 70 else 0,
            'fraud_prob': prob,
            'risk_tier': 'FRAUD' if prob >= 70 else ('SUSPICIOUS' if prob >= 40 else 'SAFE')
        }

    try:
        # Build a single-row DataFrame matching your training columns
        df = pd.DataFrame([features_dict])

        # predict_proba returns [[prob_class0, prob_class1]]
        prob = float(MODEL.predict_proba(df)[0][1]) * 100
        label = int(MODEL.predict(df)[0])
        tier = 'FRAUD' if prob >= 70 else ('SUSPICIOUS' if prob >= 40 else 'SAFE')

        return {
            'is_fraud':  label,
            'fraud_prob': round(prob, 1),
            'risk_tier': tier
        }
    except Exception as e:
        print(f"[FraudShield] Model prediction error: {e}")
        return {'is_fraud': 0, 'fraud_prob': 0.0, 'risk_tier': 'SAFE'}


# ── Login required decorator ───────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please sign in to access the dashboard.', 'error')
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated


# ── Helper: build analyst user object for templates ────────────
def get_current_user():
    return {
        'name':     session.get('analyst_name', 'Fraud Analyst'),
        'initials': session.get('analyst_initials', 'FA'),
        'email':    session.get('analyst_email', '')
    }


# ── Helper: build dashboard stats ─────────────────────────────
def get_dashboard_stats(db=None):
    """
    Replace the dummy values below with real DB queries.
    Example with SQLAlchemy:
        from merchant.models import Transaction
        total = Transaction.query.count()
        fraud = Transaction.query.filter_by(isFraud=1).count()
    """
    return {
        'total_txns':       '12,847',
        'fraud_count':      '234',
        'fraud_today':      '23',
        'suspicious_today': '41',
        'avg_fraud_prob':   '78.4%',
        'blocked_value':    '$1.2M',
        'fraud_rate':       '1.82%',
        'model_accuracy':   '99.3%',
        'merchant_count':   '142',
        'review_count':     '7',
        'alert_count':      '7',
    }


# =============================================================
#  ROUTE 1 — Login / Landing page
#  URL:  GET  /
#        POST /login
# =============================================================
@bp.route('/', methods=['GET'])
def index():
    """Landing page with project description + login card."""
    if session.get('logged_in'):
        return redirect(url_for('main.dashboard'))
    return render_template('login.html')


@bp.route('/login', methods=['POST'])
def login():
    """
    Validate credentials.
    On success: set session → redirect to dashboard.
    On failure: flash error → back to login.
    """
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '').strip()
    remember = request.form.get('remember')  # 'on' or None

    if email == ANALYST_EMAIL.lower() and password == ANALYST_PASSWORD:
        session.permanent = bool(remember)
        session['logged_in']        = True
        session['analyst_email']    = email
        session['analyst_name']     = 'Fraud Analyst'
        session['analyst_initials'] = 'FA'
        return redirect(url_for('main.dashboard'))
    else:
        flash('Invalid email or password. Please try again.', 'error')
        return redirect(url_for('main.index'))


# =============================================================
#  ROUTE 2 — Dashboard
#  URL:  GET  /dashboard
#        GET  /dashboard?txn_id=3302753   (search)
# =============================================================
@bp.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    """
    Main dashboard.
    If ?txn_id is in the query string, also fetch that transaction
    and run the .pkl model on it — result shown in the search box.
    """
    stats       = get_dashboard_stats()
    transaction = None
    searched_id = request.args.get('txn_id', '').strip()

    if searched_id:
        transaction = fetch_transaction(searched_id)
        if transaction:
            # Run .pkl model
            result = predict_fraud(transaction)
            transaction['isFraud']   = result['is_fraud']
            transaction['fraud_prob'] = result['fraud_prob']
            transaction['risk_tier']  = result['risk_tier']
        else:
            flash(f'No transaction found with ID {searched_id}.', 'error')

    # Recent transactions — replace with DB query
    recent_transactions = get_recent_transactions()

    # Live fraud alerts — replace with DB query
    fraud_alerts = get_fraud_alerts()

    # Ticker events — replace with real-time stream
    ticker_events = build_ticker_events(recent_transactions)

    return render_template(
        'dashboard.html',
        current_user        = get_current_user(),
        stats               = stats,
        transaction         = transaction,
        searched_id         = searched_id,
        recent_transactions = recent_transactions,
        fraud_alerts        = fraud_alerts,
        ticker_events       = ticker_events,
    )


# =============================================================
#  ROUTE 3 — Merchant analysis
#  URL:  GET  /merchant/<merchant_id>
# =============================================================
@bp.route('/merchant/<merchant_id>', methods=['GET'])
@login_required
def merchant(merchant_id):
    """
    Merchant detail page.
    Fetches merchant profile + latest transaction,
    runs .pkl model, passes all data to merchant.html.
    """
    merchant_data = fetch_merchant(merchant_id)
    if not merchant_data:
        flash(f'Merchant {merchant_id} not found.', 'error')
        return redirect(url_for('main.dashboard'))

    # Latest transaction for this merchant
    transaction = fetch_latest_transaction_for_merchant(merchant_id)

    # Run .pkl model on latest transaction
    fraud_score = 87  # default
    if transaction:
        result = predict_fraud(transaction)
        transaction['isFraud']   = result['is_fraud']
        transaction['fraud_prob'] = result['fraud_prob']
        transaction['risk_tier']  = result['risk_tier']
        fraud_score = result['fraud_prob']

    # Fraud history (all isFraud=1 records for this merchant)
    fraud_history = fetch_fraud_history(merchant_id)

    # AI indicators (SHAP values from your .pkl model)
    indicators = build_indicators(transaction, fraud_score)

    # NLP tags (from BERT/DistilBERT model)
    nlp_tags = build_nlp_tags(fraud_score)

    # Customer heatmap — 150 integers (0-4) representing Customer_1…Customer_150
    customer_heatmap = build_customer_heatmap(transaction)

    # Model breakdown scores
    model_scores = [
        {'name': 'XGBoost (150 features)', 'score': 89, 'color': 'var(--red)'},
        {'name': 'LSTM (temporal)',         'score': 84, 'color': 'var(--amber)'},
        {'name': 'BERT NLP',                'score': 91, 'color': 'var(--indigo)'},
        {'name': 'Ensemble Fusion',         'score': int(fraud_score), 'color': 'var(--red)'},
    ]

    return render_template(
        'merchant.html',
        current_user     = get_current_user(),
        stats            = get_dashboard_stats(),
        merchant         = merchant_data,
        transaction      = transaction,
        fraud_score      = fraud_score,
        fraud_history    = fraud_history,
        indicators       = indicators,
        nlp_tags         = nlp_tags,
        customer_heatmap = customer_heatmap,
        model_scores     = model_scores,
    )


# =============================================================
#  ROUTE 4 — Merchant action (block / refund / alert)
#  URL:  POST /merchant/<merchant_id>/action
# =============================================================
@bp.route('/merchant/<merchant_id>/action', methods=['POST'])
@login_required
def merchant_action(merchant_id):
    """
    Handle analyst actions on a merchant.
    action values: 'block' | 'refund' | 'alert'
    """
    action = request.form.get('action', '')

    if action == 'block':
        # TODO: update merchant status in DB to 'blocked'
        # Example: Merchant.query.get(merchant_id).status = 'blocked'
        flash(f'Merchant {merchant_id} has been blocked.', 'success')

    elif action == 'refund':
        # TODO: trigger refund via payment API
        flash(f'Refund issued for latest transaction on merchant {merchant_id}.', 'success')

    elif action == 'alert':
        # TODO: send email / Slack alert to compliance team
        flash(f'Investigation alert sent for merchant {merchant_id}.', 'success')

    else:
        flash('Unknown action.', 'error')

    return redirect(url_for('main.merchant', merchant_id=merchant_id))


# =============================================================
#  ROUTE 5 — Logout
#  URL:  GET /logout
# =============================================================
@bp.route('/logout')
def logout():
    """Clear session and go back to login page."""
    session.clear()
    flash('You have been signed out.', 'success')
    return redirect(url_for('main.index'))


# =============================================================
#  DATA HELPER FUNCTIONS
#  Replace these with real DB queries when your models are ready
# =============================================================

def fetch_transaction(txn_id: str) -> dict | None:
    """
    Fetch a transaction by ID from your database.
    Replace with:
        from merchant.models import Transaction
        row = Transaction.query.get(txn_id)
        return row.__dict__ if row else None
    """
    # ── DUMMY DATA (remove when DB is ready) ──
    sample = {
        '3302753': {
            'TransactionID': 3302753,
            'TransactionAmt': 10000,
            'TransactionDT': 7859603,
            'ProductCD': 'R',
            'card_id': 15333,
            'card_bank_code': 562,
            'card_user_group': 150,
            'card_network': 'visa',
            'card_user': 226,
            'payment_type': 'credit',
            'billing_region_code': 448,
            'billing_country_code': 87,
            'billing_transaction_distance': None,
            'dist2': None,
            'P_emaildomain': 'yahoo.com',
            'merchant_id': 'M001',
            'merchant_name': 'QuickBuy Electronics',
        }
    }
    return sample.get(str(txn_id))


def fetch_merchant(merchant_id: str) -> dict | None:
    """
    Fetch merchant profile from your database.
    Replace with:
        from merchant.models import Merchant
        row = Merchant.query.get(merchant_id)
        return row.__dict__ if row else None
    """
    # ── DUMMY DATA (remove when DB is ready) ──
    merchants = {
        'M001': {
            'id': 'M001',
            'name': 'QuickBuy Electronics',
            'card_bank_code': '562',
            'card_user_group': '150',
            'card_network': 'visa',
            'billing_country_code': '87',
            'total_txns': 847,
            'fraud_count': 23,
            'total_fraud_amt': '$48,290',
            'avg_fraud_prob': '87.3%',
            'chargeback_rate': '18.4%',
            'top_emaildomain': 'yahoo.com',
        }
    }
    return merchants.get(merchant_id)


def fetch_latest_transaction_for_merchant(merchant_id: str) -> dict | None:
    """
    Fetch the most recent transaction for a merchant.
    Replace with DB query ordered by TransactionDT desc.
    """
    return fetch_transaction('3302753')  # dummy


def fetch_fraud_history(merchant_id: str) -> list:
    """
    Fetch all isFraud=1 transactions for a merchant.
    Replace with:
        Transaction.query.filter_by(merchant_id=merchant_id, isFraud=1).all()
    """
    return [
        {'TransactionID':3302753,'TransactionAmt':10000,'ProductCD':'R','card_network':'visa','payment_type':'credit','P_emaildomain':'yahoo.com','billing_country_code':87,'billing_transaction_distance':None,'dist2':None,'fraud_prob':94,'isFraud':1},
        {'TransactionID':3563358,'TransactionAmt':4000, 'ProductCD':'S','card_network':'mastercard','payment_type':'debit', 'P_emaildomain':'gmail.com','billing_country_code':87,'billing_transaction_distance':None,'dist2':None,'fraud_prob':88,'isFraud':1},
        {'TransactionID':3408536,'TransactionAmt':6160, 'ProductCD':'W','card_network':'visa','payment_type':'debit', 'P_emaildomain':'gmail.com','billing_country_code':87,'billing_transaction_distance':15,  'dist2':127, 'fraud_prob':82,'isFraud':1},
        {'TransactionID':3026005,'TransactionAmt':8636, 'ProductCD':'W','card_network':'visa','payment_type':'credit','P_emaildomain':'gmail.com','billing_country_code':87,'billing_transaction_distance':None,'dist2':None,'fraud_prob':91,'isFraud':1},
        {'TransactionID':3388604,'TransactionAmt':3837, 'ProductCD':'C','card_network':'visa','payment_type':'debit', 'P_emaildomain':'gmail.com','billing_country_code':142,'billing_transaction_distance':None,'dist2':None,'fraud_prob':79,'isFraud':1},
    ]


def get_recent_transactions() -> list:
    """
    Fetch 10 most recent transactions for the dashboard table.
    Each dict must include all columns shown in dashboard.html.
    Replace with DB query.
    """
    return [
        {'TransactionID':3302753,'merchant_id':'M001','merchant_name':'QuickBuy Electronics','TransactionAmt':10000,'ProductCD':'R','card_network':'visa','payment_type':'credit','P_emaildomain':'yahoo.com','isFraud':1,'fraud_prob':94,'risk_tier':'FRAUD'},
        {'TransactionID':3563358,'merchant_id':'M002','merchant_name':'TechStore Pro',       'TransactionAmt':4000, 'ProductCD':'S','card_network':'mastercard','payment_type':'debit', 'P_emaildomain':'gmail.com','isFraud':1,'fraud_prob':88,'risk_tier':'FRAUD'},
        {'TransactionID':3501200,'merchant_id':'M003','merchant_name':'Fashion Hub',          'TransactionAmt':37276,'ProductCD':'W','card_network':'visa','payment_type':'debit', 'P_emaildomain':'gmail.com','isFraud':0,'fraud_prob':3, 'risk_tier':'SAFE'},
        {'TransactionID':3408536,'merchant_id':'M004','merchant_name':'GadgetZone',           'TransactionAmt':6160, 'ProductCD':'W','card_network':'visa','payment_type':'debit', 'P_emaildomain':'gmail.com','isFraud':1,'fraud_prob':82,'risk_tier':'FRAUD'},
        {'TransactionID':3026005,'merchant_id':'M005','merchant_name':'HomeDecor Plus',        'TransactionAmt':8636, 'ProductCD':'W','card_network':'visa','payment_type':'credit','P_emaildomain':'gmail.com','isFraud':1,'fraud_prob':61,'risk_tier':'SUSPICIOUS'},
        {'TransactionID':3388604,'merchant_id':'M006','merchant_name':'ElectroMart',           'TransactionAmt':3837, 'ProductCD':'C','card_network':'visa','payment_type':'debit', 'P_emaildomain':'gmail.com','isFraud':1,'fraud_prob':91,'risk_tier':'FRAUD'},
        {'TransactionID':3112892,'merchant_id':'M007','merchant_name':'StyleShop',             'TransactionAmt':6000, 'ProductCD':'H','card_network':'visa','payment_type':'credit','P_emaildomain':'gmail.com','isFraud':1,'fraud_prob':85,'risk_tier':'FRAUD'},
        {'TransactionID':3255690,'merchant_id':'M009','merchant_name':'SafeShop',              'TransactionAmt':831,  'ProductCD':'C','card_network':'visa','payment_type':'credit','P_emaildomain':'gmail.com','isFraud':0,'fraud_prob':7, 'risk_tier':'SAFE'},
    ]


def get_fraud_alerts() -> list:
    """Fetch top active fraud alerts. Replace with DB query."""
    return [
        {'merchant_name':'QuickBuy Electronics','merchant_id':'M001','is_fraud':1,'prob':94,'tier':'FRAUD',     'time':'2 min ago', 'level':'h'},
        {'merchant_name':'GadgetZone',           'merchant_id':'M004','is_fraud':1,'prob':88,'tier':'FRAUD',     'time':'17 min ago','level':'h'},
        {'merchant_name':'HomeDecor Plus',         'merchant_id':'M005','is_fraud':1,'prob':61,'tier':'SUSPICIOUS','time':'40 min ago','level':'m'},
        {'merchant_name':'ElectroMart',            'merchant_id':'M006','is_fraud':1,'prob':91,'tier':'FRAUD',     'time':'1 hr ago',  'level':'h'},
        {'merchant_name':'StyleShop',              'merchant_id':'M007','is_fraud':1,'prob':58,'tier':'SUSPICIOUS','time':'2 hr ago',  'level':'m'},
        {'merchant_name':'DealHub',                'merchant_id':'M008','is_fraud':1,'prob':79,'tier':'FRAUD',     'time':'3 hr ago',  'level':'h'},
        {'merchant_name':'SwiftPay Store',         'merchant_id':'M009','is_fraud':1,'prob':63,'tier':'SUSPICIOUS','time':'4 hr ago',  'level':'m'},
    ]


def build_ticker_events(transactions: list) -> list:
    """Build ticker events from recent transactions for the live feed."""
    events = []
    for t in transactions[:7]:
        events.append({
            'id':    t['TransactionID'],
            'label': t['risk_tier'],
            'prob':  t['fraud_prob'],
            'tier':  t['risk_tier'],
            'amt':   f"${t['TransactionAmt']:,}",
        })
    return events


def build_indicators(transaction: dict | None, fraud_score: float) -> list:
    """
    Build AI fraud indicator cards from SHAP values.
    When your .pkl is ready, replace with:
        import shap
        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(df)
        # Map top SHAP features → indicator cards
    """
    if not transaction:
        return []
    indicators = []
    amt = transaction.get('TransactionAmt', 0)
    if amt and amt > 5000:
        indicators.append({'ico':'💰','title':'Abnormal TransactionAmt','desc':f'${amt:,.0f} significantly exceeds merchant average. LSTM temporal model flags sharp deviation from 90-day baseline.','conf':94,'w':'94%','c':'r'})
    if transaction.get('billing_country_code') == 87:
        indicators.append({'ico':'🌍','title':'High-Risk billing_country_code=87','desc':'Maps to high-risk jurisdiction. Cross-border pattern with no prior history at this origin.','conf':91,'w':'91%','c':'r'})
    if transaction.get('P_emaildomain') in ['yahoo.com', 'hotmail.com']:
        indicators.append({'ico':'📧','title':'Suspicious P_emaildomain','desc':f"{transaction.get('P_emaildomain')} + billing_country_code=87 + high TransactionAmt matches known fraud cluster.","conf":85,'w':'85%','c':'r'})
    if transaction.get('billing_transaction_distance') is None:
        indicators.append({'ico':'📍','title':'NULL billing_transaction_distance','desc':'73% of NULL billing_transaction_distance records in training set had isFraud=1.','conf':80,'w':'80%','c':'a'})
    if transaction.get('dist2') is None:
        indicators.append({'ico':'📏','title':'NULL dist2','desc':'Missing dist2 correlates strongly with fraud in XGBoost feature importance ranking.','conf':75,'w':'75%','c':'a'})
    indicators.append({'ico':'💳','title':'card_user_group Risk Profile','desc':f"card_user_group={transaction.get('card_user_group')} + {transaction.get('card_network')} + {transaction.get('payment_type')} has elevated fraud lift.",'conf':71,'w':'71%','c':'a'})
    return indicators[:6]


def build_nlp_tags(fraud_score: float) -> list:
    """
    NLP tags from BERT/DistilBERT classifier.
    Replace with real model predictions when BERT is ready.
    """
    if fraud_score >= 70:
        return [
            {'label':'fake_storefront',    'level':'h'},
            {'label':'misleading_listing', 'level':'h'},
            {'label':'chargeback_pattern', 'level':'h'},
            {'label':'synthetic_email',    'level':'m'},
            {'label':'price_manipulation', 'level':'m'},
            {'label':'velocity_spike',     'level':'m'},
            {'label':'new_card_id',        'level':'l'},
            {'label':'unverified_billing', 'level':'l'},
        ]
    elif fraud_score >= 40:
        return [
            {'label':'velocity_spike',     'level':'m'},
            {'label':'unusual_pattern',    'level':'m'},
            {'label':'new_card_id',        'level':'l'},
        ]
    return [{'label':'normal_pattern', 'level':'l'}]


def build_customer_heatmap(transaction: dict | None) -> list:
    """
    Build 150 intensity values (0-4) for Customer_1…Customer_150.
    Replace with real Customer_ column values from your dataset.
    Example:
        return [int(transaction.get(f'Customer_{i+1}', 0)) for i in range(150)]
    """
    if not transaction:
        return [0] * 150
    # Dummy: cycle through levels with some randomness
    import random
    random.seed(42)
    return [random.choices([0,1,2,3,4], weights=[30,25,20,15,10])[0] for _ in range(150)]