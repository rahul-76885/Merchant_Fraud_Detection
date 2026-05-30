# Merchant AI Fraud Detection

Local Flask app for merchant fraud scoring with a two-model ensemble, SQLite prediction history, and a working dashboard UI.

## What is included

- Flask login and dashboard routes
- `/predict` JSON or form inference endpoint
- `/history` SQLite-backed prediction history
- Schema-driven preprocessing saved to `merchant/model/preprocessor.pkl`
- XGBoost and LightGBM model loading from `merchant/model`
- Simple fallback scoring when artefacts are missing

## Expected model artefacts

Place these files inside `merchant/model/` when available:

- `xgb_model.pkl`
- `lgb_model.pkl`
- `preprocessor.pkl`
- `ensemble.pkl`

If any are missing, the app still runs using fallback scoring.

## Run locally

1. Create and activate your virtual environment.
2. Install dependencies with:

```bash
pip install -r requirements.txt
```

3. Train the models:

```bash
python train_ensemble.py
```

4. Start the app from the `merchant` folder:

```bash
python routes.py
```

5. Open `http://127.0.0.1:5000` and sign in with:

- Email: `analyst@merchant.ai`
- Password: `merchant123`

## Updated structure

- `merchant/__init__.py` app factory
- `merchant/routes.py` Flask routes and SQLite storage
- `merchant/ensemble.py` simplified inference bundle
- `merchant/preprocessing.py` schema-driven preprocessing
- `merchant/dl_models.py` pickle model helpers
- `merchant/template/login.html` login UI
- `merchant/template/dashboard.html` dashboard UI
- `merchant/static/css/login.css` login styling
- `merchant/static/css/dashboard.css` dashboard styling
- `merchant/static/js/login.js` login enhancements
- `merchant/static/js/dashboard.js` AJAX prediction flow
