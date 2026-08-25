from pathlib import Path
import base64
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "model_artifact.joblib"

NUMERICAL_COLS = [
    "loan_amount", "credit_score", "annual_income", "dti_ratio",
    "employment_length_years", "num_existing_loans", "account_age_months",
    "days_to_first_tx", "pct_spent_48h", "pct_spent_7d",
    "cash_withdrawal_ratio", "high_risk_spend_ratio",
    "international_tx_ratio", "nighttime_tx_ratio", "num_unique_merchants",
    "num_total_transactions", "avg_tx_amount", "max_single_tx_pct"
]
CATEGORICAL_COLS = [
    "declared_purpose", "primary_mcc_category", "secondary_mcc_category"
]
MODEL_COLS = NUMERICAL_COLS + CATEGORICAL_COLS + ["mcc_mismatch_flag"]
TARGET = "is_flagged_misuse"

hero_path = Path(__file__).parent / "assets"  / "ChatGPT Image Aug 25, 2026, 11_14_42 PM.png"

if hero_path.exists():
    hero_base64 = get_image_base64(hero_path)

    st.markdown(
        f"""
        <style>
        .hero-section {{
            background-image:
                url("data:image/png;base64,{hero_base64}");

            background-size: 100%;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 800px;
            border-radius: 24px;
            padding: 45px;
            display: flex;
            filter: none !important;
            opacity: 1 !important;
            align-items: flex-end;
        }}

        .hero-title {{
            font-size: 46px;
            font-weight: 800;
            color: #8D1930 !important;
            margin: 0;
        }}

        .hero-subtitle {{
            font-size: 18px;
            color: #d6dbe3 !important;
            margin-top: 8px;
        }}

        .hero-subtitle-2 {{
            font-size: 18px;
            color: #131313 !important;
            margin-top: 8px;
        }}
        </style>

        <div class="hero-section">
            <div>
                <div class="hero-title">BANQUE MISR MODEL</div>
                <div class="hero-subtitle">
                    RISK DETECTOR AI MODEL
                <div class="hero-subtitle">
                    AI MODEL TO PREDICT IF THE CUSTOMER MISUSE OR NOT
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Hero image not found: assets/hero.png")

PURPOSES = [
    "Medical", "Equipment Purchase", "Education",
    "Business Expansion", "Debt Consolidation"
]
MCC = [
    "Retail", "Casino", "Crypto Exchange",
    "Equipment Vendor", "Luxury Travel", "ATM Cash"
]

def validate_input(df, require_target=False):
    errors = []
    required = MODEL_COLS + (["borrower_id"] if "borrower_id" in df.columns else [])
    if require_target:
        required.append(TARGET)

    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append("Missing columns: " + ", ".join(missing))
        return errors

    for col in NUMERICAL_COLS:
        n = pd.to_numeric(df[col], errors="coerce")
        if n.isna().any():
            errors.append(f"{col}: invalid or missing numeric value.")

    if "credit_score" in df:
        n = pd.to_numeric(df["credit_score"], errors="coerce")
        if ((n < 550) | (n > 850)).any():
            errors.append("credit_score must be between 550 and 850.")

    ratio_cols = [
        "dti_ratio", "pct_spent_48h", "pct_spent_7d",
        "cash_withdrawal_ratio", "high_risk_spend_ratio",
        "international_tx_ratio", "nighttime_tx_ratio", "max_single_tx_pct"
    ]
    for col in ratio_cols:
        n = pd.to_numeric(df[col], errors="coerce")
        if ((n < 0) | (n > 1)).any():
            errors.append(f"{col} must be between 0 and 1.")

    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            errors.append(f"{col}: missing value.")

    if not df["declared_purpose"].isin(PURPOSES).all():
        errors.append("declared_purpose contains an invalid category.")
    if not df["primary_mcc_category"].isin(MCC).all():
        errors.append("primary_mcc_category contains an invalid category.")

    if require_target:
        y = pd.to_numeric(df[TARGET], errors="coerce")
        if y.isna().any() or not y.isin([0, 1]).all():
            errors.append("is_flagged_misuse must contain only 0/1.")

    return errors

@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


artifact = load_artifact()

if artifact is None:
    st.error("Model artifact not found.")
    st.markdown("""
    The Streamlit UI is ready, but it needs the trained model artifact.

    Run:

    `python train_model.py --data loan_diversion_dataset_22f.csv`

    This creates `model_artifact.joblib`. Put that file beside `app.py`, then redeploy.
    """)
    st.stop()

model = artifact["model"]
metrics = artifact.get("metrics", {})

m1, m2, m3 = st.columns(3)
m1.metric("Model ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
m2.metric("Model Accuracy", f"{metrics.get('accuracy', 0):.3f}")
m3.metric("CV Score", f"{metrics.get('best_cv_score', 0):.3f}")

st.subheader("🔎 Borrower Investigation")

with st.form("borrower_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        borrower_id = st.text_input("Borrower ID", "BW-DEMO-001")
        loan_amount = st.number_input("Loan amount", 0.0, value=50000.0)
        credit_score = st.number_input("Credit score", 550, 850, 700)
        annual_income = st.number_input("Annual income", 0.0, value=100000.0)
        dti_ratio = st.number_input("DTI ratio", 0.0, 1.0, 0.35)
        employment_length_years = st.number_input("Employment length", 0, 60, 10)
        num_existing_loans = st.number_input("Existing loans", 0, 50, 3)
        account_age_months = st.number_input("Account age months", 1, 600, 60)

    with c2:
        declared_purpose = st.selectbox("Declared purpose", PURPOSES)
        primary_mcc_category = st.selectbox("Primary MCC", MCC)
        secondary_mcc_category = st.selectbox("Secondary MCC", MCC)
        mcc_mismatch_flag = st.selectbox("MCC mismatch", [0, 1])
        days_to_first_tx = st.number_input("Days to first transaction", 0, 365, 10)
        pct_spent_48h = st.number_input("Spent in 48h", 0.0, 1.0, 0.50)
        pct_spent_7d = st.number_input("Spent in 7d", 0.0, 1.0, 0.70)
        cash_withdrawal_ratio = st.number_input("Cash withdrawal ratio", 0.0, 1.0, 0.30)

    with c3:
        high_risk_spend_ratio = st.number_input("High-risk spend ratio", 0.0, 1.0, 0.20)
        international_tx_ratio = st.number_input("International tx ratio", 0.0, 1.0, 0.20)
        nighttime_tx_ratio = st.number_input("Nighttime tx ratio", 0.0, 1.0, 0.20)
        num_unique_merchants = st.number_input("Unique merchants", 1, 10000, 15)
        num_total_transactions = st.number_input("Total transactions", 1, 100000, 25)
        avg_tx_amount = st.number_input("Average transaction", 0.0, value=2000.0)
        max_single_tx_pct = st.number_input("Max single transaction %", 0.0, 1.0, 0.50)

    submitted = st.form_submit_button("🚨 PREDICT ", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "borrower_id": borrower_id,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "annual_income": annual_income,
        "dti_ratio": dti_ratio,
        "employment_length_years": employment_length_years,
        "num_existing_loans": num_existing_loans,
        "account_age_months": account_age_months,
        "declared_purpose": declared_purpose,
        "primary_mcc_category": primary_mcc_category,
        "secondary_mcc_category": secondary_mcc_category,
        "mcc_mismatch_flag": mcc_mismatch_flag,
        "days_to_first_tx": days_to_first_tx,
        "pct_spent_48h": pct_spent_48h,
        "pct_spent_7d": pct_spent_7d,
        "cash_withdrawal_ratio": cash_withdrawal_ratio,
        "high_risk_spend_ratio": high_risk_spend_ratio,
        "international_tx_ratio": international_tx_ratio,
        "nighttime_tx_ratio": nighttime_tx_ratio,
        "num_unique_merchants": num_unique_merchants,
        "num_total_transactions": num_total_transactions,
        "avg_tx_amount": avg_tx_amount,
        "max_single_tx_pct": max_single_tx_pct,
    }])

    errors = validate_input(row)
    if errors:
        st.error("Validation failed — prediction blocked.")
        for e in errors:
            st.write("•", e)
    else:
        x = row[MODEL_COLS].copy()
        for col in NUMERICAL_COLS:
            x[col] = pd.to_numeric(x[col])

        probability = float(model.predict_proba(x)[0, 1])
        prediction = int(probability >= 0.5)

        if probability >= 0.75:
            level, message = "🔴 HIGH RISK", "Immediate review recommended."
        elif probability >= 0.50:
            level, message = "🟠 MEDIUM RISK", "Enhanced review recommended."
        else:
            level, message = "🟢 LOW RISK", "No elevated model signal."

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Misuse Probability", f"{probability:.1%}")
        r2.metric("Prediction", "MISUSE" if prediction else "LEGITIMATE")
        r3.metric("Risk Level", level)

        st.progress(min(probability, 1.0))
        st.info(message)


