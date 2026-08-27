from pathlib import Path
import base64
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model_artifact.joblib"

def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


artifact = load_artifact()


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

hero_path = Path(__file__).parent / "assets"  / "bank misr.png"

if hero_path.exists():
    hero_base64 = get_image_base64(hero_path)

    # ==============================
# HERO IMAGE
# ==============================
st.markdown(
    f"""
    <style>
    .hero-image-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 18px;
    }}

    .hero-image {{
        width: 255px;
        height: 255px;
        object-fit: cover;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }}

    .bank-card {{
        background: white;
        border: 1px solid #e5e5e5;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-bottom: 18px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    }}

    .bank-title {{
        color: #FFC107;
        font-size: 34px;
        font-weight: 800;
        margin: 0;
    }}
    </style>

    <div class="hero-image-container">
        <img src="data:image/png;base64,{hero_base64}"
             class="hero-image">
    </div>

    <div class="bank-card">
        <div class="bank-title">
            BANK MISR MODEL
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown( """
        <div style="
        background-color: #f9f9f9; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #e0e0e0;
        margin-top: 35px;
        margin-bottom: 20px;">
        <h4 style="color: #333; margin-top: 0;">RISK DETECTOR AI MODEL</h4>
        <p style="color: #666; margin-bottom: 0;">AI MODEL TO PREDICT IF THE CUSTOMER MISUSE OR NOT</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# TOGGLE VISUALIZATIONS & PLOTS
# ==========================================
show_plots = st.checkbox("📊 عرض كافة الرسميات البيانية والتحليلات (Show Analytics & Plots)")

if show_plots:
    if artifact is not None:
        metrics = artifact.get("metrics", {})
        
        st.markdown("### 📊 Model Performance Analytics (Charts)")
        
        # 1. عرض المؤشرات الرئيسية (AUC, Accuracy, CV Score)
        auc_val = float(metrics.get('roc_auc', 0))
        acc_val = float(metrics.get('accuracy', 0))
        cv_val = float(metrics.get('best_cv_score', 0))
        
        combined_metrics_df = pd.DataFrame({
            "Score": [auc_val, acc_val, cv_val]
        }, index=["Model ROC-AUC", "Model Accuracy", "CV Score"])
        
        # عرض الرسم البياني الشريطي المدمج
        st.bar_chart(combined_metrics_df, height=300)

        st.divider()

        # 2. رسومات بيانية تفاعلية إضافية للبيانات
        st.markdown("### 📈 Data Insights & Distribution Charts")
        
        if "uploaded_df" in st.session_state:
            st.subheader("📊 رسم بياني لمبالغ القروض والدخل السنوي")
            st.area_chart(st.session_state["uploaded_df"][["loan_amount", "annual_income"]].head(25))
            
            st.subheader("📊 توزيع نسبة DTI والخصائص المالية")
            st.bar_chart(st.session_state["uploaded_df"][["dti_ratio", "pct_spent_48h", "pct_spent_7d"]].head(25))
        else:
            st.info("💡 قم برفع ملف CSV لعرض التحليلات والرسومات البيانية الخاصة بالعملاء.")

    st.divider()

MODEL_COLS = NUMERICAL_COLS + CATEGORICAL_COLS + ["mcc_mismatch_flag"]
TARGET = "is_flagged_misuse"


PURPOSES = [
    "Medical",
    "Equipment Purchase",
    "Education",
    "Business Expansion",
    "Debt Consolidation"
]

MCC = [
    "Retail",
    "Casino",
    "Crypto Exchange",
    "Equipment Vendor",
    "Luxury Travel",
    "ATM Cash"
]

def validate_batch_data(df):
    errors = []

    # Check required columns
    required_columns = MODEL_COLS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        errors.append(
            "Missing columns: " + ", ".join(missing_columns)
        )
        return errors

    # Validate numerical columns
    for col in NUMERICAL_COLS:
        numeric_values = pd.to_numeric(
            df[col],
            errors="coerce"
        )



# Validate credit score
    if "credit_score" in df.columns:
        credit_score = pd.to_numeric(
            df["credit_score"],
            errors="coerce"
        )
        if ((credit_score < 550) | (credit_score > 850)).any():
            errors.append("credit_score must be between 550 and 850.")

    # Validate ratio columns
    ratio_cols = [
        "dti_ratio",
        "pct_spent_48h",
        "pct_spent_7d",
        "cash_withdrawal_ratio",
        "high_risk_spend_ratio",
        "international_tx_ratio",
        "nighttime_tx_ratio",
        "max_single_tx_pct"
    ]

    for col in ratio_cols:
        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )
        if ((values < 0) | (values > 1)).any():
            errors.append(f"{col} must be between 0 and 1.")

    # Validate categorical columns
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            errors.append(f"{col}: missing value.")

    if not df["declared_purpose"].isin(PURPOSES).all():
        errors.append("declared_purpose contains an invalid category.")

    if not df["primary_mcc_category"].isin(MCC).all():
        errors.append("primary_mcc_category contains an invalid category.")

    return errors


if st.session_state.get("dataset_ready", False):
    st.markdown("## 👤 Customer Risk Assessment")
    st.caption("Enter the customer's information to evaluate misuse risk.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

         loan_amount = st.number_input("Loan Amount")
         credit_score = st.number_input(
             "Credit Score",
             min_value=550,
             max_value=850
    )
        
        with col1:
            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0.0,
                value=50000.0
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=550,
                max_value=850,
                value=700
            )

            annual_income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=100000.0
            )

            dti_ratio = st.number_input(
                "DTI Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.30
            )

            employment_length_years = st.number_input(
                "Employment Length (Years)",
                min_value=0,
                value=5
            )

            num_existing_loans = st.number_input(
                "Existing Loans",
                min_value=0,
                value=2
            )

        with col2:
            account_age_months = st.number_input(
                "Account Age (Months)",
                min_value=0,
                value=36
            )

            declared_purpose = st.selectbox(
                "Declared Purpose",
                PURPOSES
            )

            primary_mcc_category = st.selectbox(
                "Primary MCC Category",
                MCC
            )

            secondary_mcc_category = st.selectbox(
                "Secondary MCC Category",
                MCC
            )

            mcc_mismatch_flag = st.selectbox(
                "MCC Mismatch",
                [0, 1]
            )

            days_to_first_tx = st.number_input(
                "Days to First Transaction",
                min_value=0,
                value=10
            )

        with col3:
            pct_spent_48h = st.number_input(
                "Spent in 48h",
                min_value=0.0,
                max_value=1.0,
                value=0.50
            )

            pct_spent_7d = st.number_input(
                "Spent in 7 Days",
                min_value=0.0,
                max_value=1.0,
                value=0.70
            )

            cash_withdrawal_ratio = st.number_input(
                "Cash Withdrawal Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.20
            )

            high_risk_spend_ratio = st.number_input(
                "High Risk Spend Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.20
            )

            international_tx_ratio = st.number_input(
                "International Transaction Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.10
            )

            nighttime_tx_ratio = st.number_input(
                "Nighttime Transaction Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.10
            )

            num_unique_merchants = st.number_input(
                "Unique Merchants",
                min_value=0,
                value=10
            )

            num_total_transactions = st.number_input(
                "Total Transactions",
                min_value=0,
                value=20
            )

            avg_tx_amount = st.number_input(
                "Average Transaction Amount",
                min_value=0.0,
                value=1000.0
            )

            max_single_tx_pct = st.number_input(
                "Maximum Single Transaction %",
                min_value=0.0,
                max_value=1.0,
                value=0.50
            )
        submitted = st.form_submit_button(
            "🚨 Predict Customer Risk",
            use_container_width=True
        )

if submitted:

    customer_data = pd.DataFrame([{
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "annual_income": annual_income,
        "dti_ratio": dti_ratio,
        "employment_length_years": employment_length_years,
        "num_existing_loans": num_existing_loans,
        "account_age_months": account_age_months,
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
        "declared_purpose": declared_purpose,
        "primary_mcc_category": primary_mcc_category,
        "secondary_mcc_category": secondary_mcc_category,
        "mcc_mismatch_flag": mcc_mismatch_flag
    }])

    # Validate entered customer data
    errors = validate_batch_data(customer_data)

    if errors:
        st.error("❌ Invalid Customer Data")

        for error in errors:
            st.write(f"• {error}")

    else:
        try:
            model = artifact["model"]

            X = customer_data[MODEL_COLS].copy()

            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0, 1]

            if probability >= 0.75:
                risk = "HIGH"
            elif probability >= 0.50:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            st.success("✅ Prediction completed successfully!")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Fraud Probability",
                f"{probability:.1%}"
            )

            col2.metric(
                "Prediction",
                "MISUSE" if prediction == 1 else "LEGITIMATE"
            )

            col3.metric(
                "Risk Level",
                risk
            )

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
# =========================================================
# BATCH DATA UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📁 Upload Dataset",
    type=["csv"],
    help="Upload a CSV file containing the model features."
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validation
    errors = validate_batch_data(df)

    if errors:
        st.error("❌ Data Validation Failed")

        for error in errors:
            st.write(f"• {error}")

    else:
        st.success("✅ Data Validation Passed")

        if st.button(
            "🔄 Apply Changes",
            use_container_width=True
        ):
            st.session_state["dataset_ready"] = True
            st.session_state["uploaded_df"] = df.copy()
            st.rerun()

        # Make a copy of model features
        X = df[MODEL_COLS].copy()

        # Convert numerical columns to numeric
        for col in NUMERICAL_COLS:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

        # ==========================================
        # USE THE TRAINED MODEL
        # ==========================================
        if artifact is None:
            st.error("❌ Model artifact not found.")
            st.stop()

        model = artifact["model"]

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        # ==========================================
        # ADD RESULTS
        # ==========================================

        df["fraud_probability"] = probabilities

        df["prediction"] = predictions

        df["risk_level"] = pd.cut(
            probabilities,
            bins=[-0.01, 0.50, 0.75, 1.0],
            labels=[
                "LOW",
                "MEDIUM",
                "HIGH"
            ]
        )

       

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

    errors = validate_batch_data(row)
    if errors:
        st.error("❌ Invalid Customer Data")
        for error in errors:
            st.write(f"• {error}")
    else:
        model = artifact["model"]
        x = row[MODEL_COLS].copy()
        prediction = model.predict(x)[0]

        probability = model.predict_proba(x)[0, 1]

        if probability >= 0.75:
            level, message = "🔴 HIGH RISK", "Immediate review recommended."
        elif probability >= 0.50:
            level, message = "🟠 MEDIUM RISK", "Enhanced review recommended."
        else:
            level, message = "🟢 LOW RISK", "No elevated model signal."
        st.markdown("## 🎯 Risk Assessment")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Fraud Probability",
            f"{probability:.1%}"
        )
        c2.metric(
            "Prediction",
            "MISUSE" if prediction == 1 else "LEGITIMATE"
        )

        c3.metric(
            "Risk Level",
            level
        )

        st.progress(min(
            float(probability), 1.0
        ))
