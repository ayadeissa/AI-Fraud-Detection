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
            min-height: 810px;
            border-radius: 25px;
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
            text-align: center;
            margin-bottom: -10px;
        }}

        
        </style>

        <div class="hero-section"><div>
                <div class="hero-title">BANQUE MISR MODEL</div>
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
errors="coerce")
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
            errors="coerce")
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

         with col1:
           loan_amount = st.number_input("Loan Amount",
                min_value=0.0,
                value=50000.0)
            
           credit_score = st.number_input(
                "Credit Score",
                min_value=550,
                max_value=850,
                value=700)
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

if st.button("🔄 Apply Changes",
   use_container_width=True):
   st.session_state["dataset_ready"] = True
   st.session_state["uploaded_df"] = df.copy()
   st.rerun()
            # Make a copy of model features
X = df[MODEL_COLS].copy()

            # Convert numerical columns to numeric
for col in NUMERICAL_COLS:
    X[col] = pd.to_numeric(X[col],
    errors="coerce")
