
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Schema from the assignment
# -----------------------------
SCHEMA = {
    "borrower_id": {"type": "String", "kind": "string", "description": "Unique account/borrower identifier"},
    "loan_amount": {"type": "Float", "kind": "float", "description": "Total principal loan disbursed ($)"},
    "credit_score": {"type": "Integer", "kind": "int", "description": "Borrower credit score at application", "min": 550, "max": 850},
    "annual_income": {"type": "Float", "kind": "float", "description": "Borrower declared annual income ($)", "min": 0},
    "dti_ratio": {"type": "Float", "kind": "float", "description": "Debt-to-income ratio", "min": 0, "max": 1},
    "employment_length_years": {"type": "Integer", "kind": "int", "description": "Years in current employment", "min": 0},
    "num_existing_loans": {"type": "Integer", "kind": "int", "description": "Count of active open loans", "min": 0},
    "account_age_months": {"type": "Integer", "kind": "int", "description": "Bank account tenure in months", "min": 0},
    "declared_purpose": {"type": "Categorical", "kind": "category", "description": "Stated loan intent",
                         "allowed": ["Medical", "Equipment Purchase", "Education", "Business Expansion", "Debt Consolidation"]},
    "primary_mcc_category": {"type": "Categorical", "kind": "category", "description": "Primary merchant category",
                             "allowed": ["Retail", "Casino", "Crypto Exchange", "Equipment Vendor", "Luxury Travel", "ATM Cash"]},
    "secondary_mcc_category": {"type": "Categorical", "kind": "category", "description": "Secondary merchant category"},
    "mcc_mismatch_flag": {"type": "Binary", "kind": "binary", "description": "Purpose/vendor contradiction flag", "allowed": [0, 1]},
    "days_to_first_tx": {"type": "Integer", "kind": "int", "description": "Days until first transaction", "min": 0},
    "pct_spent_48h": {"type": "Float", "kind": "float", "description": "Loan spent within first 48h", "min": 0, "max": 1},
    "pct_spent_7d": {"type": "Float", "kind": "float", "description": "Loan spent within first 7d", "min": 0, "max": 1},
    "cash_withdrawal_ratio": {"type": "Float", "kind": "float", "description": "Proportion converted to cash", "min": 0, "max": 1},
    "high_risk_spend_ratio": {"type": "Float", "kind": "float", "description": "High-risk spend proportion", "min": 0, "max": 1},
    "international_tx_ratio": {"type": "Float", "kind": "float", "description": "Overseas transaction proportion", "min": 0, "max": 1},
    "nighttime_tx_ratio": {"type": "Float", "kind": "float", "description": "Transactions from 12 AM–6 AM", "min": 0, "max": 1},
    "num_unique_merchants": {"type": "Integer", "kind": "int", "description": "Distinct merchants", "min": 1},
    "num_total_transactions": {"type": "Integer", "kind": "int", "description": "Total transaction records", "min": 1},
    "avg_tx_amount": {"type": "Float", "kind": "float", "description": "Mean transaction amount ($)", "min": 0},
    "max_single_tx_pct": {"type": "Float", "kind": "float", "description": "Largest transaction as proportion of loan", "min": 0, "max": 1},
    "is_flagged_misuse": {"type": "Binary", "kind": "target", "description": "Ground truth: 0 legitimate / 1 misuse", "allowed": [0, 1]},
}

NUMERICAL_COLS = [
    "loan_amount", "credit_score", "annual_income", "dti_ratio",
    "employment_length_years", "num_existing_loans", "account_age_months",
    "days_to_first_tx", "pct_spent_48h", "pct_spent_7d",
    "cash_withdrawal_ratio", "high_risk_spend_ratio",
    "international_tx_ratio", "nighttime_tx_ratio", "num_unique_merchants",
    "num_total_transactions", "avg_tx_amount", "max_single_tx_pct"
]
CATEGORICAL_COLS = ["declared_purpose", "primary_mcc_category", "secondary_mcc_category"]
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS + ["mcc_mismatch_flag"]
TARGET = "is_flagged_misuse"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def make_demo_data(n=2500, seed=7):
    rng = np.random.default_rng(seed)
    purposes = SCHEMA["declared_purpose"]["allowed"]
    mcc = SCHEMA["primary_mcc_category"]["allowed"]
    secondary = mcc + ["Other Services"]

    df = pd.DataFrame({
        "borrower_id": [f"BW-{i:05d}" for i in range(1, n + 1)],
        "loan_amount": rng.uniform(5000, 100000, n).round(2),
        "credit_score": rng.integers(550, 850, n),
        "annual_income": rng.uniform(30000, 180000, n).round(2),
        "dti_ratio": rng.uniform(0.10, 0.65, n).round(3),
        "employment_length_years": rng.integers(0, 25, n),
        "num_existing_loans": rng.integers(0, 8, n),
        "account_age_months": rng.integers(1, 120, n),
        "declared_purpose": rng.choice(purposes, n),
        "primary_mcc_category": rng.choice(mcc, n),
        "secondary_mcc_category": rng.choice(secondary, n),
        "mcc_mismatch_flag": rng.binomial(1, 0.13, n),
        "days_to_first_tx": rng.integers(0, 30, n),
        "pct_spent_48h": rng.uniform(0, 1, n).round(3),
        "pct_spent_7d": rng.uniform(0, 1, n).round(3),
        "cash_withdrawal_ratio": rng.uniform(0, 0.8, n).round(3),
        "high_risk_spend_ratio": rng.beta(1.2, 3.0, n).round(3),
        "international_tx_ratio": rng.uniform(0, 0.5, n).round(3),
        "nighttime_tx_ratio": rng.uniform(0, 0.6, n).round(3),
        "num_unique_merchants": rng.integers(1, 30, n),
        "num_total_transactions": rng.integers(1, 49, n),
        "avg_tx_amount": rng.lognormal(7.5, 0.65, n).round(2),
        "max_single_tx_pct": rng.uniform(0.1, 0.95, n).round(3),
    })
    risk = (
        1.3 * df["mcc_mismatch_flag"]
        + 1.7 * df["high_risk_spend_ratio"]
        + 0.9 * df["cash_withdrawal_ratio"]
        + 0.7 * df["international_tx_ratio"]
        + 0.7 * df["nighttime_tx_ratio"]
        + 0.8 * df["max_single_tx_pct"]
        + 0.7 * df["pct_spent_48h"]
        + 0.5 * df["pct_spent_7d"]
        + rng.normal(0, 0.7, n)
    )
    p = 1 / (1 + np.exp(-(risk - 2.9)))
    df[TARGET] = rng.binomial(1, p)
    return df

def validate_dataframe(df):
    rows = []
    for col, spec in SCHEMA.items():
        exists = col in df.columns
        actual = str(df[col].dtype) if exists else "MISSING"
        type_ok = False
        range_ok = True
        allowed_ok = True
        invalid_count = 0
        if exists:
            s = df[col]
            if spec["kind"] == "string":
                type_ok = pd.api.types.is_string_dtype(s) or s.dtype == object
            elif spec["kind"] == "float":
                type_ok = pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)
            elif spec["kind"] == "int":
                type_ok = pd.api.types.is_integer_dtype(s)
            elif spec["kind"] in ("binary", "target"):
                type_ok = pd.api.types.is_numeric_dtype(s)
                non_null = s.dropna()
                invalid_count = int((~non_null.isin(spec["allowed"])).sum())
                allowed_ok = invalid_count == 0
            elif spec["kind"] == "category":
                type_ok = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s)
                if "allowed" in spec:
                    non_null = s.dropna()
                    invalid_count = int((~non_null.isin(spec["allowed"])).sum())
                    allowed_ok = invalid_count == 0
            if "min" in spec and pd.api.types.is_numeric_dtype(s):
                range_ok &= not bool((s.dropna() < spec["min"]).any())
            if "max" in spec and pd.api.types.is_numeric_dtype(s):
                range_ok &= not bool((s.dropna() > spec["max"]).any())
        passed = exists and type_ok and range_ok and allowed_ok
        rows.append({
            "Column": col,
            "Expected Type": spec["type"],
            "Actual dtype": actual,
            "Type": "✅ PASS" if type_ok else "❌ FAIL",
            "Range / Values": "✅ PASS" if (range_ok and allowed_ok) else f"❌ FAIL ({invalid_count} invalid)",
            "Missing": int(df[col].isna().sum()) if exists else "-",
            "Overall": "✅ VALID" if passed else "❌ INVALID",
        })
    return pd.DataFrame(rows)

def iqr_stats(df, col):
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    count = int(((df[col] < low) | (df[col] > high)).sum())
    return q1, q3, iqr, low, high, count

def make_preprocessor():
    return ColumnTransformer([
        ("standard_scaling", StandardScaler(), NUMERICAL_COLS),
        ("one_hot_encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ], remainder="passthrough")

def train_models(df, tune=False):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=1
    )

    baseline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1500))
    ])
    baseline.fit(X_train, y_train)

    rf = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", RandomForestClassifier(class_weight="balanced", random_state=1))
    ])

    best_params = None
    best_score = None
    if tune:
        params = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10]
        }
        search = RandomizedSearchCV(
            rf, params, n_iter=10, scoring="accuracy",
            random_state=1, n_jobs=-1, cv=3
        )
        search.fit(X_train, y_train)
        rf = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
    else:
        rf.fit(X_train, y_train)

    results = {}
    for name, model in [("Logistic Regression", baseline), ("Random Forest", rf)]:
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, proba),
            "pred": pred, "proba": proba
        }

    return results, X_test, y_test, best_params, best_score

def risk_band(p):
    if p < .30:
        return "LOW", "🟢"
    if p < .60:
        return "MEDIUM", "🟡"
    if p < .80:
        return "HIGH", "🟠"
    return "CRITICAL", "🔴"

def predict_single(model, values):
    row = pd.DataFrame([values])
    p = float(model.predict_proba(row[FEATURE_COLS])[:, 1][0])
    band, icon = risk_band(p)
    return p, band, icon

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main-title {font-size: 2.6rem; font-weight: 800; margin-bottom: 0;}
.subtitle {color:#9ca3af; font-size:1.05rem; margin-bottom:1.5rem;}
.card {padding:18px; border-radius:16px; background:linear-gradient(135deg,#111827,#172033);
       border:1px solid #263246; margin-bottom:12px;}
.metric-big {font-size:2rem; font-weight:800;}
.small-muted {color:#9ca3af;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar / data loading
# -----------------------------
st.sidebar.markdown("## 🛡️ FraudShield AI")
st.sidebar.caption("Loan Purpose Diversion Intelligence")

uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
use_demo = st.sidebar.checkbox("Use built-in demo dataset", value=True)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        source_label = uploaded.name
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
elif use_demo:
    df = make_demo_data()
    source_label = "Built-in demo data"
else:
    st.info("Upload a CSV or enable the demo dataset from the sidebar.")
    st.stop()

st.session_state["df"] = df

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🛡️ FraudShield AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered loan purpose diversion detection • Data quality • Risk analytics • Explainable ML</div>', unsafe_allow_html=True)

# top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Records", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns)}")
c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
if TARGET in df.columns:
    c4.metric("Misuse Rate", f"{df[TARGET].mean()*100:.1f}%")
else:
    c4.metric("Target", "Missing")
c5.metric("Source", "DEMO" if uploaded is None else "CSV")

tabs = st.tabs(["🏠 Overview", "🔎 Data Quality", "📊 EDA", "🤖 AI Model", "🎯 Live Risk Check", "📋 Data Dictionary"])

# -----------------------------
# Overview
# -----------------------------
with tabs[0]:
    st.subheader("Executive Overview")
    if TARGET in df.columns:
        legit = int((df[TARGET] == 0).sum())
        misuse = int((df[TARGET] == 1).sum())
        a, b, c = st.columns(3)
        a.metric("Legitimate", f"{legit:,}")
        b.metric("Flagged Misuse", f"{misuse:,}")
        c.metric("Flag Rate", f"{misuse/len(df)*100:.1f}%")

        fig = px.pie(
            values=[legit, misuse],
            names=["Legitimate", "Misuse"],
            hole=.62,
            title="Portfolio Risk Composition"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🚨 Fast Risk Signals")
        signal_cols = ["mcc_mismatch_flag", "high_risk_spend_ratio", "cash_withdrawal_ratio",
                       "international_tx_ratio", "nighttime_tx_ratio"]
        available = [c for c in signal_cols if c in df.columns]
        signal_df = df.groupby(TARGET)[available].mean().T.reset_index().rename(columns={"index":"Signal"})
        if 0 in signal_df.columns and 1 in signal_df.columns:
            signal_df["Difference"] = signal_df[1] - signal_df[0]
            signal_df = signal_df.sort_values("Difference", ascending=False)
        st.dataframe(signal_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Target column is missing. Upload the assignment dataset to enable model analytics.")

# -----------------------------
# Data Quality
# -----------------------------
with tabs[1]:
    st.subheader("🔎 Data Type & Integrity Validation")
    st.caption("Validation is driven by the Feature Dictionary in the assignment: expected type, allowed values, missingness and logical ranges.")

    validation = validate_dataframe(df)
    st.dataframe(validation, use_container_width=True, hide_index=True)

    valid_rows = int((validation["Overall"] == "✅ VALID").sum())
    total_rows = len(validation)
    q1, q2, q3 = st.columns(3)
    q1.metric("Schema Checks Passed", f"{valid_rows}/{total_rows}")
    q2.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    q3.metric("Duplicate borrower_id", f"{int(df['borrower_id'].duplicated().sum()) if 'borrower_id' in df.columns else 'N/A'}")

    st.markdown("### 📦 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if valid_rows < total_rows:
        st.error("Schema validation failed. Fix the invalid columns before trusting model results.")

# -----------------------------
# EDA
# -----------------------------
with tabs[2]:
    st.subheader("📊 Exploratory Risk Analytics")
    if TARGET not in df.columns:
        st.warning("EDA target analysis requires is_flagged_misuse.")
    else:
        a, b = st.columns(2)
        with a:
            fig = px.histogram(df, x=TARGET, color=TARGET, title="Target Distribution", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.box(df, x=TARGET, y="high_risk_spend_ratio",
                         title="High-Risk Spend Ratio by Target")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📐 IQR Outlier Scanner")
        out_cols = [c for c in ["avg_tx_amount", "loan_amount"] if c in df.columns]
        out_rows = []
        for col in out_cols:
            q1, q3, iqr, low, high, count = iqr_stats(df, col)
            out_rows.append({
                "Feature": col, "Q1": q1, "Q3": q3, "IQR": iqr,
                "Lower Bound": low, "Upper Bound": high, "Outliers": count
            })
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

        st.markdown("### ⚡ Spending Velocity")
        vel = [c for c in ["pct_spent_48h", "pct_spent_7d", "cash_withdrawal_ratio",
                           "high_risk_spend_ratio"] if c in df.columns]
        long = df[vel].melt(var_name="Feature", value_name="Ratio")
        fig = px.violin(long, x="Feature", y="Ratio", box=True, points=False,
                        title="Velocity & Risk Ratio Distributions")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🎯 Composite Risk Score")
        score = (
            0.20*df["high_risk_spend_ratio"] +
            0.15*df["cash_withdrawal_ratio"] +
            0.15*df["international_tx_ratio"] +
            0.15*df["nighttime_tx_ratio"] +
            0.15*df["max_single_tx_pct"] +
            0.10*df["pct_spent_48h"] +
            0.10*df["mcc_mismatch_flag"]
        )
        temp = pd.DataFrame({"Risk Score": score, "Target": df[TARGET]})
        fig = px.box(temp, x="Target", y="Risk Score", color="Target",
                     title="Composite Risk Score vs Ground Truth")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Model
# -----------------------------
with tabs[3]:
    st.subheader("🤖 Model Lab")
    missing = [c for c in FEATURE_COLS + [TARGET] if c not in df.columns]
    if missing:
        st.error("Required columns are missing: " + ", ".join(missing))
    else:
        tune = st.checkbox("Run RandomizedSearchCV tuning (slower, more rigorous)", value=False)
        if st.button("🚀 Train & Evaluate Models", type="primary"):
            with st.spinner("Training models and evaluating on a stratified 80/20 split..."):
                results, X_test, y_test, best_params, best_score = train_models(df, tune)
            st.session_state["results"] = results
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["best_params"] = best_params
            st.session_state["best_score"] = best_score
            st.success("Training completed.")

        if "results" in st.session_state:
            results = st.session_state["results"]
            comparison = pd.DataFrame([
                {"Model": name, "Accuracy": r["accuracy"], "Precision": r["precision"],
                 "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["auc"]}
                for name, r in results.items()
            ]).sort_values("ROC-AUC", ascending=False)
            st.dataframe(comparison.style.format("{:.3f}", subset=["Accuracy","Precision","Recall","F1","ROC-AUC"]),
                         use_container_width=True, hide_index=True)

            best_name = comparison.iloc[0]["Model"]
            st.markdown(f"### 🏆 Best current model: **{best_name}**")
            r = results[best_name]

            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Accuracy", f"{r['accuracy']:.1%}")
            m2.metric("Precision", f"{r['precision']:.1%}")
            m3.metric("Recall", f"{r['recall']:.1%}")
            m4.metric("F1", f"{r['f1']:.1%}")
            m5.metric("ROC-AUC", f"{r['auc']:.3f}")

            cm = confusion_matrix(st.session_state["y_test"], r["pred"])
            fig = px.imshow(cm, text_auto=True, x=["Predicted Legit","Predicted Misuse"],
                            y=["Actual Legit","Actual Misuse"], title=f"Confusion Matrix — {best_name}")
            st.plotly_chart(fig, use_container_width=True)

            fpr, tpr, _ = roc_curve(st.session_state["y_test"], r["proba"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={r['auc']:.3f}"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random"))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

            if best_params:
                st.markdown("### 🔧 Tuned Hyperparameters")
                st.json({"Best Parameters": best_params, "CV Accuracy": best_score})

            # Feature importance / coefficients
            model = r["model"]
            clf = model.named_steps["classifier"]
            pre = model.named_steps["preprocessor"]
            try:
                names = pre.get_feature_names_out()
                if hasattr(clf, "feature_importances_"):
                    vals = clf.feature_importances_
                else:
                    vals = np.abs(clf.coef_[0])
                fi = pd.DataFrame({"Feature": names, "Importance": vals}).sort_values("Importance", ascending=False).head(10)
                fig = px.bar(fi.sort_values("Importance"), x="Importance", y="Feature",
                             orientation="h", title="Top 10 Model Drivers")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

# -----------------------------
# Live risk check
# -----------------------------
with tabs[4]:
    st.subheader("🎯 Live Risk Check")
    st.caption("Enter one borrower profile and let the trained model estimate misuse probability.")

    if "results" not in st.session_state:
        st.info("Train the models first from the AI Model tab.")
    else:
        # Prefer Random Forest if available
        model_name = "Random Forest" if "Random Forest" in st.session_state["results"] else list(st.session_state["results"])[0]
        model = st.session_state["results"][model_name]["model"]

        left, right = st.columns(2)
        with left:
            loan_amount = st.number_input("Loan Amount ($)", 1000.0, 200000.0, 50000.0)
            credit_score = st.number_input("Credit Score", 550, 850, 700)
            annual_income = st.number_input("Annual Income ($)", 10000.0, 500000.0, 100000.0)
            dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.38)
            employment = st.number_input("Employment Length (years)", 0, 50, 10)
            loans = st.number_input("Existing Loans", 0, 30, 3)
            age = st.number_input("Account Age (months)", 1, 240, 60)
            purpose = st.selectbox("Declared Purpose", SCHEMA["declared_purpose"]["allowed"])
            primary = st.selectbox("Primary MCC", SCHEMA["primary_mcc_category"]["allowed"])
            secondary = st.selectbox("Secondary MCC", SCHEMA["primary_mcc_category"]["allowed"] + ["Other Services"])
        with right:
            mismatch = st.selectbox("MCC Mismatch Flag", [0, 1])
            days = st.number_input("Days to First Transaction", 0, 365, 14)
            p48 = st.slider("Spent in 48h", 0.0, 1.0, 0.50)
            p7 = st.slider("Spent in 7d", 0.0, 1.0, 0.70)
            cash = st.slider("Cash Withdrawal Ratio", 0.0, 1.0, 0.30)
            high = st.slider("High Risk Spend Ratio", 0.0, 1.0, 0.20)
            intl = st.slider("International Transaction Ratio", 0.0, 1.0, 0.20)
            night = st.slider("Nighttime Transaction Ratio", 0.0, 1.0, 0.20)
            merchants = st.number_input("Unique Merchants", 1, 100, 15)
            txs = st.number_input("Total Transactions", 1, 500, 25)
            avg_tx = st.number_input("Average Transaction ($)", 10.0, 100000.0, 2500.0)
            max_tx = st.slider("Max Single Transaction %", 0.0, 1.0, 0.50)

        if st.button("🔍 Analyze Borrower", type="primary"):
            values = {
                "loan_amount": loan_amount, "credit_score": credit_score, "annual_income": annual_income,
                "dti_ratio": dti_ratio, "employment_length_years": employment, "num_existing_loans": loans,
                "account_age_months": age, "declared_purpose": purpose, "primary_mcc_category": primary,
                "secondary_mcc_category": secondary, "mcc_mismatch_flag": mismatch,
                "days_to_first_tx": days, "pct_spent_48h": p48, "pct_spent_7d": p7,
                "cash_withdrawal_ratio": cash, "high_risk_spend_ratio": high,
                "international_tx_ratio": intl, "nighttime_tx_ratio": night,
                "num_unique_merchants": merchants, "num_total_transactions": txs,
                "avg_tx_amount": avg_tx, "max_single_tx_pct": max_tx
            }
            p, band, icon = predict_single(model, values)
            st.markdown(f"## {icon} {band} RISK")
            st.progress(p)
            st.metric("Predicted misuse probability", f"{p:.1%}")
            if p >= .80:
                st.error("Immediate review recommended: multiple strong diversion signals are present.")
            elif p >= .60:
                st.warning("Enhanced review recommended before approving or releasing funds.")
            elif p >= .30:
                st.info("Moderate risk. Consider manual verification of the declared purpose and transaction behavior.")
            else:
                st.success("Low predicted diversion risk under the trained model.")

# -----------------------------
# Dictionary
# -----------------------------
with tabs[5]:
    st.subheader("📋 Feature Dictionary & Validation Rules")
    dd = pd.DataFrame([
        {
            "Column": col,
            "Expected Data Type": spec["type"],
            "Description": spec["description"],
            "Rules": (
                f"{spec.get('min','')}" + (f" → {spec.get('max','')}" if "max" in spec else "")
                if ("min" in spec or "max" in spec) else
                (", ".join(map(str, spec["allowed"])) if "allowed" in spec else "—")
            )
        } for col, spec in SCHEMA.items()
    ])
    st.dataframe(dd, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download validation report",
        validation.to_csv(index=False).encode("utf-8"),
        file_name="data_validation_report.csv",
        mime="text/csv"
    )

st.caption("FraudShield AI • Internship project demo • Model outputs are decision-support estimates, not automatic financial decisions.")
