
import argparse
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

REQUIRED_COLS = ["borrower_id"] + MODEL_COLS + [TARGET]

def validate_training_data(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    if df["borrower_id"].isna().any() or df["borrower_id"].duplicated().any():
        raise ValueError("borrower_id must be non-null and unique.")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("is_flagged_misuse must contain only 0/1.")

    for col in NUMERICAL_COLS:
        n = pd.to_numeric(df[col], errors="coerce")
        if n.isna().any():
            raise ValueError(f"{col} contains invalid/missing numeric values.")

    for col in ["credit_score"]:
        n = pd.to_numeric(df[col])
        if ((n < 550) | (n > 850)).any():
            raise ValueError("credit_score must be between 550 and 850.")

    ratio_cols = [
        "dti_ratio", "pct_spent_48h", "pct_spent_7d",
        "cash_withdrawal_ratio", "high_risk_spend_ratio",
        "international_tx_ratio", "nighttime_tx_ratio", "max_single_tx_pct"
    ]
    for col in ratio_cols:
        n = pd.to_numeric(df[col])
        if ((n < 0) | (n > 1)).any():
            raise ValueError(f"{col} must be between 0 and 1.")

    return df

def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("standard_scaling", StandardScaler(), NUMERICAL_COLS),
            (
                "one_hot_encoding",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="passthrough",
    )

    rf_pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=1,
            ),
        ),
    ])

    param_distributions = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
    }

    return rf_pipeline, param_distributions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training CSV path")
    parser.add_argument(
        "--output",
        default="model_artifact.joblib",
        help="Output model artifact path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    validate_training_data(df)

    X = df[MODEL_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=1,
    )

    rf_pipeline, params = build_model()

    search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=params,
        n_iter=10,
        scoring="accuracy",
        random_state=1,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    model = search.best_estimator_
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "classification_report": classification_report(
            y_test, pred, output_dict=True
        ),
    }

    artifact = {
        "model": model,
        "feature_columns": MODEL_COLS,
        "target": TARGET,
        "metrics": metrics,
        "schema_version": "1.0",
    }

    joblib.dump(artifact, args.output)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model artifact to: {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
