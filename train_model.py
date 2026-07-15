from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
TARGET = "approved"

CATEGORICAL_FEATURES = [
    "gender",
    "income_type",
    "education",
    "family_status",
    "housing_type",
    "employment_status",
]

NUMERIC_FEATURES = [
    "annual_income",
    "employment_years",
    "age",
    "credit_history_years",
    "existing_loan_balance",
    "credit_inquiries",
    "past_due_count",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_demo_dataset(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], rows),
            "income_type": rng.choice(["Working", "Commercial associate", "Pensioner", "Student"], rows),
            "education": rng.choice(["Secondary", "Higher education", "Incomplete higher"], rows),
            "family_status": rng.choice(["Single", "Married", "Separated"], rows),
            "housing_type": rng.choice(["House", "Rented apartment", "With parents"], rows),
            "employment_status": rng.choice(["Employed", "Self-employed", "Unemployed"], rows),
            "annual_income": rng.normal(62000, 26000, rows).clip(12000, 220000),
            "employment_years": rng.normal(6, 4, rows).clip(0, 35),
            "age": rng.normal(39, 11, rows).clip(18, 70),
            "credit_history_years": rng.normal(5, 3, rows).clip(0, 25),
            "existing_loan_balance": rng.normal(18000, 16000, rows).clip(0, 130000),
            "credit_inquiries": rng.poisson(2, rows).clip(0, 12),
            "past_due_count": rng.poisson(0.7, rows).clip(0, 8),
        }
    )

    risk_score = (
        0.000045 * frame["annual_income"]
        + 0.12 * frame["employment_years"]
        + 0.15 * frame["credit_history_years"]
        - 0.00006 * frame["existing_loan_balance"]
        - 0.45 * frame["credit_inquiries"]
        - 0.85 * frame["past_due_count"]
        + frame["education"].eq("Higher education").astype(float) * 0.45
        + frame["employment_status"].eq("Unemployed").astype(float) * -1.4
    )
    probability = 1 / (1 + np.exp(-(risk_score - 2.2)))
    frame[TARGET] = (rng.random(rows) < probability).astype(int)
    return frame


def normalize_target(frame: pd.DataFrame) -> pd.DataFrame:
    if TARGET in frame.columns:
        frame[TARGET] = frame[TARGET].astype(int)
        return frame

    status_columns = [column for column in frame.columns if column.lower().startswith("status")]
    if not status_columns:
        raise ValueError(
            f"Dataset must contain `{TARGET}` or payment status columns such as STATUS."
        )

    risk_values = {"1", "2", "3", "4", "5", "C", "X"}
    status_text = frame[status_columns].astype(str)
    high_risk = status_text.apply(lambda row: any(value in risk_values for value in row), axis=1)
    frame[TARGET] = (~high_risk).astype(int)
    return frame


def load_dataset(path: str | None) -> pd.DataFrame:
    if path:
        frame = pd.read_csv(path)
        frame = normalize_target(frame)
        missing_features = [feature for feature in FEATURES if feature not in frame.columns]
        if missing_features:
            raise ValueError(f"Dataset is missing required features: {', '.join(missing_features)}")
        return frame[FEATURES + [TARGET]]

    return make_demo_dataset()


def build_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=random_state),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    return models


def build_pipeline(classifier: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train(dataset_path: str | None = None) -> dict[str, object]:
    frame = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=0.2,
        random_state=42,
        stratify=frame[TARGET],
    )

    results = []
    best_bundle = None
    for name, classifier in build_models().items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        results.append({"model": name, "accuracy": accuracy, "f1_score": f1})

        if best_bundle is None or f1 > best_bundle["f1_score"]:
            best_bundle = {
                "pipeline": pipeline,
                "model_name": name,
                "features": FEATURES,
                "f1_score": f1,
                "accuracy": accuracy,
            }

    assert best_bundle is not None
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(best_bundle, MODEL_DIR / "credit_card_approval_model.joblib")

    pd.DataFrame(results).sort_values("f1_score", ascending=False).to_csv(
        REPORT_DIR / "model_comparison.csv", index=False
    )
    report = classification_report(y_test, best_bundle["pipeline"].predict(x_test))
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    return best_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit card approval classifiers.")
    parser.add_argument("--data", help="Optional CSV containing engineered applicant records.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bundle = train(args.data)
    print(
        f"Saved {bundle['model_name']} with F1={bundle['f1_score']:.3f} "
        f"and accuracy={bundle['accuracy']:.3f}"
    )
