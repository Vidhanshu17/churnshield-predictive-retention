import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "customer_churn_validated.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "Churn"


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    return preprocessor


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required_columns = {"customerID", "Churn", "is_valid"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Train only on valid records. Remove identifiers and quality metadata
    # to prevent leakage or meaningless identifier-based learning.
    df = df[df["is_valid"].astype(str).str.lower().isin(["true", "1"])].copy()
    df = df.dropna(subset=[TARGET])

    X = df.drop(columns=["customerID", TARGET, "is_valid", "quality_issues"],
                errors="ignore")
    y = df[TARGET].astype(str).str.strip()

    if set(y.unique()) != {"Yes", "No"}:
        raise ValueError(f"Expected Churn labels Yes/No; got {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_results = {}

    scoring_functions = {
        "precision": make_scorer(precision_score, pos_label="Yes", zero_division=0),
        "recall": make_scorer(recall_score, pos_label="Yes", zero_division=0),
        "f1": make_scorer(f1_score, pos_label="Yes", zero_division=0),
        "roc_auc": "roc_auc",
    }

    models = {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    for model_name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", classifier),
        ])
        
        if model_name != "dummy_baseline":
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring_functions,
                n_jobs=1,
                error_score="raise",
            )

            cv_results[model_name] = {
                metric: {
                    "mean": round(float(scores[f"test_{metric}"].mean()), 4),
                    "std": round(float(scores[f"test_{metric}"].std()), 4),
                }
                for metric in ["precision", "recall", "f1", "roc_auc"]
            }

            print(f"\n--- {model_name}: 5-fold cross-validation ---")
            print(json.dumps(cv_results[model_name], indent=2))
        
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        class_labels = list(pipeline.named_steps["classifier"].classes_)
        yes_index = class_labels.index("Yes")
        churn_probabilities = probabilities[:, yes_index]

        metrics = {
            "model": model_name,
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_churn": round(float(
                precision_score(y_test, predictions, pos_label="Yes", zero_division=0)
            ), 4),
            "recall_churn": round(float(
                recall_score(y_test, predictions, pos_label="Yes", zero_division=0)
            ), 4),
            "f1_churn": round(float(
                f1_score(y_test, predictions, pos_label="Yes", zero_division=0)
            ), 4),
            "roc_auc": round(float(roc_auc_score(
                (y_test == "Yes").astype(int), churn_probabilities
            )), 4),
        }
        results.append(metrics)
        fitted_models[model_name] = pipeline

        print(f"\n--- {model_name} ---")
        print(json.dumps(metrics, indent=2))
        print("Confusion matrix (labels: No, Yes):")
        print(confusion_matrix(y_test, predictions, labels=["No", "Yes"]))
        print(classification_report(y_test, predictions, labels=["No", "Yes"],
                                    zero_division=0))

    # Select the model with the highest mean 5-fold CV F1-score.
    eligible_models = [
        item for item in results
        if item["model"] in cv_results
    ]
    best_result = max(
        eligible_models,
        key=lambda item: cv_results[item["model"]]["f1"]["mean"],
    )
    best_name = best_result["model"]
    best_model = fitted_models[best_name]

    joblib.dump(best_model, MODEL_DIR / "churn_model.joblib")

    report = {
        "dataset": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "valid_rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": TARGET,
        "positive_class": "Yes",
        "selection_metric": "cross_validation_f1_mean",
        "best_model": best_name,
        "best_model_metrics": best_result,
        "all_model_metrics": results,
        "cross_validation": cv_results,
        "note": (
            "Metrics are from a held-out stratified test split. "
            "They are not a guarantee of future performance."
        ),
    }

    report_path = REPORT_DIR / "churn_model_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nBest model by churn F1-score: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'churn_model.joblib'}")
    print(f"Saved evaluation report: {report_path}")


if __name__ == "__main__":
    main()



