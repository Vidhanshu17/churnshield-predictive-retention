from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = PROJECT_ROOT / "models" / "churn_model.joblib"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}. Train the model first."
        )
    return joblib.load(MODEL_FILE)


def predict_customers(customer_data: pd.DataFrame) -> pd.DataFrame:
    if customer_data.empty:
        raise ValueError("Provide at least one customer record.")

    model = load_model()
    data = customer_data.copy()

    customer_ids = (
        data["customerID"].astype(str)
        if "customerID" in data.columns
        else pd.Series(
            [f"customer_{i + 1}" for i in range(len(data))],
            index=data.index,
        )
    )

    # The model was trained without identifiers or quality metadata.
    features = data.drop(
        columns=["customerID", "Churn", "is_valid", "quality_issues"],
        errors="ignore",
    )

    probabilities = model.predict_proba(features)
    class_labels = list(model.named_steps["classifier"].classes_)
    churn_probability = probabilities[:, class_labels.index("Yes")]
    predictions = model.predict(features)

    results = pd.DataFrame({
        "customerID": customer_ids.to_numpy(),
        "predicted_churn": predictions,
        "churn_probability_pct": (churn_probability * 100).round(2),
    })

    results["risk_category"] = pd.cut(
        results["churn_probability_pct"],
        bins=[-0.01, 30, 60, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return results


if __name__ == "__main__":
    dataset_path = (
        PROJECT_ROOT / "data" / "processed" / "customer_churn_validated.csv"
    )
    dataset = pd.read_csv(dataset_path)
    dataset = dataset[
        dataset["is_valid"].astype(str).str.lower().isin(["true", "1"])
    ].head(10)

    predictions = predict_customers(dataset)
    output_path = PROJECT_ROOT / "reports" / "sample_churn_predictions.csv"
    predictions.to_csv(output_path, index=False)

    print(predictions.to_string(index=False))
    print(f"\nSaved sample predictions to: {output_path}")
