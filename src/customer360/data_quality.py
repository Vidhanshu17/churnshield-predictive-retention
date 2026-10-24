from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "Churn",
}

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw_data(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the source CSV while preserving its original file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, dtype={"customerID": "string"})


def validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate rows and return an auditable quality report."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    result = df.copy()
    source_rows = len(result)

    # Convert numeric fields explicitly; invalid values become missing.
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    # The source dataset can contain blank TotalCharges values.
    # Keep these rows but flag them for review.
    result["quality_issues"] = ""

    def add_issue(mask: pd.Series, message: str) -> None:
        current = result.loc[mask, "quality_issues"]
        result.loc[mask, "quality_issues"] = current.apply(
            lambda value: f"{value};{message}" if value else message
        )

    add_issue(result["customerID"].isna() | result["customerID"].str.strip().eq(""),
              "missing_customer_id")
    add_issue(result["Churn"].isna() | ~result["Churn"].isin(["Yes", "No"]),
              "invalid_churn_label")
    add_issue(result["tenure"].isna() | ~result["tenure"].between(0, 1000),
              "invalid_tenure")
    add_issue(result["MonthlyCharges"].isna() | (result["MonthlyCharges"] < 0),
              "invalid_monthly_charges")
    add_issue(result["TotalCharges"].isna() | (result["TotalCharges"] < 0),
              "invalid_total_charges")

    duplicate_mask = result.duplicated(keep=False)
    add_issue(duplicate_mask, "duplicate_row")

    result["is_valid"] = result["quality_issues"].eq("")

    report = {
        "source_file": RAW_FILE.name,
        "source_rows": int(source_rows),
        "source_columns": int(len(df.columns)),
        "duplicate_rows_involved": int(duplicate_mask.sum()),
        "missing_values_by_column": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        },
        "invalid_rows": int((~result["is_valid"]).sum()),
        "valid_rows": int(result["is_valid"].sum()),
        "churn_distribution": {
            str(key): int(value)
            for key, value in df["Churn"].value_counts(dropna=False).items()
        },
        "issues_by_type": {
            issue: int(result["quality_issues"].str.contains(
                issue, regex=False
            ).sum())
            for issue in [
                "missing_customer_id",
                "invalid_churn_label",
                "invalid_tenure",
                "invalid_monthly_charges",
                "invalid_total_charges",
                "duplicate_row",
            ]
        },
    }

    return result, report


def run_pipeline() -> dict:
    """Create the processed dataset and JSON quality report."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    validated, report = validate_data(raw)

    validated.to_csv(PROCESSED_DIR / "customer_churn_validated.csv", index=False)

    with (PROCESSED_DIR / "data_quality_report.json").open(
        "w", encoding="utf-8"
    ) as report_file:
        json.dump(report, report_file, indent=2)

    return report


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
