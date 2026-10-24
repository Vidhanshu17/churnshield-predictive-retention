from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = PROJECT_ROOT / "data" / "processed" / "customer_churn_validated.csv"

COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
    "quality_issues": "quality_issues",
    "is_valid": "is_valid",
}


def load_to_postgres() -> dict:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Processed CSV not found: {CSV_FILE}")

    database_url = os.environ["DATABASE_URL"]
    df = pd.read_csv(CSV_FILE)

    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)

    # Convert blanks and pandas NaN to SQL NULL.
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce")
    df = df.astype(object).where(pd.notna(df), None)

    # PostgreSQL requires quality_issues to be non-NULL.
    df["quality_issues"] = df["quality_issues"].fillna("")

    engine = create_engine(database_url, pool_pre_ping=True)

    with engine.begin() as connection:
        connection.execute(text("""
            TRUNCATE TABLE analytics.customer_churn;
            TRUNCATE TABLE staging.customer_churn_validated;
        """))

        df.to_sql(
            "customer_churn_validated",
            con=connection,
            schema="staging",
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi",
        )

        connection.execute(text("""
            INSERT INTO analytics.customer_churn
            SELECT *
            FROM staging.customer_churn_validated
            WHERE is_valid = TRUE;
        """))

        staging_count = connection.execute(text("""
            SELECT COUNT(*) FROM staging.customer_churn_validated
        """)).scalar_one()

        analytics_count = connection.execute(text("""
            SELECT COUNT(*) FROM analytics.customer_churn
        """)).scalar_one()

        invalid_count = connection.execute(text("""
            SELECT COUNT(*)
            FROM staging.customer_churn_validated
            WHERE is_valid = FALSE
        """)).scalar_one()

    engine.dispose()

    result = {
        "csv_rows": int(len(df)),
        "staging_rows": int(staging_count),
        "analytics_rows": int(analytics_count),
        "flagged_rows": int(invalid_count),
    }

    if result["csv_rows"] != result["staging_rows"]:
        raise RuntimeError(f"Staging row-count mismatch: {result}")

    if result["analytics_rows"] + result["flagged_rows"] != result["staging_rows"]:
        raise RuntimeError(f"Analytics reconciliation failed: {result}")

    return result


if __name__ == "__main__":
    print(load_to_postgres())

