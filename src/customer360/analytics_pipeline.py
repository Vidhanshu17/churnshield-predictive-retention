import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = PROJECT_ROOT / "exports"

ANALYTICS_VIEWS = {
    "churn_overview": "analytics.vw_churn_overview",
    "contract_analysis": "analytics.vw_contract_analysis",
    "customer_segments": "analytics.vw_customer_segments",
}


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url)

    try:
        with engine.connect() as connection:
            for export_name, view_name in ANALYTICS_VIEWS.items():
                query = text(f"SELECT * FROM {view_name}")
                dataframe = pd.read_sql_query(query, connection)

                output_file = EXPORT_DIR / f"{export_name}.csv"
                dataframe.to_csv(output_file, index=False)

                print(
                    f"{view_name}: exported {len(dataframe)} rows "
                    f"to {output_file.relative_to(PROJECT_ROOT)}"
                )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
