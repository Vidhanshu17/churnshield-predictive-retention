CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS staging.customer_churn_validated (
    customer_id TEXT PRIMARY KEY,
    gender TEXT,
    senior_citizen INTEGER,
    partner TEXT,
    dependents TEXT,
    tenure INTEGER,
    phone_service TEXT,
    multiple_lines TEXT,
    internet_service TEXT,
    online_security TEXT,
    online_backup TEXT,
    device_protection TEXT,
    tech_support TEXT,
    streaming_tv TEXT,
    streaming_movies TEXT,
    contract TEXT,
    paperless_billing TEXT,
    payment_method TEXT,
    monthly_charges NUMERIC(12, 2),
    total_charges NUMERIC(12, 2),
    churn TEXT,
    quality_issues TEXT NOT NULL DEFAULT '',
    is_valid BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS analytics.customer_churn (
    LIKE staging.customer_churn_validated INCLUDING ALL
);
