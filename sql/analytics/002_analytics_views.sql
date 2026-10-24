CREATE OR REPLACE VIEW analytics.vw_churn_overview AS
SELECT
    COUNT(*) AS total_customers,
    COUNT(*) FILTER (WHERE churn = 'Yes') AS churned_customers,
    COUNT(*) FILTER (WHERE churn = 'No') AS retained_customers,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE churn = 'Yes')
        / NULLIF(COUNT(*), 0), 2
    ) AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 2) AS total_monthly_charges,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM analytics.customer_churn;


CREATE OR REPLACE VIEW analytics.vw_contract_analysis AS
SELECT
    contract,
    COUNT(*) AS total_customers,
    COUNT(*) FILTER (WHERE churn = 'Yes') AS churned_customers,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE churn = 'Yes')
        / NULLIF(COUNT(*), 0), 2
    ) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure), 1) AS avg_tenure_months
FROM analytics.customer_churn
GROUP BY contract;


CREATE OR REPLACE VIEW analytics.vw_customer_segments AS
SELECT
    CASE
        WHEN tenure < 12 THEN 'New (0-11 months)'
        WHEN tenure < 24 THEN 'Developing (12-23 months)'
        WHEN tenure < 48 THEN 'Established (24-47 months)'
        ELSE 'Loyal (48+ months)'
    END AS tenure_segment,
    COUNT(*) AS total_customers,
    COUNT(*) FILTER (WHERE churn = 'Yes') AS churned_customers,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE churn = 'Yes')
        / NULLIF(COUNT(*), 0), 2
    ) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM analytics.customer_churn
GROUP BY 1;
