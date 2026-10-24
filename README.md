# ChurnShield — Predictive Customer Retention Engine

ChurnShield is an end-to-end customer churn analytics and retention decision-support platform built with Python, PostgreSQL, Scikit-learn, Streamlit, and Docker.

It combines data quality validation, SQL-based customer analytics, machine-learning churn risk scoring, and a retention campaign simulator in one interactive application.

## Key Features

- **Data Quality Validation:** Detect invalid records, missing values, duplicate rows, and data-quality issues.
- **PostgreSQL Analytics:** Store validated customer records and generate reusable analytical views.
- **Churn Analysis:** Explore overall churn, contract-based churn, and customer tenure segments.
- **Machine Learning:** Compare baseline, Logistic Regression, and Random Forest models.
- **Cross-Validation:** Evaluate candidate models using stratified 5-fold cross-validation.
- **Customer Risk Explorer:** Review predicted churn risk scores and filter customers by risk category.
- **Retention Decision Simulator:** Estimate hypothetical campaign costs, retained revenue, and net impact using adjustable assumptions.
- **Interactive Dashboard:** Explore analytics and model outputs through Streamlit and Plotly.
- **Dockerized Setup:** Run the application and PostgreSQL database using Docker Compose.
- **Automated Tests:** Validate data-quality logic and basic application behavior with Pytest.

## Technology Stack

- Python, Pandas, NumPy
- Scikit-learn
- PostgreSQL and SQL
- Streamlit and Plotly
- Docker and Docker Compose
- Pytest

## Architecture Overview

1. Load the customer churn dataset.
2. Validate records and identify data-quality issues.
3. Load validated customer data into PostgreSQL.
4. Generate SQL analytics views for churn and customer segmentation.
5. Train and evaluate candidate churn prediction models.
6. Explore customer risk scores in the Streamlit dashboard.
7. Simulate hypothetical retention campaign scenarios.

## Getting Started

### Prerequisites

- Docker Desktop with Docker Compose
- Git

### Setup

1. Clone the repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd churnshield-predictive-retention
   ```

2. Create your local environment file:

   ```bash
   cp .env.example .env
   ```

   On Windows PowerShell, use:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Update `.env` with a local development password and ensure the database connection settings match your Docker Compose configuration.

4. Build and start the services:

   ```bash
   docker compose up --build -d
   ```

5. Open the dashboard:

   http://localhost:8501

6. Check service status:

   ```bash
   docker compose ps
   ```

7. Run the automated tests:

   ```bash
   docker compose exec app python -m pytest tests -v
   ```

## Model Evaluation

The project compares a Dummy baseline, Logistic Regression, and Random Forest using a held-out test set and stratified 5-fold cross-validation.

The current cross-validation results used for model selection are:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Precision | 52.18% | 54.97% |
| Recall | 80.20% | 71.44% |
| F1-score | 63.22% | 62.13% |
| ROC-AUC | 84.59% | 83.95% |

Logistic Regression is selected based on its higher mean cross-validation F1-score. Test-set results are reported separately in the model evaluation report.

## Data and Evaluation Limitations

- Customer risk scores generated for the validated dataset are in-sample estimates, not guaranteed future churn outcomes.
- The retention simulator uses hypothetical campaign success rates and revenue assumptions. Its outputs are estimates, not measured campaign results or proof of causal impact.
- Model performance can change with different datasets, populations, and deployment conditions.

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── sql/
│   ├── analytics/
│   └── schema/
├── src/
│   └── customer360/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Security Notes

- Never commit `.env`, passwords, API keys, or other secrets.
- The raw and processed customer datasets and serialized model artifacts are excluded through `.gitignore`.
- Use only authorized data and appropriate access controls for real customer information.

## License

No license has been added yet. Add a license before permitting others to reuse or redistribute this project.