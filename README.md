# Customer 360 AI Decision Intelligence Platform

A portfolio project for customer analytics, churn prediction, segmentation,
and transparent retention decision support.

## Current stage

Project foundation and Docker setup.

## Planned stack

- Python, Pandas, scikit-learn
- PostgreSQL and SQL
- Streamlit and Plotly
- Docker Compose
- Pytest

## Local setup

1. Copy `.env.example` to `.env` and set a local development password.
2. Start the containers with `docker compose up --build -d`.
3. Open http://localhost:8501.
4. Check container health with `docker compose ps`.

Never commit `.env` or credentials to version control.

## Data limitations

Predictions and retention simulations will be clearly distinguished from
observed outcomes. Any value estimate will be labeled according to the
available data and assumptions.
