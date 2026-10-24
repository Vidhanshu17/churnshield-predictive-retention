
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

st.set_page_config(
    page_title="ChurnShield | Predictive Retention",
    page_icon="📊",
    layout="wide",
)

st.title("ChurnShield")
st.caption("Customer analytics • Churn intelligence • Revenue insights")

DATABASE_URL = os.getenv("DATABASE_URL")


@st.cache_resource
def get_engine(database_url):
    return create_engine(database_url, pool_pre_ping=True)


@st.cache_data(ttl=60)
def load_view(database_url, view_name):
    allowed_views = {
        "analytics.vw_churn_overview",
        "analytics.vw_contract_analysis",
        "analytics.vw_customer_segments",
    }
    if view_name not in allowed_views:
        raise ValueError("Unsupported analytics view")

    with get_engine(database_url).connect() as connection:
        return pd.read_sql_query(
            text(f"SELECT * FROM {view_name}"),
            connection,
        )


if not DATABASE_URL:
    st.error("DATABASE_URL is not configured.")
    st.stop()

try:
    overview = load_view(DATABASE_URL, "analytics.vw_churn_overview")
    contracts = load_view(DATABASE_URL, "analytics.vw_contract_analysis")
    segments = load_view(DATABASE_URL, "analytics.vw_customer_segments")

except (SQLAlchemyError, ValueError) as exc:
    st.error("Unable to load analytics from PostgreSQL.")
    st.exception(exc)
    st.stop()

if overview.empty:
    st.warning("The churn overview view returned no records.")
    st.stop()

summary = overview.iloc[0]

total_customers = int(summary["total_customers"])
churned_customers = int(summary["churned_customers"])
retained_customers = int(summary["retained_customers"])
churn_rate = float(summary["churn_rate_pct"])
monthly_charges = float(summary["total_monthly_charges"])
avg_monthly_charges = float(summary["avg_monthly_charges"])

st.success("Connected to PostgreSQL • Analytics views loaded")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid Customers", f"{total_customers:,}")
c2.metric("Churned Customers", f"{churned_customers:,}")
c3.metric("Retained Customers", f"{retained_customers:,}")
c4.metric("Churn Rate", f"{churn_rate:.2f}%")

c5, c6 = st.columns(2)
c5.metric("Total Monthly Charges", f"{monthly_charges:,.2f}")
c6.metric("Average Monthly Charges", f"{avg_monthly_charges:,.2f}")

st.divider()

st.subheader("Contract Analysis")
st.caption("Churn rates differ by contract type; these are observed associations.")
contract_chart = px.bar(
    contracts.sort_values("churn_rate_pct", ascending=False),
    x="contract",
    y="churn_rate_pct",
    color="contract",
    text="churn_rate_pct",
    labels={
        "contract": "Contract Type",
        "churn_rate_pct": "Churn Rate (%)",
    },
)
contract_chart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
contract_chart.update_layout(showlegend=False, yaxis_range=[0, 100])
st.plotly_chart(contract_chart, use_container_width=True)

st.subheader("Customer Tenure Segments")
segment_chart = px.bar(
    segments.sort_values("churn_rate_pct", ascending=False),
    x="tenure_segment",
    y="churn_rate_pct",
    color="tenure_segment",
    text="churn_rate_pct",
    labels={
        "tenure_segment": "Tenure Segment",
        "churn_rate_pct": "Churn Rate (%)",
    },
)
segment_chart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
segment_chart.update_layout(showlegend=False, yaxis_range=[0, 100])
st.plotly_chart(segment_chart, use_container_width=True)

with st.expander("View contract analysis data"):
    st.dataframe(contracts, use_container_width=True, hide_index=True)

with st.expander("View customer segment data"):
    st.dataframe(segments, use_container_width=True, hide_index=True)

st.caption(
    "Source: validated IBM Telco Customer Churn dataset. "
    "Metrics describe observed historical records and are not causal estimates."
)


import json
import joblib

st.divider()
st.header("AI Risk Workbench")
st.caption("Model-based churn risk estimates for validated customer records.")

MODEL_PATH = "/app/models/churn_model.joblib"
DATA_PATH = "/app/data/processed/customer_churn_validated.csv"
REPORT_PATH = "/app/reports/churn_model_evaluation.json"


@st.cache_resource
def load_churn_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_customer_data():
    data = pd.read_csv(DATA_PATH)
    return data[
        data["is_valid"].astype(str).str.lower().isin(["true", "1"])
    ].copy()


@st.cache_data
def load_model_report():
    with open(REPORT_PATH, encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_churn_model()
    customers = load_customer_data()
    model_report = load_model_report()

    feature_columns = [
        column for column in customers.columns
        if column not in {
            "customerID", "Churn", "is_valid", "quality_issues"
        }
    ]

    features = customers[feature_columns]
    churn_index = list(
        model.named_steps["classifier"].classes_
    ).index("Yes")

    probabilities = model.predict_proba(features)[:, churn_index]
    predictions = model.predict(features)

    risk_data = pd.DataFrame({
        "customerID": customers["customerID"].astype(str).values,
        "actual_churn": customers["Churn"].astype(str).values,
        "predicted_churn": predictions,
        "churn_probability_pct": (probabilities * 100).round(2),
    })

    risk_data["risk_category"] = pd.cut(
        risk_data["churn_probability_pct"],
        bins=[-0.01, 30, 60, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    st.subheader("Model Performance")
    metrics = model_report["best_model_metrics"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Model", model_report["best_model"].replace("_", " ").title())
    m2.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
    m3.metric("Churn Recall", f"{metrics['recall_churn']:.2%}")
    m4.metric("Churn F1-score", f"{metrics['f1_churn']:.4f}")

    st.caption(
        "Metrics above come from the held-out test set. "
        "Customer risk scores below are in-sample estimates for the "
        "validated dataset, not future churn outcomes."
    )
    
    # Display 5-fold cross-validation metrics
    st.subheader("5-Fold Cross-Validation")

    cv_results = model_report.get("cross_validation", {})
    selected_cv = cv_results.get(model_report["best_model"])

    if selected_cv:
        cv1, cv2, cv3, cv4 = st.columns(4)

        cv1.metric(
            "CV Precision",
            f"{selected_cv['precision']['mean']:.2%}",
            help=f"Standard deviation: {selected_cv['precision']['std']:.4f}",
        )
        cv2.metric(
            "CV Recall",
            f"{selected_cv['recall']['mean']:.2%}",
            help=f"Standard deviation: {selected_cv['recall']['std']:.4f}",
        )
        cv3.metric(
            "CV F1-score",
            f"{selected_cv['f1']['mean']:.2%}",
            help=f"Standard deviation: {selected_cv['f1']['std']:.4f}",
        )
        cv4.metric(
            "CV ROC-AUC",
            f"{selected_cv['roc_auc']['mean']:.2%}",
            help=f"Standard deviation: {selected_cv['roc_auc']['std']:.4f}",
        )

        st.caption(
            "Mean scores across five stratified folds of the training data. "
            "Lower standard deviation generally indicates more consistent "
            "performance across folds."
        )
    else:
        st.warning("Cross-validation results are unavailable in the model report.")


    st.subheader("Customer Risk Explorer")

    risk_filter = st.selectbox(
        "Filter by risk category",
        ["All", "High", "Medium", "Low"],
    )

    display_data = risk_data.copy()
    if risk_filter != "All":
        display_data = display_data[
            display_data["risk_category"] == risk_filter
        ]

    display_data = display_data.sort_values(
        "churn_probability_pct", ascending=False
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download customer risk predictions",
        data=display_data.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_risk_predictions.csv",
        mime="text/csv",
    )

except Exception as exc:
    st.error("Unable to load the churn prediction model or its data.")
    st.exception(exc)



st.divider()
st.header("Retention Decision Simulator")
st.caption(
    "Simulate a hypothetical retention campaign for High- and Medium-risk "
    "customers. Results depend on your assumptions, not measured campaign outcomes."
)

eligible = risk_data[
    risk_data["risk_category"].isin(["High", "Medium"])
].copy()

if eligible.empty:
    st.warning("No High- or Medium-risk customers are available.")
else:
    # Attach monthly charges from the validated customer dataset.
    customer_charges = customers[["customerID", "MonthlyCharges"]].copy()
    customer_charges["customerID"] = customer_charges["customerID"].astype(str)

    eligible = eligible.merge(
        customer_charges,
        on="customerID",
        how="left",
        validate="one_to_one",
    )

    eligible["MonthlyCharges"] = pd.to_numeric(
        eligible["MonthlyCharges"], errors="coerce"
    )
    eligible = eligible.dropna(subset=["MonthlyCharges"])

    left, right = st.columns(2)

    with left:
        target_percentage = st.slider(
            "Percentage of eligible customers to target",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
        )
        success_rate = st.slider(
            "Assumed campaign success rate (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
        )

    with right:
        offer_cost = st.number_input(
            "Offer cost per targeted customer (currency units)",
            min_value=0.0,
            max_value=10000.0,
            value=200.0,
            step=50.0,
        )
        retention_months = st.slider(
            "Revenue retention horizon (months)",
            min_value=1,
            max_value=12,
            value=3,
        )

    eligible_count = len(eligible)
    target_count = round(eligible_count * target_percentage / 100)

    # Deterministic targeting: highest predicted churn probability first.
    targeted = eligible.sort_values(
        "churn_probability_pct", ascending=False
    ).head(target_count)

    estimated_saved = target_count * success_rate / 100
    average_monthly_charges = (
        float(targeted["MonthlyCharges"].mean()) if target_count else 0.0
    )

    estimated_retained_monthly_revenue = (
        estimated_saved * average_monthly_charges
    )
    estimated_revenue_over_horizon = (
        estimated_retained_monthly_revenue * retention_months
    )
    campaign_cost = target_count * offer_cost
    estimated_net_impact = estimated_revenue_over_horizon - campaign_cost

    st.subheader("Simulation Results")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Eligible Customers", f"{eligible_count:,}")
    k2.metric("Customers Targeted", f"{target_count:,}")
    k3.metric("Estimated Retained", f"{estimated_saved:,.1f}")
    k4.metric("Campaign Cost", f"{campaign_cost:,.2f}")

    k5, k6 = st.columns(2)
    k5.metric(
        f"Estimated Revenue ({retention_months} months)",
        f"{estimated_revenue_over_horizon:,.2f}",
    )
    k6.metric(
        "Estimated Net Impact",
        f"{estimated_net_impact:,.2f}",
        delta=f"{estimated_net_impact:,.2f}",
    )

    st.caption(
        "Assumptions: targeted customers are selected by highest predicted risk; "
        "success rate is hypothetical; monthly charges are used as a revenue proxy. "
        "Revenue is not profit, and this estimate excludes discount funding beyond "
        "the entered offer cost, servicing costs, taxes, and other expenses. "
        "The calculation assumes retained revenue persists throughout the chosen "
        "horizon and does not establish causal campaign effectiveness."
    )

    with st.expander("View targeted customer list"):
        st.dataframe(
            targeted[
                [
                    "customerID",
                    "predicted_churn",
                    "churn_probability_pct",
                    "risk_category",
                    "MonthlyCharges",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download targeted customers",
            data=targeted.to_csv(index=False).encode("utf-8"),
            file_name="retention_campaign_targets.csv",
            mime="text/csv",
        )
