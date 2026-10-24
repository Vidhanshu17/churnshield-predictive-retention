import pandas as pd
import pytest

from customer360.data_quality import REQUIRED_COLUMNS, validate_data


def make_valid_row():
    return {
        "customerID": "TEST-0001",
        "gender": "Female",
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
        "Contract": "Month-to-month",
        "Churn": "No",
    }


def test_valid_customer_passes_validation():
    df = pd.DataFrame([make_valid_row()])

    validated, report = validate_data(df)

    assert validated.loc[0, "is_valid"]
    assert report["source_rows"] == 1
    assert report["valid_rows"] == 1
    assert report["invalid_rows"] == 0


def test_invalid_churn_label_is_flagged():
    row = make_valid_row()
    row["Churn"] = "Maybe"

    validated, report = validate_data(pd.DataFrame([row]))

    assert not validated.loc[0, "is_valid"]
    assert "invalid_churn_label" in validated.loc[0, "quality_issues"]
    assert report["invalid_rows"] == 1


def test_blank_total_charges_is_flagged():
    row = make_valid_row()
    row["TotalCharges"] = ""

    validated, report = validate_data(pd.DataFrame([row]))

    assert not validated.loc[0, "is_valid"]
    assert "invalid_total_charges" in validated.loc[0, "quality_issues"]


def test_duplicate_rows_are_flagged():
    row = make_valid_row()
    df = pd.DataFrame([row, row])

    validated, report = validate_data(df)

    assert report["duplicate_rows_involved"] == 2
    assert not validated["is_valid"].any()


def test_missing_required_column_raises_error():
    df = pd.DataFrame([{"customerID": "TEST-0001"}])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df)


def test_negative_monthly_charges_are_flagged():
    row = make_valid_row()
    row["MonthlyCharges"] = -1

    validated, report = validate_data(pd.DataFrame([row]))

    assert not validated.loc[0, "is_valid"]
    assert "invalid_monthly_charges" in validated.loc[0, "quality_issues"]
