"""
Unit Tests for Inference and Decision Engine
"""

import os
import pytest
import pandas as pd
from src.inference import get_classifier


@pytest.fixture(scope="module")
def classifier():
    return get_classifier(models_dir="models")


def test_classifier_loads(classifier):
    assert classifier.cat_model is not None
    assert classifier.prio_model is not None
    assert classifier.vectorizer is not None


def test_predict_critical_technical_issue(classifier):
    res = classifier.predict_ticket(
        subject="CRITICAL: Production API 500 error outage",
        description="Our whole application is down and returning 500 internal errors. Users cannot checkout!",
        customer_tier="Enterprise"
    )
    assert res["category"] == "Technical Issue"
    assert res["priority"] in ["Critical", "High"]
    assert "Tier 2" in res["assigned_team"] or "DevOps" in res["assigned_team"]
    assert res["sla_resolution_hours"] <= 4


def test_predict_billing_issue(classifier):
    res = classifier.predict_ticket(
        subject="Unauthorized charge on invoice #9921",
        description="We were charged $5,000 erroneously. Please refund this transaction immediately.",
        customer_tier="Business"
    )
    assert res["category"] in ["Billing & Invoicing", "Cancellation & Refund"]
    assert "Billing" in res["assigned_team"] or "Retention" in res["assigned_team"]


def test_predict_batch(classifier):
    df = pd.DataFrame([
        {
            "ticket_id": "TCK-TEST-01",
            "ticket_subject": "Server crash 500",
            "ticket_description": "Our server died after memory leak"
        },
        {
            "ticket_id": "TCK-TEST-02",
            "ticket_subject": "Invoice question",
            "ticket_description": "Can you provide VAT receipt for tax purposes?"
        }
    ])
    res_df = classifier.predict_batch(df)
    assert "predicted_category" in res_df.columns
    assert "predicted_priority" in res_df.columns
    assert "assigned_team" in res_df.columns
    assert len(res_df) == 2
