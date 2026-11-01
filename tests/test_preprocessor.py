"""
Unit Tests for Text Preprocessor and Feature Extraction
"""

import pytest
from src.text_preprocessor import clean_text, extract_ticket_signals, expand_contractions, TextCleanerTransformer


def test_expand_contractions():
    raw = "I can't access my account and won't be able to pay."
    expanded = expand_contractions(raw)
    assert "cannot" in expanded
    assert "will not" in expanded


def test_clean_text_removes_urls_and_emails():
    raw = "Please visit https://status.acme.com or email support@acme.com for updates!"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "acme.com" not in cleaned
    assert "@" not in cleaned


def test_clean_text_preserves_critical_negation():
    raw = "The system is down and does not work at all."
    cleaned = clean_text(raw)
    assert "down" in cleaned
    assert "not" in cleaned or "work" in cleaned


def test_extract_ticket_signals():
    text = "EMERGENCY: Server is DOWN! Immediate assistance needed!!!"
    signals = extract_ticket_signals(text)
    assert signals["exclamation_count"] >= 3
    assert signals["urgency_score"] >= 1
    assert signals["has_critical_keyword"] is True
    assert signals["uppercase_ratio"] > 0.25


def test_transformer_interface():
    transformer = TextCleanerTransformer()
    docs = ["First ticket with error", "Second ticket with billing"]
    res = transformer.transform(docs)
    assert len(res) == 2
    assert isinstance(res[0], str)
