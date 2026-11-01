"""
Unit tests for TextPreprocessor.
"""

import pytest
from src.text_preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_clean_text_removes_urls_and_emails(preprocessor):
    raw_text = "Contact me at candidate@test.com or visit https://myportfolio.dev for project details."
    cleaned = preprocessor.clean_text(raw_text)
    assert "candidate@test.com" not in cleaned
    assert "https" not in cleaned
    assert "myportfolio.dev" not in cleaned
    assert "project details" in cleaned


def test_clean_text_preserves_technical_tokens(preprocessor):
    raw_text = "Proficient in C++, C#, .NET, Node.js, and CI/CD pipelines."
    cleaned = preprocessor.clean_text(raw_text, preserve_case=True)
    assert "C++" in cleaned
    assert "C#" in cleaned
    assert "Node.js" in cleaned
    assert "CI/CD" in cleaned


def test_tokenize_and_lemmatize(preprocessor):
    text = "Engineers are engineering scalable solutions using Python models"
    tokens = preprocessor.tokenize_and_lemmatize(text)
    assert "engineer" in tokens or "engineers" in tokens
    assert "python" in tokens
    assert "are" not in tokens  # Stopword removed


def test_section_extraction(preprocessor):
    resume = """
    John Doe
    SUMMARY
    Senior Engineer with 5 years experience.
    SKILLS
    Python, Docker, Kubernetes
    EXPERIENCE
    Built microservices at Acme Corp.
    EDUCATION
    BS in Computer Science
    """
    sections = preprocessor.extract_sections(resume)
    assert "skills" in sections
    assert "Python" in sections["skills"]
    assert "experience" in sections
    assert "education" in sections
