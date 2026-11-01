"""
Unit tests for ResumeMatcher.
"""

import pytest
from src.matcher import ResumeMatcher


@pytest.fixture
def matcher():
    return ResumeMatcher()


def test_semantic_similarity(matcher):
    resume = "Senior Python developer experienced in building web APIs with FastAPI and Docker."
    jd = "Looking for a Python backend engineer with FastAPI and Docker containerization."
    sim = matcher.compute_semantic_similarity(resume, jd)
    assert 0.0 <= sim <= 1.0
    assert sim > 0.10


def test_evaluate_skills_perfect_match(matcher):
    candidate_skills = ["Python", "FastAPI", "Docker", "AWS", "Git"]
    required_skills = ["Python", "FastAPI", "Docker"]
    preferred_skills = ["AWS", "Git"]

    result = matcher.evaluate_skills(candidate_skills, required_skills, preferred_skills)
    assert result["skill_score"] == 100.0
    assert len(result["missing_required"]) == 0
    assert set(result["matched_required"]) == set(required_skills)


def test_evaluate_skills_with_gaps(matcher):
    candidate_skills = ["Python", "Git"]
    required_skills = ["Python", "Kubernetes", "AWS"]
    preferred_skills = ["Docker"]

    result = matcher.evaluate_skills(candidate_skills, required_skills, preferred_skills)
    assert result["skill_score"] < 100.0
    assert "Kubernetes" in result["missing_required"]
    assert "AWS" in result["missing_required"]
    assert "Python" in result["matched_required"]


def test_full_resume_match(matcher):
    resume = """
    Alex Chen
    Experienced Machine Learning Engineer with 5 years experience.
    Proficient in Python, PyTorch, Scikit-Learn, Docker, and AWS.
    Built NLP pipelines and FastAPI services.
    """
    jd = {
        "title": "Machine Learning Engineer",
        "description": "Looking for ML Engineer skilled in Python, PyTorch, Scikit-Learn, Docker and AWS.",
        "required_skills": ["Python", "PyTorch", "Scikit-Learn", "Docker", "AWS"],
        "preferred_skills": ["FastAPI", "NLP"],
        "experience_years": 4
    }

    match = matcher.match_resume(resume, jd, filename="resume_alex.pdf")
    assert match["composite_score"] > 60.0
    assert match["skill_score"] == 100.0
    assert len(match["missing_required"]) == 0
