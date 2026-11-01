"""
Unit tests for SkillExtractor.
"""

import pytest
from src.skill_extractor import SkillExtractor


@pytest.fixture
def extractor():
    return SkillExtractor()


def test_extract_canonical_skills(extractor):
    text = "Proficient in python, sklearn, k8s, postgres, and aws cloud services."
    result = extractor.extract_skills(text)
    found = result["found_skills"]

    assert "Python" in found
    assert "Scikit-Learn" in found
    assert "Kubernetes" in found
    assert "PostgreSQL" in found
    assert "AWS" in found


def test_special_symbol_skills(extractor):
    text = "Extensive experience in C++, C#, and .NET application architecture."
    result = extractor.extract_skills(text)
    found = result["found_skills"]

    assert "C++" in found
    assert "C#" in found
    assert "ASP.NET" in found


def test_candidate_info_extraction(extractor):
    text = """
    Sarah Jenkins
    Contact: sarah.jenkins@testcorp.com | Phone: (555) 123-4567
    Experience: 5 years of experience in full stack software engineering.
    Education: Master's Degree in Computer Science from Stanford University.
    """
    info = extractor.extract_candidate_info(text, filename="resume_sarah_jenkins.pdf")

    assert "Sarah Jenkins" in info["candidate_name"]
    assert info["email"] == "sarah.jenkins@testcorp.com"
    assert info["experience_years"] == 5.0
    assert "Master" in info["highest_degree"]


def test_categorized_skills(extractor):
    text = "Built Docker containers on AWS, deployed with Kubernetes, and managed PostgreSQL databases."
    result = extractor.extract_skills(text)
    by_cat = result["skills_by_category"]

    assert "Cloud & DevOps" in by_cat
    assert "AWS" in by_cat["Cloud & DevOps"]
    assert "Docker" in by_cat["Cloud & DevOps"]
    assert "Databases & Data Engineering" in by_cat
