"""
Unit tests for CandidateRanker.
"""

import pytest
from src.ranker import CandidateRanker


@pytest.fixture
def ranker():
    return CandidateRanker()


def test_assign_tier(ranker):
    assert "High" in ranker.assign_tier(85.0)
    assert "Potential" in ranker.assign_tier(65.0)
    assert "Low" in ranker.assign_tier(40.0)


def test_rank_candidates(ranker):
    candidates = [
        {"candidate_name": "Candidate B", "composite_score": 70.0, "skill_score": 75.0, "semantic_score": 65.0},
        {"candidate_name": "Candidate A", "composite_score": 88.0, "skill_score": 90.0, "semantic_score": 85.0},
        {"candidate_name": "Candidate C", "composite_score": 45.0, "skill_score": 40.0, "semantic_score": 50.0}
    ]

    ranked = ranker.rank_candidates(candidates)
    assert len(ranked) == 3
    assert ranked[0]["candidate_name"] == "Candidate A"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["candidate_name"] == "Candidate B"
    assert ranked[1]["rank"] == 2
    assert ranked[2]["candidate_name"] == "Candidate C"
    assert ranked[2]["rank"] == 3
    assert ranked[0]["percentile"] == 100.0
    assert ranked[2]["percentile"] == 0.0
