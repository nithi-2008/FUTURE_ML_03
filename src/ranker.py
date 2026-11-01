"""
Candidate Ranking Engine.
Sorts evaluated resumes, assigns competitive ranking tiers, percentiles,
and generates human-readable recruiter verdicts.
"""

from typing import Dict, List


class CandidateRanker:
    """
    Ranks multiple candidates against a target job role, assigns tiers,
    and formats data for recruiter decision tables and leaderboards.
    """

    TIER_HIGH = "[HIGH] Highly Recommended"
    TIER_MEDIUM = "[MEDIUM] Potential Fit (Review)"
    TIER_LOW = "[LOW] Low Fit (Not Shortlisted)"

    def __init__(self,
                 high_threshold: float = 75.0,
                 medium_threshold: float = 55.0):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def assign_tier(self, score: float) -> str:
        """Assigns categorical recommendation tier based on composite score."""
        if score >= self.high_threshold:
            return self.TIER_HIGH
        elif score >= self.medium_threshold:
            return self.TIER_MEDIUM
        return self.TIER_LOW

    def generate_verdict(self, match_data: Dict) -> str:
        """
        Creates a clear, non-technical executive verdict for recruiters.
        """
        score = match_data.get("composite_score", 0.0)
        req_pct = match_data.get("required_match_pct", 0.0)
        missing_req = match_data.get("missing_required", [])
        cand_name = match_data.get("candidate_name", "Candidate")

        if score >= self.high_threshold and req_pct >= 80.0:
            return f"Top candidate: demonstrates strong mastery of core job requirements ({req_pct}% required skills match) and high semantic alignment."
        elif score >= self.high_threshold:
            return f"Strong overall candidate ({score}% score). Solid domain familiarity with minor skill gaps to verify during interview."
        elif score >= self.medium_threshold:
            if missing_req:
                missing_str = ", ".join(missing_req[:3])
                return f"Moderate fit ({score}%). Possesses transferable skills but missing critical requirements: {missing_str}."
            return f"Moderate fit ({score}%). Meets basic profile requirements but lacks depth in primary domain competencies."
        else:
            return f"Low fit for this specific role ({score}%). Major skill mismatch; candidate background is outside primary role demands."

    def rank_candidates(self, candidate_matches: List[Dict]) -> List[Dict]:
        """
        Sorts candidates by composite_score, skill_score, and semantic_score.
        Adds rank, tier, percentile, and verdict fields.
        """
        if not candidate_matches:
            return []

        # Sort descending
        sorted_candidates = sorted(
            candidate_matches,
            key=lambda x: (
                x.get("composite_score", 0.0),
                x.get("skill_score", 0.0),
                x.get("semantic_score", 0.0)
            ),
            reverse=True
        )

        total_candidates = len(sorted_candidates)

        ranked_list = []
        for idx, cand in enumerate(sorted_candidates):
            rank = idx + 1
            score = cand.get("composite_score", 0.0)

            # Calculate percentile rank
            if total_candidates > 1:
                percentile = round(((total_candidates - rank) / (total_candidates - 1)) * 100.0, 1)
            else:
                percentile = 100.0

            cand_copy = dict(cand)
            cand_copy["rank"] = rank
            cand_copy["tier"] = self.assign_tier(score)
            cand_copy["percentile"] = percentile
            cand_copy["verdict"] = self.generate_verdict(cand)

            ranked_list.append(cand_copy)

        return ranked_list


ranker = CandidateRanker()
