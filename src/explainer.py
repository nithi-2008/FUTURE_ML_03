"""
Explainable AI (XAI) & HR Decision-Support Explainer.
Generates plain-English explanations of scoring, rank comparisons,
skill gap breakdowns, and tailored interview questions.
"""

from typing import Dict, List, Optional


class ScreeningExplainer:
    """
    Translates NLP similarity metrics and skill matrices into actionable,
    bias-free decision support for recruiters and hiring managers.
    """

    @staticmethod
    def explain_score_breakdown(match_data: Dict) -> Dict[str, any]:
        """
        Decomposes the composite role fit score into transparent components.
        """
        weights = match_data.get("weights_used", {"skill": 0.5, "semantic": 0.35, "experience": 0.15})
        skill_score = match_data.get("skill_score", 0.0)
        semantic_score = match_data.get("semantic_score", 0.0)
        exp_score = match_data.get("experience_score", 0.0)
        composite = match_data.get("composite_score", 0.0)

        skill_points = round(weights["skill"] * skill_score, 2)
        semantic_points = round(weights["semantic"] * semantic_score, 2)
        exp_points = round(weights["experience"] * exp_score, 2)

        explanation = (
            f"Candidate achieved a composite score of {composite}%. "
            f"This consists of: {skill_points} pts from direct skill match ({skill_score}% matched @ {int(weights['skill']*100)}% weight), "
            f"{semantic_points} pts from semantic context similarity ({semantic_score}% matched @ {int(weights['semantic']*100)}% weight), "
            f"and {exp_points} pts from experience alignment ({exp_score}% matched @ {int(weights['experience']*100)}% weight)."
        )

        return {
            "composite_score": composite,
            "skill_points": skill_points,
            "semantic_points": semantic_points,
            "exp_points": exp_points,
            "text_explanation": explanation
        }

    @staticmethod
    def explain_ranking_advantage(candidate: Dict, benchmark_candidate: Optional[Dict] = None) -> str:
        """
        Explains why a candidate ranks higher or lower relative to another candidate or overall benchmark.
        """
        name = candidate.get("candidate_name", "Candidate")
        score = candidate.get("composite_score", 0.0)
        req_pct = candidate.get("required_match_pct", 0.0)
        matched_req = candidate.get("matched_required", [])
        missing_req = candidate.get("missing_required", [])

        if not benchmark_candidate:
            if score >= 75:
                return (
                    f"**{name}** holds a leading rank due to exceptional alignment on core requirements "
                    f"({req_pct}% match on required skills). They verified key competencies in: "
                    f"`{', '.join(matched_req[:5])}` with consistent domain terminology across their resume."
                )
            else:
                return (
                    f"**{name}** is ranked at #{candidate.get('rank', 'N/A')} primarily because of missing requirements: "
                    f"`{', '.join(missing_req[:4]) if missing_req else 'lower semantic similarity'}`."
                )

        bench_name = benchmark_candidate.get("candidate_name", "Top Candidate")
        bench_score = benchmark_candidate.get("composite_score", 0.0)
        delta = round(score - bench_score, 1)

        if delta >= 0:
            return (
                f"**{name}** ranks ahead of **{bench_name}** by +{delta}% overall fit, having "
                f"demonstrated stronger direct required skill coverage ({len(matched_req)} vs {len(benchmark_candidate.get('matched_required', []))})."
            )
        else:
            reasons = []
            if len(missing_req) > len(benchmark_candidate.get("missing_required", [])):
                reasons.append(f"missing {len(missing_req)} required skills vs {len(benchmark_candidate.get('missing_required', []))}")
            if candidate.get("semantic_score", 0) < benchmark_candidate.get("semantic_score", 0):
                reasons.append("lower semantic keyword density in project descriptions")
            if candidate.get("experience_years", 0) < benchmark_candidate.get("experience_years", 0):
                reasons.append(f"{candidate.get('experience_years')} yrs exp vs {benchmark_candidate.get('experience_years')} yrs")

            reason_str = ", and ".join(reasons) if reasons else "lower overall similarity score"
            return (
                f"**{name}** trails **{bench_name}** ({delta}%) due to: {reason_str}."
            )

    @staticmethod
    def generate_interview_questions(candidate: Dict) -> List[Dict[str, str]]:
        """
        Generates targeted interview probe questions tailored to this candidate's specific skill gaps.
        """
        questions = []
        missing_req = candidate.get("missing_required", [])
        matched_req = candidate.get("matched_required", [])

        # Probing missing skills
        for skill in missing_req[:4]:
            questions.append({
                "type": "Skill Gap Verification",
                "skill": skill,
                "question": f"Our role requires production proficiency in **{skill}**, which was not highlighted on your resume. Could you describe any hands-on or academic experience you have had with {skill} or comparable technologies?",
                "recruiter_tip": f"Look for conceptual understanding or willingness to quickly upskill in {skill}."
            })

        # Deep-diving claimed top skills
        for skill in matched_req[:2]:
            questions.append({
                "type": "Strength Validation",
                "skill": skill,
                "question": f"You mentioned extensive experience with **{skill}**. Can you walk us through a challenging production project where {skill} was critical to your architectural decision or performance bottleneck solution?",
                "recruiter_tip": f"Verify whether experience was hands-on architectural or superficial."
            })

        return questions

    @staticmethod
    def generate_recruiter_report(candidate: Dict) -> str:
        """
        Formats a clean, executive summary ready for HR candidate packets.
        """
        name = candidate.get("candidate_name", "Candidate")
        tier = candidate.get("tier", "Review")
        score = candidate.get("composite_score", 0.0)
        exp = candidate.get("experience_years", 0.0)
        degree = candidate.get("highest_degree", "Not Specified")
        matched_req = candidate.get("matched_required", [])
        missing_req = candidate.get("missing_required", [])
        matched_pref = candidate.get("matched_preferred", [])

        report = f"""### Candidate Screening Brief: {name}
* **Evaluation Tier**: {tier}
* **Composite Fit Score**: {score}% (Rank #{candidate.get('rank', 1)})
* **Experience & Education**: {exp} Years Experience | {degree}
* **Required Skills Match**: {len(matched_req)} matched ({', '.join(matched_req) if matched_req else 'None'})
* **Critical Skill Gaps**: {', '.join(missing_req) if missing_req else 'None! Complete core coverage'}
* **Bonus / Preferred Skills**: {', '.join(matched_pref) if matched_pref else 'None'}
* **Hiring Recommendation**: {candidate.get('verdict', 'Review profile with team.')}
"""
        return report.strip()


explainer = ScreeningExplainer()
