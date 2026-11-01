"""
Resume-to-Job Matching and Similarity Engine.
Combines TF-IDF semantic vector similarity with weighted skill matching,
experience calibration, and skill gap identification.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_preprocessor import preprocessor
from src.skill_extractor import extractor


class ResumeMatcher:
    """
    Evaluates candidate resumes against job descriptions using a hybrid
    NLP scoring model (TF-IDF Cosine Similarity + Weighted Skill Matching + Experience Alignment).
    """

    def __init__(self,
                 skill_weight: float = 0.50,
                 semantic_weight: float = 0.35,
                 experience_weight: float = 0.15):
        """
        Initializes scoring weights (sum should equal 1.0).
        """
        self.skill_weight = skill_weight
        self.semantic_weight = semantic_weight
        self.experience_weight = experience_weight

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=6000,
            sublinear_tf=True
        )

    def compute_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """
        Calculates TF-IDF cosine similarity between resume text and job description.
        Returns a float between 0.0 and 1.0.
        """
        cleaned_resume = preprocessor.clean_text(resume_text)
        cleaned_jd = preprocessor.clean_text(jd_text)

        if not cleaned_resume or not cleaned_jd:
            return 0.0

        try:
            tfidf_matrix = self.vectorizer.fit_transform([cleaned_jd, cleaned_resume])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            # Clip between 0 and 1
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.0

    def evaluate_skills(self,
                        candidate_skills: List[str],
                        required_skills: List[str],
                        preferred_skills: Optional[List[str]] = None,
                        required_multiplier: float = 2.0,
                        preferred_multiplier: float = 1.0) -> Dict[str, any]:
        """
        Calculates weighted skill match and identifies skill gaps.
        """
        preferred_skills = preferred_skills or []

        candidate_set: Set[str] = set(candidate_skills)
        required_set: Set[str] = set(required_skills)
        preferred_set: Set[str] = set(preferred_skills)

        # Intersections and differences
        matched_required = sorted(list(candidate_set.intersection(required_set)))
        missing_required = sorted(list(required_set.difference(candidate_set)))

        matched_preferred = sorted(list(candidate_set.intersection(preferred_set)))
        missing_preferred = sorted(list(preferred_set.difference(candidate_set)))

        extra_skills = sorted(list(candidate_set.difference(required_set).difference(preferred_set)))

        # Weighted calculation
        total_possible_weight = (len(required_set) * required_multiplier) + (len(preferred_set) * preferred_multiplier)
        earned_weight = (len(matched_required) * required_multiplier) + (len(matched_preferred) * preferred_multiplier)

        if total_possible_weight > 0:
            skill_score = (earned_weight / total_possible_weight) * 100.0
        else:
            skill_score = 100.0 if candidate_skills else 0.0

        required_match_pct = (len(matched_required) / len(required_set) * 100.0) if required_set else 100.0
        preferred_match_pct = (len(matched_preferred) / len(preferred_set) * 100.0) if preferred_set else 100.0

        return {
            "skill_score": round(float(skill_score), 2),
            "matched_required": matched_required,
            "missing_required": missing_required,
            "matched_preferred": matched_preferred,
            "missing_preferred": missing_preferred,
            "extra_skills": extra_skills,
            "required_match_pct": round(float(required_match_pct), 1),
            "preferred_match_pct": round(float(preferred_match_pct), 1),
            "total_candidate_skills": len(candidate_set),
            "total_required_skills": len(required_set),
            "total_preferred_skills": len(preferred_set)
        }

    def evaluate_experience(self,
                            candidate_exp: float,
                            required_exp: float) -> float:
        """
        Scores candidate experience alignment against the JD requirement (0 to 100).
        """
        if required_exp <= 0:
            return 100.0

        if candidate_exp >= required_exp:
            # Full score, plus small bonus for up to 3 years above, capped at 100
            return 100.0
        else:
            # Proportional score
            ratio = candidate_exp / required_exp
            return float(np.clip(ratio * 100.0, 10.0, 95.0))

    def match_resume(self,
                     resume_text: str,
                     job_description: Dict,
                     filename: str = "",
                     custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, any]:
        """
        Runs comprehensive evaluation of a single resume against a job description.
        """
        weights = custom_weights or {
            "skill": self.skill_weight,
            "semantic": self.semantic_weight,
            "experience": self.experience_weight
        }

        # Normalize weights
        total_w = sum(weights.values()) or 1.0
        w_skill = weights.get("skill", 0.5) / total_w
        w_semantic = weights.get("semantic", 0.35) / total_w
        w_exp = weights.get("experience", 0.15) / total_w

        # Extract candidate metadata and skills
        candidate_info = extractor.extract_candidate_info(resume_text, filename=filename)
        skills_data = extractor.extract_skills(resume_text)
        candidate_skills = skills_data["found_skills"]

        # Parse JD attributes
        jd_text = job_description.get("description", "")
        required_skills = job_description.get("required_skills", [])
        preferred_skills = job_description.get("preferred_skills", [])
        required_exp = float(job_description.get("experience_years", 2.0))

        # If JD has no explicit skills defined, extract them dynamically
        if not required_skills and jd_text:
            jd_extracted = extractor.extract_skills(jd_text)["found_skills"]
            required_skills = jd_extracted[:8]
            preferred_skills = jd_extracted[8:14]

        # 1. Semantic Similarity
        semantic_sim = self.compute_semantic_similarity(resume_text, jd_text)
        semantic_score = round(semantic_sim * 100.0, 2)

        # 2. Skill Evaluation & Gap Analysis
        skill_eval = self.evaluate_skills(candidate_skills, required_skills, preferred_skills)
        skill_score = skill_eval["skill_score"]

        # 3. Experience Alignment
        exp_score = self.evaluate_experience(candidate_info["experience_years"], required_exp)

        # 4. Composite Role Fit Score
        composite_score = (w_skill * skill_score) + (w_semantic * semantic_score) + (w_exp * exp_score)
        composite_score = round(float(np.clip(composite_score, 0.0, 100.0)), 2)

        return {
            "candidate_name": candidate_info["candidate_name"],
            "email": candidate_info["email"],
            "phone": candidate_info["phone"],
            "experience_years": candidate_info["experience_years"],
            "highest_degree": candidate_info["highest_degree"],
            "filename": filename or "resume",
            "composite_score": composite_score,
            "skill_score": skill_score,
            "semantic_score": semantic_score,
            "experience_score": round(exp_score, 2),
            "matched_required": skill_eval["matched_required"],
            "missing_required": skill_eval["missing_required"],
            "matched_preferred": skill_eval["matched_preferred"],
            "missing_preferred": skill_eval["missing_preferred"],
            "extra_skills": skill_eval["extra_skills"],
            "required_match_pct": skill_eval["required_match_pct"],
            "preferred_match_pct": skill_eval["preferred_match_pct"],
            "skills_by_category": skills_data["skills_by_category"],
            "total_skills_count": skills_data["total_skills_count"],
            "weights_used": {"skill": w_skill, "semantic": w_semantic, "experience": w_exp}
        }


matcher = ResumeMatcher()
