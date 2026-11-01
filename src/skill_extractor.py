"""
Skill and Entity Extractor using spaCy, Regex, and Canonical Skills Taxonomy.
Identifies programming languages, tools, frameworks, soft skills, education, and experience.
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
import spacy
from spacy.matcher import PhraseMatcher


class SkillExtractor:
    """
    NLP-based skill extraction engine utilizing spaCy phrase matching,
    regex heuristics, and a curated multi-domain taxonomy.
    """

    def __init__(self, taxonomy_path: Optional[str] = None):
        # Locate taxonomy file
        if not taxonomy_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            taxonomy_path = os.path.join(base_dir, "data", "skills_taxonomy.json")

        self.taxonomy_path = taxonomy_path
        self.skills_taxonomy = self._load_taxonomy()

        # Build alias -> canonical and canonical -> category mappings
        self.alias_to_canonical: Dict[str, str] = {}
        self.canonical_to_category: Dict[str, str] = {}
        self.canonical_skills: Set[str] = set()

        for item in self.skills_taxonomy:
            canonical = item["canonical"]
            category = item["category"]
            self.canonical_skills.add(canonical)
            self.canonical_to_category[canonical] = category

            # Map canonical itself (lowercase)
            self.alias_to_canonical[canonical.lower()] = canonical

            # Map aliases
            for alias in item.get("aliases", []):
                self.alias_to_canonical[alias.lower().strip()] = canonical

        # Load spaCy NLP model
        self.nlp = self._init_spacy()
        self.phrase_matcher = self._init_phrase_matcher()

        # Regex patterns for special symbols and contact details
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.phone_pattern = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}")
        self.exp_pattern = re.compile(r"\b(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)(?:\s*of)?\s*(?:experience|exp|work\s*experience)?\b", re.IGNORECASE)

        # Degree patterns
        self.degree_patterns = {
            "Doctorate (Ph.D.)": re.compile(r"\b(ph\.?d|doctorate|doctor of philosophy)\b", re.IGNORECASE),
            "Master's Degree": re.compile(r"\b(master'?s?|m\.?s|m\.?tech|m\.?sc|m\.?b\.?a|post[\s-]graduate)\b", re.IGNORECASE),
            "Bachelor's Degree": re.compile(r"\b(bachelor'?s?|b\.?s|b\.?tech|b\.?sc|b\.?e|undergraduate)\b", re.IGNORECASE),
            "Associate / Diploma": re.compile(r"\b(associate'?s?|diploma)\b", re.IGNORECASE)
        }

    def _load_taxonomy(self) -> List[Dict]:
        """Loads the skills taxonomy JSON file."""
        if os.path.exists(self.taxonomy_path):
            with open(self.taxonomy_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _init_spacy(self):
        """Initializes spaCy model with graceful fallback."""
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            try:
                return spacy.blank("en")
            except Exception:
                return None

    def _init_phrase_matcher(self) -> Optional[PhraseMatcher]:
        """Builds spaCy PhraseMatcher from all taxonomy aliases."""
        if not self.nlp:
            return None

        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        # Add patterns for each alias
        for alias, canonical in self.alias_to_canonical.items():
            # Skip very short 1-letter tokens for phrase matcher (handled via regex to avoid false positives)
            if len(alias) <= 1:
                continue
            doc = self.nlp.make_doc(alias)
            matcher.add(canonical, [doc])

        return matcher

    def extract_skills(self, text: str) -> Dict[str, any]:
        """
        Extracts skills from text using spaCy PhraseMatcher + regex boundaries.
        Returns:
            - found_skills: list of canonical skill names (unique)
            - skills_by_category: dict of category -> [skills]
            - skill_counts: dict of canonical_skill -> count of occurrences
        """
        if not text:
            return {
                "found_skills": [],
                "skills_by_category": {},
                "skill_counts": {}
            }

        found_skills: Set[str] = set()
        skill_counts: Dict[str, int] = {}

        # 1. spaCy PhraseMatcher extraction
        if self.nlp and self.phrase_matcher:
            try:
                # Process in chunks if text is exceptionally long
                doc = self.nlp(text[:50000])
                matches = self.phrase_matcher(doc)
                for match_id, start, end in matches:
                    canonical = self.nlp.vocab.strings[match_id]
                    found_skills.add(canonical)
                    skill_counts[canonical] = skill_counts.get(canonical, 0) + 1
            except Exception:
                pass

        # 2. Regex fallback and boundary check for special symbols like C, R, C++, C#, .NET
        special_checks = [
            (r"\b[cC]\+\+\b", "C++"),
            (r"\b[cC]#\b", "C#"),
            (r"(?:^|[\s,;])\.net(?:\b|[\s,;])", "ASP.NET"),
            (r"\b[rR]\s+(?:programming|language|script)\b", "R"),
            (r"\b[cC]\s+(?:programming|language)\b", "C++"),
            (r"\bci\s*/\s*cd\b", "CI/CD"),
            (r"\ba\s*/\s*b\s*testing\b", "A/B Testing")
        ]

        for pattern, canonical in special_checks:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                found_skills.add(canonical)
                skill_counts[canonical] = skill_counts.get(canonical, 0) + len(matches)

        # 3. Direct lowercase token check against taxonomy for robust coverage
        text_lower = " " + text.lower() + " "
        for alias, canonical in self.alias_to_canonical.items():
            if len(alias) <= 2:
                # Require explicit word boundary for short aliases
                escaped = re.escape(alias)
                if re.search(rf"\b{escaped}\b", text_lower):
                    found_skills.add(canonical)
                    skill_counts[canonical] = skill_counts.get(canonical, 0) + 1
            else:
                if f" {alias} " in text_lower or f" {alias}," in text_lower or f" {alias}." in text_lower:
                    found_skills.add(canonical)
                    skill_counts[canonical] = skill_counts.get(canonical, 0) + 1

        # Group by domain category
        skills_by_category: Dict[str, List[str]] = {}
        for skill in sorted(found_skills):
            cat = self.canonical_to_category.get(skill, "Other Skills")
            skills_by_category.setdefault(cat, []).append(skill)

        return {
            "found_skills": sorted(list(found_skills)),
            "skills_by_category": skills_by_category,
            "skill_counts": skill_counts,
            "total_skills_count": len(found_skills)
        }

    def extract_candidate_info(self, text: str, filename: str = "") -> Dict[str, any]:
        """
        Extracts candidate metadata: Name, Email, Phone, Experience Years, Education Degree.
        """
        email_match = self.email_pattern.search(text)
        email = email_match.group(0) if email_match else "Not Provided"

        phone_match = self.phone_pattern.search(text)
        phone = phone_match.group(0).strip() if phone_match else "Not Provided"

        # Experience years extraction
        exp_matches = self.exp_pattern.findall(text)
        exp_years = 0.0
        if exp_matches:
            try:
                exp_values = [float(v) for v in exp_matches if float(v) < 45]
                exp_years = max(exp_values) if exp_values else 0.0
            except Exception:
                exp_years = 0.0

        # Degree extraction
        highest_degree = "Not Specified"
        for degree_name, pattern in self.degree_patterns.items():
            if pattern.search(text):
                highest_degree = degree_name
                break

        # Candidate name heuristics
        name = self._extract_candidate_name(text, filename)

        return {
            "candidate_name": name,
            "email": email,
            "phone": phone,
            "experience_years": exp_years,
            "highest_degree": highest_degree
        }

    def _extract_candidate_name(self, text: str, filename: str = "") -> str:
        """
        Heuristic candidate name extractor from the top 5 lines of the resume,
        spaCy PERSON entities, or filename.
        """
        # If filename contains a recognizable name like resume_alex_chen.pdf
        if filename:
            clean_name = os.path.splitext(filename)[0]
            clean_name = re.sub(r"(resume|cv|profile|updated|202[0-9]|draft)[_\-\s]*", "", clean_name, flags=re.IGNORECASE)
            clean_name = re.sub(r"[_\-]+", " ", clean_name).strip().title()
            if len(clean_name.split()) >= 2 and len(clean_name) > 4:
                return clean_name

        # spaCy PERSON entity on the top 300 characters
        if self.nlp:
            top_text = text[:400]
            try:
                doc = self.nlp(top_text)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        cand_name = ent.text.strip()
                        # Avoid names that are just common words or email fragments
                        if len(cand_name.split()) in [2, 3] and "@" not in cand_name:
                            return cand_name.title()
            except Exception:
                pass

        # First non-empty line of resume often contains candidate name
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:3]:
            # Check if line looks like a person name (2-3 words, all letters)
            words = line.split()
            if 2 <= len(words) <= 3 and all(w.isalpha() for w in words):
                if words[0].lower() not in ["curriculum", "resume", "cv", "profile", "summary"]:
                    return line.title()

        return "Candidate Profile"


extractor = SkillExtractor()
