"""
Text Preprocessor Module for Resume and Job Description Processing.
Includes noise removal, tokenization, lemmatization, and section extraction.
"""

import re
import string
from typing import Dict, List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure essential NLTK resources are available
for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


class TextPreprocessor:
    """
    Handles robust cleaning, normalization, tokenization, and sectioning
    of unstructured resume and job description text.
    """

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()

        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

        # Technical terms that shouldn't be stripped as stopwords
        self.stop_words.difference_update({"r", "c", "ai", "ml", "nlp", "sql", "aws", "gcp", "git"})

        # Regex patterns for contact data and noise
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.phone_pattern = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}")
        self.bullet_pattern = re.compile(r"[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25AB\u25CF\u25CB\u25BA\u25AA\-\*]")

        # Common resume section header keywords
        self.section_headers = {
            "skills": re.compile(r"\b(skills|technical skills|technologies|proficiencies|competencies|tech stack)\b", re.IGNORECASE),
            "experience": re.compile(r"\b(experience|work experience|employment history|work history|professional experience)\b", re.IGNORECASE),
            "education": re.compile(r"\b(education|academic background|qualifications|academic history|degrees)\b", re.IGNORECASE),
            "projects": re.compile(r"\b(projects|key projects|academic projects|personal projects)\b", re.IGNORECASE),
            "certifications": re.compile(r"\b(certifications|certificates|licenses|credentials|accreditations)\b", re.IGNORECASE),
            "summary": re.compile(r"\b(summary|professional summary|executive summary|about me|profile|objective)\b", re.IGNORECASE)
        }

    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Cleans raw unstructured text:
        - Removes URLs, emails, phone numbers, bullets, and excessive spaces.
        - Preserves important compound tokens like C++, C#, .NET, Node.js.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(" ", text)
        text = self.email_pattern.sub(" ", text)
        text = self.phone_pattern.sub(" ", text)

        # Replace bullet points with whitespace
        text = self.bullet_pattern.sub(" ", text)

        # Protect special language names before removing general punctuation
        replacements = {
            r"\b[cC]\+\+": "__CPP__",
            r"\b[cC]#": "__CSHARP__",
            r"\.net\b": "__DOTNET__",
            r"\bnode\.js\b": "__NODEJS__",
            r"\breact\.js\b": "__REACTJS__",
            r"\bvue\.js\b": "__VUEJS__",
            r"\bci/cd\b": "__CICD__",
            r"\ba/b testing\b": "__ABTESTING__",
            r"\bpl/sql\b": "__PLSQL__"
        }

        for pattern, token in replacements.items():
            text = re.sub(pattern, token, text, flags=re.IGNORECASE)

        # Remove other special symbols except alphanumeric, whitespace, and periods/hyphens between words
        text = re.sub(r"[^\w\s\.\-]", " ", text)

        # Restore protected tokens
        restore_map = {
            "__CPP__": "C++",
            "__CSHARP__": "C#",
            "__DOTNET__": ".NET",
            "__NODEJS__": "Node.js",
            "__REACTJS__": "React.js",
            "__VUEJS__": "Vue.js",
            "__CICD__": "CI/CD",
            "__ABTESTING__": "A/B Testing",
            "__PLSQL__": "PL/SQL"
        }
        for token, original in restore_map.items():
            text = text.replace(token, original)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if not preserve_case:
            return text.lower()
        return text

    def tokenize_and_lemmatize(self, text: str, remove_stop: bool = True) -> List[str]:
        """
        Tokenizes text, strips stopwords, and lemmatizes words.
        """
        cleaned = self.clean_text(text, preserve_case=False)
        if not cleaned:
            return []

        try:
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = cleaned.split()

        processed_tokens = []
        for token in tokens:
            # Strip punctuation
            token = token.strip(string.punctuation)
            if not token or len(token) < 2 and token not in {"r", "c"}:
                continue

            if remove_stop and token in self.stop_words:
                continue

            if self.lemmatizer:
                try:
                    lemma = self.lemmatizer.lemmatize(token)
                except Exception:
                    lemma = token
            else:
                lemma = token

            processed_tokens.append(lemma)

        return processed_tokens

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Segments resume text into standard sections based on heading heuristics.
        Returns a dict of section_name -> section_content.
        """
        if not text:
            return {}

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sections: Dict[str, List[str]] = {
            "summary": [],
            "skills": [],
            "experience": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "other": []
        }

        current_section = "other"

        for line in lines:
            line_matched = False
            # Check if this line is a section header (short line with header match)
            if len(line.split()) <= 4:
                for section_name, pattern in self.section_headers.items():
                    if pattern.search(line):
                        current_section = section_name
                        line_matched = True
                        break

            if not line_matched:
                sections[current_section].append(line)

        return {k: " ".join(v) for k, v in sections.items() if v}


# Module level singleton for convenience
preprocessor = TextPreprocessor()
