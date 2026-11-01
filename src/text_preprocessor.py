"""
Text Preprocessing and Feature Extraction Module
Support Ticket Classification & Prioritization
"""

import re
import string
import unicodedata
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# NLTK imports with graceful fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Ensure required resources exist
    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))

    try:
        LEMMATIZER = WordNetLemmatizer()
        _ = LEMMATIZER.lemmatize("tickets")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        LEMMATIZER = WordNetLemmatizer()

except Exception:
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
        "more", "most", "my", "myself", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
        "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
        "would", "you", "your", "yours", "yourself", "yourselves"
    }
    LEMMATIZER = None

# Critical negation and urgency words that must NOT be removed as stopwords
PRESERVED_WORDS = {
    "not", "no", "never", "cannot", "cant", "wont", "without", "against", "down",
    "fail", "failed", "crash", "bug", "error", "urgent", "broken", "issue", "problem"
}
FILTERED_STOPWORDS = STOPWORDS - PRESERVED_WORDS

# Contraction dictionary for normalizing English colloquialisms
CONTRACTIONS_DICT = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that's": "that is",
    "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is",
    "who'll": "who will", "who's": "who is", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

# Domain urgency keywords indicating high or critical priority
URGENCY_KEYWORDS = {
    "urgent", "urgently", "asap", "emergency", "critical", "outage", "down",
    "crashed", "crashing", "fatal", "broken", "breach", "hacked", "loss",
    "lost", "refund", "lawsuit", "immediate", "immediately", "deadline",
    "unusable", "stuck", "cannot login", "locked out", "failed payment",
    "overcharged", "illegal", "security", "compromised", "exploit", "unauthorized"
}


def expand_contractions(text: str) -> str:
    """Expand standard English contractions into full forms."""
    if not isinstance(text, str):
        return ""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS_DICT.keys()) + r')\b', re.IGNORECASE)

    def replace(match):
        match_str = match.group(0).lower()
        expanded = CONTRACTIONS_DICT.get(match_str, match_str)
        # Match case
        if match.group(0).isupper():
            return expanded.upper()
        if match.group(0)[0].isupper():
            return expanded.capitalize()
        return expanded

    return pattern.sub(replace, text)


def clean_text(
    text: str,
    lower: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_special_chars: bool = True,
    remove_stops: bool = True,
    lemmatize: bool = True
) -> str:
    """
    Clean and normalize raw customer support ticket text.
    
    Steps:
    1. Unicode normalization and contraction expansion
    2. Removal of URLs, emails, IP addresses, HTML tags
    3. Punctuation and digit normalization
    4. Stopword removal (preserving negation and critical problem tokens)
    5. WordNet Lemmatization
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Normalize unicode (accents, curly quotes)
    text = unicodedata.normalize("NFKD", text)

    # Expand contractions
    text = expand_contractions(text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Strip URLs and web links
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Strip emails
    if remove_emails:
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

    # Strip IP addresses
    text = re.sub(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", " ", text)

    # Lowercase if requested
    if lower:
        text = text.lower()

    # Special characters & extra punctuation
    if remove_special_chars:
        # Keep alphanumeric, hyphen, and apostrophe for words
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Tokenize by whitespace
    tokens = text.split()

    # Stopword removal
    if remove_stops:
        tokens = [t for t in tokens if t not in FILTERED_STOPWORDS and len(t) > 1]

    # Lemmatization
    if lemmatize and LEMMATIZER is not None:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def extract_ticket_signals(text: str) -> Dict[str, Any]:
    """
    Extract linguistic and behavioral signals useful for priority triage:
    - Character and word count
    - Exclamation mark count
    - Question mark count
    - Uppercase ratio (shouting / panic metric)
    - Presence of high-urgency keywords
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "avg_word_len": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "uppercase_ratio": 0.0,
            "urgency_score": 0,
            "has_critical_keyword": False,
        }

    raw = text.strip()
    words = raw.split()
    total_chars = len(raw)
    total_words = len(words)

    # Uppercase chars (excluding spaces)
    non_space_chars = [c for c in raw if not c.isspace()]
    upper_count = sum(1 for c in non_space_chars if c.isupper())
    upper_ratio = (upper_count / len(non_space_chars)) if non_space_chars else 0.0

    exclamation_count = raw.count("!")
    question_count = raw.count("?")

    lower_text = raw.lower()
    matched_urgency = [w for w in URGENCY_KEYWORDS if re.search(r'\b' + re.escape(w) + r'\b', lower_text)]
    urgency_score = len(matched_urgency)

    return {
        "char_count": total_chars,
        "word_count": total_words,
        "avg_word_len": round(total_chars / max(1, total_words), 2),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "uppercase_ratio": round(upper_ratio, 3),
        "urgency_score": urgency_score,
        "has_critical_keyword": urgency_score > 0,
        "urgency_keywords_found": matched_urgency
    }


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies clean_text over an iterable.
    """
    def __init__(
        self,
        lower: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = True,
        remove_stops: bool = True,
        lemmatize: bool = True
    ):
        self.lower = lower
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.remove_stops = remove_stops
        self.lemmatize = lemmatize

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> List[str]:
        if isinstance(X, pd.Series):
            return [clean_text(str(val), self.lower, self.remove_urls, self.remove_emails,
                               self.remove_special_chars, self.remove_stops, self.lemmatize)
                    for val in X]
        elif isinstance(X, (list, np.ndarray)):
            return [clean_text(str(val), self.lower, self.remove_urls, self.remove_emails,
                               self.remove_special_chars, self.remove_stops, self.lemmatize)
                    for val in X]
        else:
            return [clean_text(str(X), self.lower, self.remove_urls, self.remove_emails,
                               self.remove_special_chars, self.remove_stops, self.lemmatize)]
