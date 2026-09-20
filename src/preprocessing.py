"""Text preprocessing module for Support Ticket Classification & Prioritization.

Provides:
1. `clean_text(text: str) -> str`: Normalizes raw ticket text (lowercasing, URL/email stripping,
   punctuation normalization, whitespace cleaning) while preserving domain-critical tokens (e.g. 500, 403).
2. `TextCleanerTransformer`: A scikit-learn BaseEstimator / TransformerMixin for seamless Pipeline integration.
"""

import re
import string
from typing import List, Union, Iterable
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Precompiled regex patterns for efficiency
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
WHITESPACE_PATTERN = re.compile(r"\s+")
# Retain alphanumeric characters, hyphens, and essential error syntax, stripping decorative punctuation
PUNCT_TRANSLATOR = str.maketrans("", "", "".join([c for c in string.punctuation if c not in ["-", "_"]]))

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "pls": "please",
    "plz": "please",
    "asap": "urgent as soon as possible"
}
CONTRACTION_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS.keys()) + r")\b", re.IGNORECASE)


def expand_contractions(text: str) -> str:
    """Expands common English contractions and informal support abbreviations."""
    def replace(match):
        return CONTRACTIONS[match.group(0).lower()]
    return CONTRACTION_PATTERN.sub(replace, text)


def clean_text(text: Union[str, float]) -> str:
    """Cleans and normalizes a single ticket text string.
    
    Operations:
    1. Handle non-string / null inputs gracefully.
    2. Lowercase text.
    3. Remove URLs and email addresses.
    4. Expand common contractions.
    5. Strip non-essential punctuation while retaining hyphenated terms and error codes.
    6. Normalize multiple whitespaces into a single space.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Strip URLs & Emails
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    
    # 3. Expand contractions
    text = expand_contractions(text)
    
    # 4. Remove unwanted punctuation
    text = text.translate(PUNCT_TRANSLATOR)
    
    # 5. Clean whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    
    return text


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn transformer to embed text cleaning directly inside a Pipeline."""
    
    def __init__(self):
        pass
        
    def fit(self, X: Iterable[str], y=None):
        """No parameters to fit (stateless transformer)."""
        return self
        
    def transform(self, X: Iterable[str]) -> List[str]:
        """Applies clean_text across an iterable/Series/array of ticket texts."""
        if isinstance(X, pd.Series):
            return X.apply(clean_text).tolist()
        return [clean_text(item) for item in X]


if __name__ == "__main__":
    sample_text = "URGENT!! App can't connect to https://portal.acme.com for user j.smith@acme.com with 500 error! Pls fix asap."
    cleaned = clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned :", cleaned)
