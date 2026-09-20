"""Feature engineering module for Support Ticket Classification & Prioritization.

Configures and builds TF-IDF vectorization pipelines with:
- n-gram extraction (unigrams + bigrams)
- Document frequency thresholds (min_df, max_df)
- Sublinear term frequency scaling
- Vocabulary size constraints (max_features)
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


def build_tfidf_vectorizer(
    ngram_range: tuple = config.TFIDF_PARAMS["ngram_range"],
    min_df: int = config.TFIDF_PARAMS["min_df"],
    max_df: float = config.TFIDF_PARAMS["max_df"],
    max_features: int = config.TFIDF_PARAMS["max_features"],
    sublinear_tf: bool = config.TFIDF_PARAMS["sublinear_tf"]
) -> TfidfVectorizer:
    """Builds a scikit-learn TfidfVectorizer configured for support ticket text.
    
    Args:
        ngram_range: (min_n, max_n) tuple. (1, 2) extracts words and adjacent 2-word phrases.
        min_df: Minimum document frequency. Terms appearing in fewer documents are ignored.
        max_df: Maximum document frequency proportion (0.0 to 1.0). Terms appearing in more
                than this proportion of documents are pruned as corpus-level stopwords.
        max_features: Maximum number of features ordered by term frequency across the corpus.
        sublinear_tf: If True, applies sublinear scaling (1 + log(tf)) to reduce the impact
                      of repeated tokens in lengthy tickets.
                      
    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z0-9_-]+\b"  # Retains alphanumeric codes (e.g. 500, 403, mapi32)
    )


def extract_top_features_per_class(
    vectorizer: TfidfVectorizer,
    X_tfidf,
    y: pd.Series,
    top_n: int = 10
) -> Dict[str, List[str]]:
    """Extracts the top N highest average TF-IDF features for each class.
    
    Useful for model interpretability and verifying feature relevance during EDA.
    """
    import numpy as np
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = {}
    
    for class_label in y.unique():
        class_mask = (y == class_label).values
        class_mean_tfidf = np.asarray(X_tfidf[class_mask].mean(axis=0)).flatten()
        top_indices = class_mean_tfidf.argsort()[::-1][:top_n]
        top_features[str(class_label)] = feature_names[top_indices].tolist()
        
    return top_features


if __name__ == "__main__":
    sample_corpus = [
        "Application crashes with 500 internal server error on checkout page",
        "Credit card declined for invoice payment refund requested",
        "Cannot log in with Okta SSO SAML assertion expired",
        "Feature request please add dark mode to the web dashboard",
        "Corporate VPN connection failed certificate expired"
    ]
    vec = build_tfidf_vectorizer()
    X = vec.fit_transform(sample_corpus)
    print(f"Sample corpus vectorized: {X.shape[0]} documents, {X.shape[1]} features extracted.")
    print(f"Sample features: {vec.get_feature_names_out()[:15]}")
