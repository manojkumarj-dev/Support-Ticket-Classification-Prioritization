"""Tests for feature extraction and TF-IDF vectorizer configuration."""

import pytest
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.feature_engineering import build_tfidf_vectorizer


def test_tfidf_vectorizer_shapes_and_ngrams():
    corpus = [
        "application crashed with 500 internal server error",
        "cannot log in okta sso token expired",
        "credit card payment failed invoice overdue"
    ]
    vec = build_tfidf_vectorizer(ngram_range=(1, 2), min_df=1, max_features=100)
    X = vec.fit_transform(corpus)
    
    feature_names = vec.get_feature_names_out()
    assert X.shape[0] == 3
    assert X.shape[1] <= 100
    
    # Check that bigrams are captured
    assert any(" " in feat for feat in feature_names), "Should extract bigrams"
    assert "500" in feature_names, "Should preserve error code 500"


def test_tfidf_sublinear_tf():
    corpus = [
        "error error error error error error crash",
        "error crash"
    ]
    # Pass max_df=1.0 for tiny 2-document test corpus so terms appearing in both are not pruned
    vec = build_tfidf_vectorizer(min_df=1, max_df=1.0, sublinear_tf=True)
    X = vec.fit_transform(corpus).toarray()
    
    error_idx = vec.vocabulary_["error"]
    # With sublinear tf, 6 occurrences is 1 + log(6) ~ 2.79, not 6 times larger
    ratio = X[0, error_idx] / X[1, error_idx]
    assert ratio < 3.0, "Sublinear scaling should dampen repeated term weight"
