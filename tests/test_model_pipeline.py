"""Tests for model loading, end-to-end inference, and prediction interface."""

import pytest
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.predict import TicketClassifier


@pytest.fixture(scope="module")
def classifier():
    return TicketClassifier()


def test_model_loading(classifier):
    """Verify that both pipelines load successfully."""
    assert classifier.category_pipeline is not None
    assert classifier.priority_pipeline is not None


def test_predict_standard_ticket(classifier):
    """Verify prediction structure and validity on standard input."""
    text = "Application crashes with 500 Internal Server Error when generating the annual report"
    result = classifier.predict(text)
    
    assert "category" in result
    assert "priority" in result
    assert "category_confidence" in result
    assert "priority_confidence" in result
    
    assert result["category"] in config.CATEGORIES
    assert result["priority"] in config.PRIORITIES
    
    assert 0.0 <= result["category_confidence"] <= 1.0
    assert 0.0 <= result["priority_confidence"] <= 1.0


def test_predict_empty_input(classifier):
    """Verify edge-case handling for empty strings."""
    result = classifier.predict("   ")
    assert result["category"] == "Uncertain / Unclassified"
    assert result["category_confidence"] == 0.0
    assert result["priority"] == "Low"
