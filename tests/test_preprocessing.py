"""Tests for text cleaning and TextCleanerTransformer."""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing import clean_text, TextCleanerTransformer


def test_clean_text_lowercasing_and_punctuation():
    raw = "ERROR: System Crashed with 500! Please Fix."
    cleaned = clean_text(raw)
    assert "error" in cleaned
    assert "system crashed with 500" in cleaned
    assert "!" not in cleaned


def test_clean_text_strips_urls_and_emails():
    raw = "Contact admin@corp.com or check https://status.corp.com for updates"
    cleaned = clean_text(raw)
    assert "admin@corp.com" not in cleaned
    assert "https" not in cleaned
    assert "statuscorpcom" not in cleaned


def test_clean_text_expands_contractions():
    raw = "I can't access, pls help asap!"
    cleaned = clean_text(raw)
    assert "cannot" in cleaned
    assert "please" in cleaned
    assert "urgent as soon as possible" in cleaned


def test_clean_text_handles_empty_and_null():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_text_cleaner_transformer():
    transformer = TextCleanerTransformer()
    inputs = ["Bug in app!", "VPN down at https://vpn.corp.com"]
    transformed = transformer.transform(inputs)
    assert isinstance(transformed, list)
    assert len(transformed) == 2
    assert "!" not in transformed[0]
    assert "https" not in transformed[1]
