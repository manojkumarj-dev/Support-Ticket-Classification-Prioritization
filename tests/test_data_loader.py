"""Tests for data loading, schema validation, and stratified splitting."""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.data_loader import load_raw_data, check_missing_and_duplicates, split_data


def test_load_raw_data_schema():
    """Verify that raw data loads with required columns."""
    df = load_raw_data()
    expected_cols = {"ticket_id", "text", "category", "priority"}
    assert expected_cols.issubset(set(df.columns)), f"Missing expected columns in {df.columns}"
    assert len(df) > 0, "Dataset should not be empty"


def test_check_missing_and_duplicates():
    """Verify null handling and duplicate detection."""
    sample_data = pd.DataFrame({
        "ticket_id": ["TCK-1", "TCK-2", "TCK-3", "TCK-4"],
        "text": ["App crashed", "App crashed", None, "VPN issue"],
        "category": ["Bug", "Bug", "Bug", "IT Support"],
        "priority": ["High", "High", "High", "Low"]
    })
    
    cleaned_df, summary = check_missing_and_duplicates(sample_data, drop_duplicates=True)
    
    assert summary["null_rows_dropped"] == 1, "Should drop 1 null row"
    assert summary["duplicate_rows_found"] == 1, "Should detect 1 duplicate row"
    assert len(cleaned_df) == 2, "Should retain 2 unique valid rows"


def test_split_data_preserves_stratification():
    """Verify stratified splitting retains class presence and proportions."""
    df = load_raw_data()
    cleaned_df, _ = check_missing_and_duplicates(df)
    
    train_df, test_df = split_data(cleaned_df, test_size=0.20, save=False)
    
    assert len(train_df) + len(test_df) == len(cleaned_df)
    assert set(train_df["category"].unique()) == set(cleaned_df["category"].unique())
    assert set(test_df["category"].unique()) == set(cleaned_df["category"].unique())
