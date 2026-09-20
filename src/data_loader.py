"""Data loading, validation, and stratified splitting module.

Handles:
1. Loading raw CSV tickets.
2. Checking and reporting missing values and duplicate rows.
3. Class distribution computation.
4. Stratified train/test splitting to prevent data leakage.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
# Allow running as standalone script or imported module
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


def load_raw_data(filepath: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the raw support ticket dataset from disk with basic schema validation."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found at {filepath}. Please generate or provide it first.")
    
    df = pd.read_csv(filepath)
    expected_cols = {"ticket_id", "text", "category", "priority"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    
    return df


def check_missing_and_duplicates(df: pd.DataFrame, drop_duplicates: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Checks for null values and duplicate tickets.
    
    Args:
        df: Input DataFrame.
        drop_duplicates: If True, drops duplicate tickets based on the 'text' column.
        
    Returns:
        Cleaned DataFrame and a summary dictionary of findings.
    """
    initial_count = len(df)
    missing_counts = df.isnull().sum().to_dict()
    
    # Drop rows with null text or target labels
    df_clean = df.dropna(subset=["text", "category", "priority"]).copy()
    null_rows_dropped = initial_count - len(df_clean)
    
    # Check duplicates
    duplicate_count = df_clean.duplicated(subset=["text"]).sum()
    if drop_duplicates and duplicate_count > 0:
        df_clean = df_clean.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    summary = {
        "initial_rows": initial_count,
        "null_counts": missing_counts,
        "null_rows_dropped": null_rows_dropped,
        "duplicate_rows_found": duplicate_count,
        "final_rows": len(df_clean)
    }
    return df_clean, summary


def get_class_distributions(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Computes absolute and percentage class distributions for category and priority."""
    distributions = {
        "category_counts": df["category"].value_counts(),
        "category_percentages": df["category"].value_counts(normalize=True) * 100,
        "priority_counts": df["priority"].value_counts(),
        "priority_percentages": df["priority"].value_counts(normalize=True) * 100,
    }
    return distributions


def split_data(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    stratify_col: str = config.STRATIFY_COL,
    save: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs stratified train/test split to preserve target distributions.
    
    Args:
        df: Cleaned DataFrame.
        test_size: Proportion of test split (default 0.20).
        random_state: Seed for reproducibility.
        stratify_col: Column used for stratification.
        save: If True, saves train.csv and test.csv to data/processed/.
        
    Returns:
        train_df, test_df
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col]
    )
    
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    if save:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
        test_df.to_csv(config.TEST_DATA_PATH, index=False)
        print(f"Saved processed train split ({len(train_df)} rows) -> {config.TRAIN_DATA_PATH}")
        print(f"Saved processed test split  ({len(test_df)} rows) -> {config.TEST_DATA_PATH}")
        
    return train_df, test_df


if __name__ == "__main__":
    print("Executing Data Loader verification...")
    try:
        raw_df = load_raw_data()
        cleaned_df, report = check_missing_and_duplicates(raw_df)
        print(f"Data loading report: {report}")
        train, test = split_data(cleaned_df)
        print("Data Loader execution completed successfully.")
    except Exception as e:
        print(f"Data Loader notice: {e}")
