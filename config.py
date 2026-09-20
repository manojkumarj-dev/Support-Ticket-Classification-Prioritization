"""Central configuration module for Support Ticket Classification & Prioritization.

Defines file paths, random seeds, category/priority labels, and model hyperparameter grids.
Ensures consistency across data loading, training, evaluation, and prediction.
"""

from pathlib import Path

# ==========================================
# 1. Base Paths & Directory Structure
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "tickets.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = BASE_DIR / "models"
CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.joblib"
PRIORITY_MODEL_PATH = MODELS_DIR / "priority_model.joblib"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Random Seed for Reproducibility
# ==========================================
RANDOM_STATE = 42

# ==========================================
# 3. Target Labels & Domain Taxonomy
# ==========================================
CATEGORIES = [
    "Bug / System Error",
    "Billing & Payment",
    "Account & Access",
    "Feature Request",
    "Technical / IT Support",
]

PRIORITIES = [
    "Critical",
    "High",
    "Medium",
    "Low",
]

# Priority ordinal rank (useful for severity comparison or cost-sensitive metrics)
PRIORITY_ORDER = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}

# ==========================================
# 4. Data Split & Preprocessing Settings
# ==========================================
TEST_SIZE = 0.20
STRATIFY_COL = "category"  # Primary stratification column

# ==========================================
# 5. TF-IDF Feature Engineering Settings
# ==========================================
TFIDF_PARAMS = {
    "ngram_range": (1, 2),        # Unigrams + Bigrams to capture phrases like "cannot login", "500 error"
    "min_df": 2,                  # Ignore rare terms / typos appearing in < 2 tickets
    "max_df": 0.85,               # Ignore terms appearing in > 85% of tickets (corpus-specific stopwords)
    "max_features": 5000,         # Cap dimensionality to prevent curse of dimensionality and memory bloat
    "sublinear_tf": True,         # Apply sublinear scaling (1 + log(tf)) to dampen high-frequency term effects
}

# ==========================================
# 6. Model Training & Cross-Validation
# ==========================================
CV_FOLDS = 5
SCORING_METRIC = "f1_macro"
