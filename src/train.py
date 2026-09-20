"""Model training, cross-validation, hyperparameter tuning, and persistence module.

Trains and evaluates:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (LinearSVC with CalibratedClassifierCV)

For both Ticket Category and Ticket Priority.
Selects best models based on 5-fold Stratified Cross-Validation (Macro F1)
and persists end-to-end pipelines to disk with joblib.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.preprocessing import TextCleanerTransformer
from src.feature_engineering import build_tfidf_vectorizer
from src.data_loader import load_raw_data, check_missing_and_duplicates, split_data


def create_candidate_pipelines() -> Dict[str, Pipeline]:
    """Builds the 3 candidate model pipelines with identical preprocessing & TF-IDF vectorizer."""
    pipelines = {
        "Multinomial Naive Bayes": Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", build_tfidf_vectorizer()),
            ("classifier", MultinomialNB(alpha=0.5))
        ]),
        "Logistic Regression": Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", build_tfidf_vectorizer()),
            ("classifier", LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=config.RANDOM_STATE
            ))
        ]),
        "Linear SVM (Calibrated)": Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", build_tfidf_vectorizer()),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE
                ),
                method="sigmoid",
                cv=3
            ))
        ])
    }
    return pipelines


def evaluate_candidate_models(
    X_train: pd.Series,
    y_train: pd.Series,
    target_name: str
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Runs 5-fold Stratified Cross-Validation on all candidate models.
    
    Returns:
        Best model name, and a dictionary of cross-validation results.
    """
    print(f"\n=======================================================")
    print(f"Benchmarking Models for Target: '{target_name}' (5-Fold Stratified CV)")
    print(f"=======================================================")
    
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    candidate_pipelines = create_candidate_pipelines()
    results = {}
    best_model_name = None
    best_score = -1.0
    
    for model_name, pipeline in candidate_pipelines.items():
        # Evaluate using macro F1 (balances performance across all classes)
        f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=config.SCORING_METRIC, n_jobs=-1)
        acc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        
        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        mean_acc = float(np.mean(acc_scores))
        std_acc = float(np.std(acc_scores))
        
        results[model_name] = {
            "mean_f1_macro": mean_f1,
            "std_f1_macro": std_f1,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc
        }
        
        print(f"[{model_name}]")
        print(f"  -> Macro F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
        print(f"  -> Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
        
        if mean_f1 > best_score:
            best_score = mean_f1
            best_model_name = model_name
            
    print(f"\n>> Selected Winner for {target_name}: '{best_model_name}' (Macro F1 = {best_score:.4f})")
    return best_model_name, results


def tune_and_train_best_model(
    X_train: pd.Series,
    y_train: pd.Series,
    best_model_name: str,
    target_name: str,
    save_path: Path
) -> Pipeline:
    """Performs targeted hyperparameter tuning on the winning model architecture,
    fits the final tuned pipeline on the complete training set, and saves to disk.
    """
    print(f"\n--- Hyperparameter Tuning for {target_name} ({best_model_name}) ---")
    
    if "Logistic Regression" in best_model_name:
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "classifier__C": [0.5, 1.0, 2.0],
        }
        base_pipeline = Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", build_tfidf_vectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE))
        ])
    elif "Linear SVM" in best_model_name:
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__estimator__C": [0.5, 1.0, 2.0],
        }
        base_pipeline = Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", build_tfidf_vectorizer()),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", random_state=config.RANDOM_STATE),
                method="sigmoid",
                cv=3
            ))
        ])
    else:  # Multinomial Naive Bayes
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__alpha": [0.1, 0.5, 1.0],
        }
        base_pipeline = Pipeline([
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", build_tfidf_vectorizer()),
            ("classifier", MultinomialNB())
        ])
        
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=config.SCORING_METRIC,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Optimal Hyperparameters: {grid_search.best_params_}")
    print(f"Best CV Macro F1: {grid_search.best_score_:.4f}")
    
    final_pipeline = grid_search.best_estimator_
    
    # Save trained pipeline
    joblib.dump(final_pipeline, save_path)
    print(f"Saved optimized {target_name} pipeline -> {save_path}")
    
    return final_pipeline


def train_all_models():
    """Main training orchestration function."""
    print("Initializing Training Pipeline...")
    
    # 1. Ensure train and test data exist
    if not config.TRAIN_DATA_PATH.exists():
        print("Processed data not found. Loading and splitting raw data...")
        raw_df = load_raw_data()
        cleaned_df, _ = check_missing_and_duplicates(raw_df)
        train_df, test_df = split_data(cleaned_df)
    else:
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        print(f"Loaded training data: {len(train_df)} rows from {config.TRAIN_DATA_PATH}")
        
    X_train = train_df["text"]
    y_train_cat = train_df["category"]
    y_train_pri = train_df["priority"]
    
    # 2. Train Category Model
    best_cat_model, cat_results = evaluate_candidate_models(X_train, y_train_cat, "Category")
    tune_and_train_best_model(X_train, y_train_cat, best_cat_model, "Category", config.CATEGORY_MODEL_PATH)
    
    # 3. Train Priority Model
    best_pri_model, pri_results = evaluate_candidate_models(X_train, y_train_pri, "Priority")
    tune_and_train_best_model(X_train, y_train_pri, best_pri_model, "Priority", config.PRIORITY_MODEL_PATH)
    
    print("\nTraining completed successfully! Both models are trained, calibrated, and saved.")


if __name__ == "__main__":
    train_all_models()
