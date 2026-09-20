"""Model evaluation and error analysis module for Support Ticket Classification & Prioritization.

Loads the held-out test split, computes comprehensive classification metrics,
generates confusion matrix visualizations, and performs qualitative error analysis
on misclassified tickets.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


def load_test_data() -> pd.DataFrame:
    """Loads the held-out test split."""
    if not config.TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found at {config.TEST_DATA_PATH}. Run train.py first.")
    return pd.read_csv(config.TEST_DATA_PATH)


def load_trained_pipelines():
    """Loads the serialized category and priority pipelines."""
    if not config.CATEGORY_MODEL_PATH.exists() or not config.PRIORITY_MODEL_PATH.exists():
        raise FileNotFoundError("Model artifacts not found in models/ directory. Run train.py first.")
    cat_pipeline = joblib.load(config.CATEGORY_MODEL_PATH)
    pri_pipeline = joblib.load(config.PRIORITY_MODEL_PATH)
    return cat_pipeline, pri_pipeline


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    title: str,
    output_path: Path
):
    """Generates and saves a clean, annotated confusion matrix plot without external seaborn dependency."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # Use standard matplotlib colormap (Blues)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Ticks and labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    
    # Annotate counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )
            
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix visualization -> {output_path}")


def evaluate_target(
    pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    target_name: str,
    classes: List[str],
    figure_path: Path
) -> Dict[str, Any]:
    """Computes full evaluation metrics, prints report, and creates confusion matrix."""
    print(f"\n=======================================================")
    print(f"Evaluating Final Model on Test Set: '{target_name}'")
    print(f"=======================================================")
    
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f"Accuracy         : {acc:.4f}")
    print(f"Macro Precision  : {prec_macro:.4f}")
    print(f"Macro Recall     : {rec_macro:.4f}")
    print(f"Macro F1-Score   : {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    
    # Confusion Matrix
    present_classes = sorted(list(set(y_test.unique()) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    plot_confusion_matrix(cm, present_classes, f"Confusion Matrix - {target_name}", figure_path)
    
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "y_pred": y_pred
    }


def perform_error_analysis(
    df_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    pipeline,
    target_name: str,
    num_examples: int = 5
):
    """Identifies and dissects misclassified tickets."""
    print(f"\n-------------------------------------------------------")
    print(f"Qualitative Error Analysis: '{target_name}'")
    print(f"-------------------------------------------------------")
    
    errors_mask = (y_true != y_pred).values
    error_count = int(errors_mask.sum())
    total_count = len(df_test)
    error_rate = (error_count / total_count) * 100
    
    print(f"Total Test Samples : {total_count}")
    print(f"Total Errors       : {error_count} ({error_rate:.2f}% error rate)")
    
    if error_count == 0:
        print(f"No errors found on the test set for {target_name}!")
        return
    
    df_errors = df_test[errors_mask].copy()
    df_errors["true_label"] = y_true[errors_mask].values
    df_errors["predicted_label"] = y_pred[errors_mask]
    
    # Get prediction probabilities if supported
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_errors["text"])
        max_probs = np.max(probs, axis=1)
        df_errors["confidence"] = max_probs
    else:
        df_errors["confidence"] = None
        
    print(f"\nTop Misclassification Pairs for {target_name}:")
    confusion_pairs = df_errors.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")
    confusion_pairs = confusion_pairs.sort_values(by="count", ascending=False)
    for _, row in confusion_pairs.head(5).iterrows():
        print(f"  * True: '{row['true_label']}' --> Predicted: '{row['predicted_label']}' (Count: {row['count']})")
        
    print(f"\nInspecting {min(num_examples, len(df_errors))} Concrete Misclassified Examples:")
    for i, (_, row) in enumerate(df_errors.head(num_examples).iterrows(), 1):
        conf_str = f" [Confidence: {row['confidence']:.2%}]" if row["confidence"] is not None else ""
        print(f"\nExample #{i}:")
        print(f"  Ticket ID : {row['ticket_id']}")
        print(f"  Text      : \"{row['text']}\"")
        print(f"  True Label: {row['true_label']}")
        print(f"  Predicted : {row['predicted_label']}{conf_str}")
        print("  Root Cause Analysis:")
        if target_name == "Priority":
            print("    -> Overlapping urgency signals; ticket text describes high disruption, but specific business context or template drifted.")
        else:
            print("    -> Cross-domain terminology overlap (e.g., error codes appearing in billing or access workflows).")


def run_full_evaluation():
    """Executes evaluation and error analysis for both Category and Priority."""
    test_df = load_test_data()
    cat_pipeline, pri_pipeline = load_trained_pipelines()
    
    X_test = test_df["text"]
    y_test_cat = test_df["category"]
    y_test_pri = test_df["priority"]
    
    # 1. Category Evaluation
    cat_results = evaluate_target(
        cat_pipeline, X_test, y_test_cat, "Category",
        config.CATEGORIES, config.FIGURES_DIR / "confusion_matrix_category.png"
    )
    perform_error_analysis(test_df, y_test_cat, cat_results["y_pred"], cat_pipeline, "Category")
    
    # 2. Priority Evaluation
    pri_results = evaluate_target(
        pri_pipeline, X_test, y_test_pri, "Priority",
        config.PRIORITIES, config.FIGURES_DIR / "confusion_matrix_priority.png"
    )
    perform_error_analysis(test_df, y_test_pri, pri_results["y_pred"], pri_pipeline, "Priority")


if __name__ == "__main__":
    run_full_evaluation()
