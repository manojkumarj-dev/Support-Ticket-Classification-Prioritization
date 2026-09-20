"""Inference and CLI prediction interface for Support Ticket Classification & Prioritization.

Provides:
1. `predict_ticket(text: str) -> dict`: Programmatic API for downstream services.
2. CLI interface: `python src/predict.py --text "Application crashes when generating report"`
"""

import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


class TicketClassifier:
    """Production inference wrapper for category and priority pipelines."""
    
    def __init__(
        self,
        category_model_path: Path = config.CATEGORY_MODEL_PATH,
        priority_model_path: Path = config.PRIORITY_MODEL_PATH
    ):
        if not category_model_path.exists() or not priority_model_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {category_model_path} or {priority_model_path}. "
                "Please run `python src/train.py` first."
            )
            
        self.category_pipeline = joblib.load(category_model_path)
        self.priority_pipeline = joblib.load(priority_model_path)
        
    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts Category and Priority with calibrated confidence scores.
        
        Args:
            text: Raw support ticket string.
            
        Returns:
            Dict containing predicted category, priority, confidence scores, and class distributions.
        """
        # Edge case: Handle empty or whitespace-only input
        if not text or not text.strip():
            return {
                "text": text,
                "category": "Uncertain / Unclassified",
                "category_confidence": 0.0,
                "priority": "Low",
                "priority_confidence": 0.0,
                "warning": "Input text is empty or blank."
            }
            
        # 1. Category Prediction
        cat_pred = self.category_pipeline.predict([text])[0]
        if hasattr(self.category_pipeline, "predict_proba"):
            cat_probs = self.category_pipeline.predict_proba([text])[0]
            cat_classes = self.category_pipeline.classes_
            cat_conf = float(np.max(cat_probs))
            cat_prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(cat_classes, cat_probs)}
        else:
            cat_conf = None
            cat_prob_dict = {}
            
        # 2. Priority Prediction
        pri_pred = self.priority_pipeline.predict([text])[0]
        if hasattr(self.priority_pipeline, "predict_proba"):
            pri_probs = self.priority_pipeline.predict_proba([text])[0]
            pri_classes = self.priority_pipeline.classes_
            pri_conf = float(np.max(pri_probs))
            pri_prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(pri_classes, pri_probs)}
        else:
            pri_conf = None
            pri_prob_dict = {}
            
        return {
            "text": text,
            "category": str(cat_pred),
            "category_confidence": cat_conf,
            "category_probabilities": cat_prob_dict,
            "priority": str(pri_pred),
            "priority_confidence": pri_conf,
            "priority_probabilities": pri_prob_dict,
        }


def format_prediction_output(result: Dict[str, Any]) -> str:
    """Formats prediction dictionary into an elegant CLI display string."""
    cat_conf_str = f" ({result['category_confidence']:.1%} confidence)" if result.get("category_confidence") is not None else ""
    pri_conf_str = f" ({result['priority_confidence']:.1%} confidence)" if result.get("priority_confidence") is not None else ""
    
    lines = [
        "==================================================",
        f"Input Ticket : \"{result['text']}\"",
        "--------------------------------------------------",
        f"Category     : {result['category']}{cat_conf_str}",
        f"Priority     : {result['priority']}{pri_conf_str}",
        "=================================================="
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Predict category and priority for a support ticket.")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Support ticket text to classify."
    )
    args = parser.parse_args()
    
    classifier = TicketClassifier()
    result = classifier.predict(args.text)
    print(format_prediction_output(result))


if __name__ == "__main__":
    main()
