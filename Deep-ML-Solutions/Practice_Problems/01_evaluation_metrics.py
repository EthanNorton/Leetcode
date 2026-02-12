"""
CMU Practice Problem: Evaluation Metrics
10-701/715: Introduction to Machine Learning

===========================================
LLM EVALUATION MOTIVATION
===========================================
LLM outputs are often evaluated by classifiers or human judges that produce
binary labels (e.g., "helpful"/"not helpful", "safe"/"unsafe", "correct"/"wrong").
This file provides the core metrics for those binary classification tasks.

Common LLM eval scenarios:
  - Safety classifier: toxic vs non-toxic (e.g., RTP toxicity scoring)
  - Factuality: correct vs hallucinated
  - Instruction following: followed vs not followed
  - Helpfulness: helpful vs unhelpful

Once you have y_true (human/model labels) and y_pred (LLM or classifier outputs),
use these functions to compute metrics and compare models.
"""

import numpy as np


def calculate_metrics(tp, fp, fn, tn):
    """
    Calculate classification metrics from confusion matrix components.

    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        tn: True Negatives

    Returns:
        Dictionary with accuracy, precision, recall, f1_score

    LLM EVAL USE:
        - Accuracy: Overall correctness when classes are balanced. Often misleading
          for imbalanced data (e.g., few toxic examples). Use with caution.
        - Precision: Fraction of positive predictions that are correct. Important
          when false positives are costly (e.g., wrongly flagging safe content).
        - Recall: Fraction of actual positives that were caught. Important when
          missing positives is costly (e.g., safety filter missing toxic content).
        - F1: Balances precision and recall. Good default when you care about
          both false positives and false negatives equally.
    """
    # Total predictions
    total = tp + fp + fn + tn
    
    # Accuracy: (TP + TN) / Total
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall (Sensitivity): TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1-Score: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def confusion_matrix_from_predictions(y_true, y_pred):
    """
    Calculate confusion matrix from predictions.

    Args:
        y_true: True labels (binary: 0 or 1) - e.g., human annotations or gold labels
        y_pred: Predicted labels (binary: 0 or 1) - e.g., classifier output or thresholded score

    Returns:
        Dictionary with tp, fp, fn, tn

    LLM EVAL USE:
        - y_true: Labels from human raters, answer key, or another trusted model.
        - y_pred: Output from a toxicity classifier (e.g., Perspective API), factuality
          checker, or instruction-following detector. For continuous scores, convert
          to binary with a threshold (e.g., toxicity > 0.5 -> 1).
        - Example: Evaluate RTP toxicity by thresholding toxicity scores and comparing
          to human-annotated safe/unsafe labels.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# Test cases
if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATION METRICS - PRACTICE PROBLEM")
    print("=" * 60)
    print()
    
    # Test 1: Given confusion matrix
    print("Test 1: Calculate metrics from confusion matrix")
    print("Confusion Matrix:")
    print("                Predicted")
    print("              Positive  Negative")
    print("Actual Pos       80       20")
    print("      Neg        10       90")
    print()
    
    metrics = calculate_metrics(tp=80, fp=10, fn=20, tn=90)
    
    print("Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1_score']:.3f}")
    print()
    
    # Expected:
    # Accuracy = 170/200 = 0.85
    # Precision = 80/90 = 0.889
    # Recall = 80/100 = 0.80
    # F1 = 2 × (0.889 × 0.80) / (0.889 + 0.80) = 0.842
    
    # Test 2: From predictions
    print("Test 2: Calculate from predictions")
    y_true = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0]
    
    cm = confusion_matrix_from_predictions(y_true, y_pred)
    metrics2 = calculate_metrics(**cm)
    
    print(f"True labels:  {y_true}")
    print(f"Pred labels:  {y_pred}")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {cm['tp']}, FP: {cm['fp']}")
    print(f"  FN: {cm['fn']}, TN: {cm['tn']}")
    print()
    print("Metrics:")
    print(f"  Accuracy:  {metrics2['accuracy']:.3f}")
    print(f"  Precision: {metrics2['precision']:.3f}")
    print(f"  Recall:    {metrics2['recall']:.3f}")
    print(f"  F1-Score:  {metrics2['f1_score']:.3f}")
    print()
    
    # =========================================================================
    # LLM EVAL EXAMPLE: Toxicity classification (like RTP dataset)
    # =========================================================================
    print()
    print("=" * 60)
    print("LLM EVAL EXAMPLE: Toxicity Classification")
    print("=" * 60)
    print()
    print("Scenario: You threshold toxicity scores (e.g., from Real Toxicity Prompts)")
    print("          and compare to human labels. 1 = toxic, 0 = safe.")
    print()
    # Simulated: 10 prompts, human labels vs model's toxicity classifier
    y_true_llm = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0])  # Human labels
    y_pred_llm = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 1])   # Classifier (FP on idx 4, FN on idx 3)
    cm_llm = confusion_matrix_from_predictions(y_true_llm, y_pred_llm)
    metrics_llm = calculate_metrics(**cm_llm)
    print("Confusion matrix (TP, FP, FN, TN):", cm_llm)
    print("F1 (balanced metric for imbalanced safety data):", f"{metrics_llm['f1_score']:.3f}")
    print("Precision (avoid over-flagging safe content):", f"{metrics_llm['precision']:.3f}")
    print("Recall (catch toxic content):", f"{metrics_llm['recall']:.3f}")
    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

