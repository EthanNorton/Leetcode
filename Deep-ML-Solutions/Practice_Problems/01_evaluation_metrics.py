"""
CMU Practice Problem: Evaluation Metrics
10-701/715: Introduction to Machine Learning
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
        y_true: True labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)
    
    Returns:
        Dictionary with tp, fp, fn, tn
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
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

