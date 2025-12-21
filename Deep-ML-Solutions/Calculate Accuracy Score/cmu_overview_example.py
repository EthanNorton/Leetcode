"""
Evaluation Metrics - CMU MSML Prep Overview
Framed for someone preparing for CMU MSML program
"""

import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate accuracy: correct predictions / total"""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def confusion_matrix_simple(y_true, y_pred, classes):
    """Simple confusion matrix visualization"""
    matrix = {}
    for true_class in classes:
        matrix[true_class] = {}
        for pred_class in classes:
            matrix[true_class][pred_class] = 0
    
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    
    return matrix

print("=" * 70)
print("EVALUATION METRICS - CMU MSML PREP OVERVIEW")
print("=" * 70)

print("\nWHY THIS MATTERS FOR CMU MSML")
print("-" * 70)
print("Relevant Courses:")
print("  - 10-701/715 (Introduction to ML): Evaluation metrics, model assessment")
print("  - 10-718 (ML in Practice): Evaluation science, reproducibility")
print("  - Research Alignment: Dr. Shah's work on evaluation pitfalls")
print("\nKey Insight: Understanding evaluation is crucial for research and practice!")

# Example 1: Basic Accuracy
print("\n\nExample 1: Accuracy Score - The Foundation")
print("-" * 70)

y_true = ['cat', 'dog', 'cat', 'bird', 'cat', 'dog', 'bird']
y_pred = ['cat', 'dog', 'bird', 'bird', 'cat', 'dog', 'bird']

print("True labels:    ", y_true)
print("Predictions:    ", y_pred)

correct = [t == p for t, p in zip(y_true, y_pred)]
print("Correct:        ", ['Yes' if c else 'No' for c in correct])

accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {sum(correct)}/{len(y_true)} = {accuracy:.2%}")

print("\nInterpretation:")
print("  - Simple: How many did we get right?")
print("  - Works well for balanced classes")
print("  - Foundation for understanding other metrics")

# Example 2: When Accuracy Fails
print("\n\nExample 2: When Accuracy Fails (Imbalanced Classes)")
print("-" * 70)

print("Medical Diagnosis Example:")
print("  - 99% of patients: No cancer")
print("  - 1% of patients: Cancer")

print("\nDumb Model: Always predict 'No cancer'")
dumb_true = ['no'] * 99 + ['yes'] * 1
dumb_pred = ['no'] * 100
dumb_acc = accuracy_score(dumb_true, dumb_pred)
print(f"  Accuracy: {dumb_acc:.2%} (looks great!)")
print("  Problem: Misses ALL cancers! (0% recall)")
print("  Real-world: Completely useless!")

print("\nLesson: Accuracy alone can be misleading!")
print("  Need: Precision, Recall, F1-Score")

# Example 3: Confusion Matrix
print("\n\nExample 3: Confusion Matrix (More Informative)")
print("-" * 70)

print("Image Classification Results:")
cm_true = ['cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'bird', 'bird', 'bird']
cm_pred = ['cat', 'cat', 'dog', 'dog', 'dog', 'bird', 'bird', 'bird', 'cat']

classes = ['cat', 'dog', 'bird']
cm = confusion_matrix_simple(cm_true, cm_pred, classes)

print("\nConfusion Matrix:")
print("                Predicted")
print("              Cat  Dog  Bird")
for i, true_class in enumerate(classes):
    row = f"Actual {true_class:4s}  "
    for pred_class in classes:
        count = cm[true_class][pred_class]
        row += f"[{count:2d}] "
    print(row)

print("\nWhat it shows:")
print("  - Diagonal: Correct predictions (True Positives)")
print("  - Off-diagonal: Errors (False Positives/Negatives)")
print("  - More informative than accuracy alone!")

# Example 4: Precision and Recall
print("\n\nExample 4: Precision & Recall (CMU 10-701/715)")
print("-" * 70)

print("Spam Detection Example:")
print("  True positives (TP): Correctly identified spam = 80")
print("  False positives (FP): Real email marked spam = 5")
print("  False negatives (FN): Spam missed = 20")

tp, fp, fn = 80, 5, 20

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f"\nPrecision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.2%}")
print("  Meaning: Of emails we marked spam, {:.1f}% were actually spam".format(precision * 100))
print("  High precision = Few false alarms")

print(f"\nRecall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.2%}")
print("  Meaning: We caught {:.1f}% of all spam".format(recall * 100))
print("  High recall = Don't miss much spam")

print("\nTrade-off:")
print("  - High precision, low recall: Very sure when we mark spam, but miss some")
print("  - Low precision, high recall: Catch most spam, but some false alarms")
print("  - Need to balance based on use case!")

# Example 5: CMU Course Connections
print("\n\nExample 5: CMU Course Connections")
print("-" * 70)

print("10-701/715 (Introduction to ML):")
print("  Topics:")
print("    - Evaluation metrics (accuracy, precision, recall, F1)")
print("    - Cross-validation")
print("    - Train/validation/test splits")
print("    - Overfitting detection")
print("  Why: Foundation for all ML work!")

print("\n10-718 (ML in Practice):")
print("  Topics:")
print("    - Evaluation pitfalls (Dr. Shah's research!)")
print("    - Reproducibility in evaluation")
print("    - Annotation bias")
print("    - Benchmark evaluation")
print("  Why: Research alignment! Connects to your interests.")

print("\nResearch Connection (Dr. Shah's Work):")
print("  - 'The More You Automate, The Less You See'")
print("  - Evaluation science")
print("  - Annotation bias")
print("  - Reviewer assignment")
print("  Your understanding of evaluation connects to this research!")

# Example 6: Real-World Challenges
print("\n\nExample 6: Real-World Evaluation Challenges")
print("-" * 70)

print("Challenge 1: Data Leakage")
print("  Problem: Test data info leaks into training")
print("  Result: Artificially high accuracy")
print("  Solution: Proper train/test splits")

print("\nChallenge 2: Distribution Shift")
print("  Problem: Training data != Real-world data")
print("  Example: Train on ImageNet, deploy on phone photos")
print("  Result: High training accuracy, poor real-world performance")

print("\nChallenge 3: Evaluation Bias")
print("  Problem: Test set doesn't represent real distribution")
print("  Example: Test set 90% class A, real world 50/50")
print("  Result: Misleading metrics")

print("\nChallenge 4: Metric Gaming")
print("  Problem: Optimizing for metric that doesn't matter")
print("  Example: High accuracy on easy examples, fails on hard ones")
print("  Solution: Use multiple metrics, understand what matters")

# Example 7: Evaluation Best Practices
print("\n\nExample 7: Evaluation Best Practices (CMU-Level)")
print("-" * 70)

print("1. Use Multiple Metrics")
print("   - Don't rely on accuracy alone")
print("   - Use precision, recall, F1, ROC-AUC")
print("   - Understand what each tells you")

print("\n2. Proper Data Splits")
print("   - Train: Learn from this")
print("   - Validation: Tune hyperparameters")
print("   - Test: Final evaluation (only once!)")

print("\n3. Cross-Validation")
print("   - K-fold cross-validation")
print("   - More robust evaluation")
print("   - Reduces variance in estimates")

print("\n4. Understand Your Domain")
print("   - Medical: Recall matters (don't miss cancers)")
print("   - Spam detection: Precision matters (don't block real emails)")
print("   - Research: Understand what actually matters!")

# Summary
print("\n" + "=" * 70)
print("KEY TAKEAWAYS FOR CMU PREP")
print("=" * 70)
print("\n1. Evaluation is Fundamental")
print("   - Every ML project needs evaluation")
print("   - Understanding metrics is essential")
print("   - Connects to research (evaluation science)")

print("\n2. Accuracy is Just the Start")
print("   - Many metrics exist for good reasons")
print("   - Different metrics for different problems")
print("   - Need to understand trade-offs")

print("\n3. Evaluation is Hard")
print("   - Many pitfalls and biases")
print("   - Requires careful design")
print("   - Active research area (Dr. Shah!)")

print("\n4. Connects to Your Research Interests")
print("   - Evaluation science")
print("   - Annotation bias")
print("   - Reproducibility")
print("   - All relate to evaluation metrics!")

print("\n" + "=" * 70)
print("BOTTOM LINE")
print("=" * 70)
print("Evaluation metrics are how we measure ML success.")
print("\nFor CMU MSML:")
print("  - Foundation: 10-701/715 covers metrics")
print("  - Practice: 10-718 covers evaluation science")
print("  - Research: Connects to Dr. Shah's work")
print("  - Career: Essential for any ML role")
print("\nMastering evaluation = Essential ML skill!")
print("=" * 70)

