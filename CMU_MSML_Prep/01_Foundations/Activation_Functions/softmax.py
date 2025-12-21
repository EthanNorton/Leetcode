"""
SOFTMAX ACTIVATION FUNCTION - COMPLETE WALKTHROUGH
==================================================

WHAT IS SOFTMAX?
----------------
Softmax converts a vector of scores (logits) into a probability distribution.
It ensures all outputs sum to 1 and are between 0 and 1.

Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

Example:
  Input scores:  [2.0, 1.0, 0.1]
  Output probabilities: [0.659, 0.242, 0.099]  (sums to 1.0)

WHY IS IT SO IMPORTANT?
-----------------------
1. Used in multi-class classification (final layer)
2. Converts raw scores to probabilities
3. Used in transformers (GPT, BERT) for attention
4. Works with cross-entropy loss
5. Makes outputs interpretable (probabilities!)

WHERE IS IT USED?
-----------------
- Final layer of classification networks
- Language models (GPT, BERT, T5)
- Image classification (ResNet, VGG)
- Any multi-class classification problem
"""

import math

def softmax(scores: list[float]) -> list[float]:
    """
    Convert scores (logits) to probability distribution.
    
    Parameters:
    -----------
    scores : list of floats
        Raw scores/logits from neural network (can be any numbers)
    
    Returns:
    --------
    list of floats
        Probability distribution (all values between 0 and 1, sum to 1.0)
    
    Formula:
    --------
    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    Examples:
    --------
    >>> softmax([1.0, 2.0, 3.0])
    [0.09, 0.2447, 0.6652]  (approximately)
    
    >>> softmax([2.0, 1.0, 0.1])
    [0.659, 0.242, 0.099]  (approximately)
    """
    # Step 1: Apply exponential to all scores
    # This makes all values positive and amplifies differences
    # Example: [2, 1, 0.1] -> [7.39, 2.72, 1.11]
    exp_scores = [math.exp(score) for score in scores]
    
    # Step 2: Calculate sum of all exponentials
    # This is the normalization factor
    # Example: 7.39 + 2.72 + 1.11 = 11.22
    total = sum(exp_scores)
    
    # Step 3: Divide each exponential by the total
    # This normalizes so all probabilities sum to 1.0
    # Example: [7.39/11.22, 2.72/11.22, 1.11/11.22] = [0.659, 0.242, 0.099]
    probabilities = [round(score / total, 4) for score in exp_scores]
    
    return probabilities


# ============================================================================
# COMPREHENSIVE EXAMPLES AND EXPLANATIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOFTMAX ACTIVATION FUNCTION - COMPLETE WALKTHROUGH")
    print("=" * 70)
    
    # Example 1: Basic Understanding
    print("\nExample 1: Basic Softmax Calculation")
    print("-" * 70)
    
    scores1 = [2.0, 1.0, 0.1]
    print(f"Input scores (logits): {scores1}")
    print("\nStep-by-step calculation:")
    
    # Show intermediate steps
    exp_scores1 = [math.exp(s) for s in scores1]
    total1 = sum(exp_scores1)
    probs1 = [s / total1 for s in exp_scores1]
    
    print(f"\nStep 1: Apply exponential (exp):")
    for i, (score, exp_score) in enumerate(zip(scores1, exp_scores1)):
        print(f"  exp({score}) = {exp_score:.4f}")
    
    print(f"\nStep 2: Sum all exponentials:")
    print(f"  Total = {exp_scores1[0]:.4f} + {exp_scores1[1]:.4f} + {exp_scores1[2]:.4f} = {total1:.4f}")
    
    print(f"\nStep 3: Normalize (divide each by total):")
    for i, (exp_score, prob) in enumerate(zip(exp_scores1, probs1)):
        print(f"  Probability {i} = {exp_score:.4f} / {total1:.4f} = {prob:.4f}")
    
    result1 = softmax(scores1)
    print(f"\nFinal probabilities: {result1}")
    print(f"Sum: {sum(result1):.4f} (should be 1.0)")
    print("\nInterpretation:")
    print("  Class 0: 65.9% probability (most likely!)")
    print("  Class 1: 24.2% probability")
    print("  Class 2: 9.9% probability")
    
    # Example 2: Image Classification
    print("\n\nExample 2: Image Classification (Cat, Dog, Bird)")
    print("-" * 70)
    
    # Raw scores from neural network
    image_scores = {
        "cat": 5.2,
        "dog": 2.1,
        "bird": 0.8
    }
    
    print("Raw scores from neural network:")
    for class_name, score in image_scores.items():
        print(f"  {class_name:5s}: {score:.1f}")
    
    scores_list = list(image_scores.values())
    probabilities = softmax(scores_list)
    
    print("\nAfter Softmax (probabilities):")
    for (class_name, _), prob in zip(image_scores.items(), probabilities):
        percentage = prob * 100
        print(f"  {class_name:5s}: {prob:.4f} ({percentage:.1f}%)")
    
    print("\nInterpretation:")
    max_idx = probabilities.index(max(probabilities))
    predicted_class = list(image_scores.keys())[max_idx]
    confidence = probabilities[max_idx] * 100
    print(f"  Predicted: {predicted_class} ({confidence:.1f}% confident)")
    
    # Example 3: Why Exponential?
    print("\n\nExample 3: Why Use Exponential?")
    print("-" * 70)
    
    print("Problem: Raw scores can be negative, zero, or any value")
    print("Solution: Exponential makes everything positive and amplifies differences")
    
    test_scores = [-1.0, 0.0, 1.0, 2.0]
    print(f"\nScores: {test_scores}")
    
    print("\nWithout exponential (just normalize):")
    total_raw = sum(test_scores)
    if total_raw != 0:
        raw_probs = [s / total_raw for s in test_scores]
        print(f"  Probabilities: {[round(p, 4) for p in raw_probs]}")
        print("  Problem: Negative scores give negative probabilities!")
    else:
        print("  Problem: Can't normalize if sum is zero!")
    
    print("\nWith exponential (Softmax):")
    exp_test = [math.exp(s) for s in test_scores]
    total_exp = sum(exp_test)
    softmax_probs = [s / total_exp for s in exp_test]
    print(f"  exp(scores): {[round(e, 4) for e in exp_test]}")
    print(f"  Probabilities: {[round(p, 4) for p in softmax_probs]}")
    print("  All positive! All sum to 1!")
    
    print("\nKey insight: Exponential amplifies differences")
    print("  Small difference in scores -> Large difference in probabilities")
    print("  Example: Score 2.0 vs 1.0 (difference of 1.0)")
    print(f"  But probability: {softmax_probs[3]:.4f} vs {softmax_probs[2]:.4f}")
    print(f"  (difference of {abs(softmax_probs[3] - softmax_probs[2]):.4f})")
    
    # Example 4: Numerical Stability Issue
    print("\n\nExample 4: Numerical Stability (Advanced)")
    print("-" * 70)
    
    print("Problem: Large scores cause exp() to overflow")
    large_scores = [100, 101, 102]
    print(f"\nLarge scores: {large_scores}")
    print("  exp(100) is HUGE! (2.69 × 10^43)")
    print("  Can cause overflow in computers")
    
    print("\nSolution: Subtract max score before applying exp")
    print("  This doesn't change the result (mathematically equivalent)")
    print("  But prevents overflow!")
    
    max_score = max(large_scores)
    stabilized_scores = [s - max_score for s in large_scores]
    print(f"\nOriginal: {large_scores}")
    print(f"Stabilized: {stabilized_scores} (subtract max = {max_score})")
    
    # Show both give same result
    result_original = softmax(large_scores)
    result_stabilized = softmax(stabilized_scores)
    print(f"\nOriginal softmax: {result_original}")
    print(f"Stabilized softmax: {result_stabilized}")
    print("  Same result! But stabilized is safer.")
    
    # Example 5: Comparison with Alternatives
    print("\n\nExample 5: Softmax vs Alternatives")
    print("-" * 70)
    
    comparison_scores = [3.0, 1.0, 0.5]
    
    print(f"Scores: {comparison_scores}")
    
    # Softmax
    softmax_result = softmax(comparison_scores)
    print(f"\nSoftmax: {softmax_result}")
    print("  - Sums to 1.0")
    print("  - All between 0 and 1")
    print("  - Probabilities!")
    
    # Just normalize (divide by sum)
    total = sum(comparison_scores)
    normalized = [s / total for s in comparison_scores]
    print(f"\nJust normalize: {[round(n, 4) for n in normalized]}")
    print("  - Also sums to 1.0")
    print("  - But doesn't amplify differences")
    print("  - Less useful for classification")
    
    # Argmax (just pick highest)
    max_idx = comparison_scores.index(max(comparison_scores))
    print(f"\nArgmax (just pick highest): Class {max_idx}")
    print("  - Gives hard decision (100% one class)")
    print("  - No probability information")
    print("  - Can't see confidence")
    
    print("\nSoftmax advantage: Gives probabilities with confidence!")
    
    # Example 6: Real-World Use Cases
    print("\n\nExample 6: Real-World Applications")
    print("-" * 70)
    
    print("1. Image Classification:")
    print("   Input: Image of animal")
    print("   Scores: [cat: 8.5, dog: 2.1, bird: 0.3]")
    img_scores = [8.5, 2.1, 0.3]
    img_probs = softmax(img_scores)
    print(f"   Probabilities: {[round(p, 4) for p in img_probs]}")
    print(f"   Prediction: Cat ({img_probs[0]*100:.1f}% confident)")
    
    print("\n2. Language Models (GPT):")
    print("   Input: 'The cat sat on the'")
    print("   Scores for next word: [mat: 5.2, floor: 3.1, table: 1.8]")
    word_scores = [5.2, 3.1, 1.8]
    word_probs = softmax(word_scores)
    print(f"   Probabilities: {[round(p, 4) for p in word_probs]}")
    print(f"   Most likely: 'mat' ({word_probs[0]*100:.1f}%)")
    
    print("\n3. Sentiment Analysis:")
    print("   Input: 'I love this movie!'")
    print("   Scores: [positive: 7.2, neutral: 1.5, negative: 0.1]")
    sentiment_scores = [7.2, 1.5, 0.1]
    sentiment_probs = softmax(sentiment_scores)
    print(f"   Probabilities: {[round(p, 4) for p in sentiment_probs]}")
    print(f"   Sentiment: Positive ({sentiment_probs[0]*100:.1f}%)")
    
    # Example 7: Properties of Softmax
    print("\n\nExample 7: Key Properties of Softmax")
    print("-" * 70)
    
    test_scores2 = [1.0, 2.0, 3.0]
    probs2 = softmax(test_scores2)
    
    print(f"Scores: {test_scores2}")
    print(f"Probabilities: {probs2}")
    
    print("\nProperty 1: All probabilities sum to 1.0")
    print(f"  Sum: {sum(probs2):.4f} (correct!)")
    
    print("\nProperty 2: All probabilities are between 0 and 1")
    print(f"  All in range [0, 1]: {all(0 <= p <= 1 for p in probs2)} (correct!)")
    
    print("\nProperty 3: Higher score = Higher probability")
    print(f"  Score 3.0 (highest) -> Probability {probs2[2]:.4f} (highest) (correct!)")
    print(f"  Score 1.0 (lowest) -> Probability {probs2[0]:.4f} (lowest) (correct!)")
    
    print("\nProperty 4: Preserves order (monotonic)")
    print("  If score_i > score_j, then prob_i > prob_j (correct!)")
    
    # Example 8: Common Mistakes
    print("\n\nExample 8: Common Mistakes to Avoid")
    print("-" * 70)
    
    print("Mistake 1: Forgetting to apply exponential")
    wrong_scores = [2.0, 1.0, 0.1]
    wrong_total = sum(wrong_scores)
    wrong_probs = [s / wrong_total for s in wrong_scores]
    correct_probs = softmax(wrong_scores)
    print(f"  Scores: {wrong_scores}")
    print(f"  Wrong (no exp): {[round(p, 4) for p in wrong_probs]}")
    print(f"  Correct (softmax): {correct_probs}")
    print("  Problem: Doesn't amplify differences properly!")
    
    print("\nMistake 2: Applying softmax to each score independently")
    print("  Wrong: softmax([2]) = [1.0], softmax([1]) = [1.0]")
    print("  Correct: softmax([2, 1]) = [0.73, 0.27]")
    print("  Problem: Need all scores together for normalization!")
    
    print("\nMistake 3: Not handling numerical stability")
    print("  For large scores, use: scores - max(scores) before exp")
    print("  Prevents overflow!")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Softmax converts scores to probabilities (sum to 1.0)")
    print("2. Uses exponential to make all values positive")
    print("3. Amplifies differences between scores")
    print("4. Used in final layer of classification networks")
    print("5. Works with cross-entropy loss")
    print("6. Makes outputs interpretable (confidence levels)")
    print("7. Essential for multi-class classification")
    print("=" * 70)
    
    # Interactive test
    print("\n\nTest Your Understanding:")
    print("-" * 70)
    test_cases = [
        ([1.0, 1.0, 1.0], "Equal scores"),
        ([10.0, 1.0, 0.1], "One very high score"),
        ([-1.0, 0.0, 1.0], "Negative and zero scores"),
        ([0.1, 0.2, 0.3], "Small scores")
    ]
    
    for scores, description in test_cases:
        result = softmax(scores)
        max_prob = max(result)
        max_idx = result.index(max_prob)
        print(f"\n{description}:")
        print(f"  Scores: {scores}")
        print(f"  Probabilities: {[round(r, 4) for r in result]}")
        print(f"  Sum: {sum(result):.4f}")
        print(f"  Highest probability: Class {max_idx} ({max_prob*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Softmax Complete! Next: Learn about Cross-Entropy Loss!")
    print("=" * 70)
