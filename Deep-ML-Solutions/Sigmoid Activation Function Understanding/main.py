"""
SIGMOID ACTIVATION FUNCTION - COMPLETE WALKTHROUGH
==================================================

WHAT IS SIGMOID?
----------------
Sigmoid is an activation function that squashes any input to a value between 0 and 1.
It's S-shaped (sigmoid = "S-shaped" in Greek).

Formula: sigmoid(x) = 1 / (1 + e^(-x))

Visual:
  Input:  -5  -2  -1   0   1   2   5
  Output:  0   0.1  0.3  0.5  0.7  0.9  1.0

WHY IS IT IMPORTANT?
--------------------
1. Used in binary classification (logistic regression)
2. Outputs probabilities (0 to 1 range)
3. Smooth, differentiable (good for gradients)
4. Historical importance (used before ReLU became popular)

WHERE IS IT USED?
-----------------
- Logistic regression (binary classification)
- Output layer for binary problems
- Gating mechanisms (LSTM, GRU)
- Less common now (ReLU is more popular)

KEY DIFFERENCE FROM ReLU:
- ReLU: Output can be any positive number
- Sigmoid: Output always between 0 and 1 (probability!)
"""

import math
from math import exp 
import numpy as np 

def sigmoid(z: float) -> float:
    """
    Sigmoid activation function.
    
    Parameters:
    -----------
    z : float
        Input value (can be any number)
    
    Returns:
    --------
    float
        Value between 0 and 1
    
    Formula: sigmoid(z) = 1 / (1 + e^(-z))
    
    Examples:
    --------
    >>> sigmoid(0)
    0.5
    >>> sigmoid(5)
    0.9933 (approximately)
    >>> sigmoid(-5)
    0.0067 (approximately)
    """
    # Step 1: Calculate e^(-z)
    # This makes large negative numbers → very small
    # Large positive numbers → very large
    exp_neg_z = exp(-z)
    
    # Step 2: Add 1
    # 1 + e^(-z)
    denominator = 1 + exp_neg_z
    
    # Step 3: Divide 1 by denominator
    # This gives us a value between 0 and 1
    result = 1 / denominator
    
    # Step 4: Round to 4 decimal places
    return np.round(result, 4)


# ============================================================================
# COMPREHENSIVE EXAMPLES AND EXPLANATIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SIGMOID ACTIVATION FUNCTION - COMPLETE WALKTHROUGH")
    print("=" * 70)
    
    # Example 1: Basic Understanding
    print("\nExample 1: Basic Sigmoid Behavior")
    print("-" * 70)
    
    test_values = [-5, -2, -1, 0, 1, 2, 5]
    print("Input -> Output:")
    for val in test_values:
        result = sigmoid(val)
        print(f"  {val:4.1f} -> {result:.4f}")
    
    print("\nKey observations:")
    print("  - All outputs between 0 and 1")
    print("  - Negative inputs -> small outputs (close to 0)")
    print("  - Positive inputs -> large outputs (close to 1)")
    print("  - Zero input -> 0.5 (middle)")
    
    # Example 2: Why S-Shaped?
    print("\n\nExample 2: Why S-Shaped?")
    print("-" * 70)
    
    print("Sigmoid curve:")
    print("""
    Output
      1.0 |                    *
          |                  *
          |                *
      0.5 |              *
          |            *
          |          *
      0.0 |*--------*--------*--------*--------*
          -5      -2       0       2       5   Input
    """)
    
    print("Why S-shaped?")
    print("  - Steep in middle (around 0)")
    print("  - Flat at extremes (very negative or very positive)")
    print("  - Smooth transition")
    print("  - Always increasing (monotonic)")
    
    # Example 3: Binary Classification
    print("\n\nExample 3: Binary Classification Use Case")
    print("-" * 70)
    
    print("Spam Detection Example:")
    print("  Input: Email features -> Neural network -> Raw score")
    print("  Raw score: -2.5 (negative = not spam)")
    sig_neg = sigmoid(-2.5)
    print("  After sigmoid: {:.4f} ({:.1f}% probability of spam)".format(
        sig_neg, sig_neg * 100))
    
    print("\n  Raw score: 3.2 (positive = spam)")
    sig_pos = sigmoid(3.2)
    print("  After sigmoid: {:.4f} ({:.1f}% probability of spam)".format(
        sig_pos, sig_pos * 100))
    
    print("\nInterpretation:")
    print("  - Output < 0.5: Not spam (low probability)")
    print("  - Output > 0.5: Spam (high probability)")
    print("  - Output = 0.5: Uncertain (exactly in middle)")
    
    # Example 4: Comparison with ReLU
    print("\n\nExample 4: Sigmoid vs ReLU")
    print("-" * 70)
    
    comparison_values = [-2, 0, 2, 5]
    print("Input | Sigmoid | ReLU")
    print("-" * 70)
    for val in comparison_values:
        sig = sigmoid(val)
        relu = max(0, val)
        print(f"  {val:3.0f}  |  {sig:.4f}  |  {relu:.1f}")
    
    print("\nKey Differences:")
    print("  Sigmoid:")
    print("    - Always outputs 0-1 (probability)")
    print("    - Smooth curve")
    print("    - Can have vanishing gradient problem")
    print("    - Used for binary classification")
    
    print("\n  ReLU:")
    print("    - Outputs 0 to infinity")
    print("    - Linear for positive, flat for negative")
    print("    - No vanishing gradient (for positives)")
    print("    - Used in hidden layers")
    
    # Example 5: Vanishing Gradient Problem
    print("\n\nExample 5: The Vanishing Gradient Problem")
    print("-" * 70)
    
    print("Problem with Sigmoid:")
    print("  - Gradient is small for extreme values")
    print("  - In deep networks, gradients multiply")
    print("  - After many layers: gradient ≈ 0")
    print("  - Network stops learning!")
    
    print("\nExample:")
    print("  Layer 1 gradient: 0.2")
    print("  Layer 2 gradient: 0.2")
    print("  Layer 3 gradient: 0.2")
    print("  After 10 layers: 0.2^10 = 0.0000001 (vanished!)")
    
    print("\nWhy ReLU is better:")
    print("  - Gradient = 1 for positive values")
    print("  - After 10 layers: 1^10 = 1 (still flows!)")
    
    # Example 6: When to Use Sigmoid
    print("\n\nExample 6: When to Use Sigmoid")
    print("-" * 70)
    
    print("Use Sigmoid for:")
    print("  1. Binary classification output layer")
    print("     Example: Spam/Not spam, Cat/Not cat")
    print("     Why: Outputs probability (0-1)")
    
    print("\n  2. Gating in LSTM/GRU")
    print("     Example: Decide what to remember/forget")
    print("     Why: Smooth 0-1 control")
    
    print("\n  3. When you need probabilities")
    print("     Example: Medical diagnosis (80% chance of disease)")
    print("     Why: Interpretable as probability")
    
    print("\nDon't use Sigmoid for:")
    print("  - Hidden layers (use ReLU instead)")
    print("  - Deep networks (vanishing gradient)")
    print("  - When you need speed (ReLU is faster)")
    
    # Example 7: Step-by-Step Calculation
    print("\n\nExample 7: Step-by-Step Calculation")
    print("-" * 70)
    
    example_input = 2.0
    print(f"Input: {example_input}")
    print("\nStep 1: Calculate e^(-z)")
    exp_neg = exp(-example_input)
    print(f"  e^(-{example_input}) = {exp_neg:.4f}")
    
    print("\nStep 2: Add 1")
    denominator = 1 + exp_neg
    print(f"  1 + {exp_neg:.4f} = {denominator:.4f}")
    
    print("\nStep 3: Divide 1 by denominator")
    result = 1 / denominator
    print(f"  1 / {denominator:.4f} = {result:.4f}")
    
    print(f"\nFinal: sigmoid({example_input}) = {result:.4f}")
    
    # Example 8: Real-World Analogy
    print("\n\nExample 8: Real-World Analogy")
    print("-" * 70)
    
    print("Think of sigmoid like a dimmer switch:")
    print("  - Very negative input: Light off (0)")
    print("  - Zero input: Light at 50% (0.5)")
    print("  - Very positive input: Light fully on (1)")
    print("  - Smooth transition between states")
    
    print("\nOr like a probability:")
    print("  - Input = -5: 0.7% chance (almost certainly no)")
    print("  - Input = 0: 50% chance (uncertain)")
    print("  - Input = 5: 99.3% chance (almost certainly yes)")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Sigmoid squashes any input to 0-1 range")
    print("2. S-shaped curve (smooth transition)")
    print("3. Used for binary classification (probabilities)")
    print("4. Has vanishing gradient problem (ReLU is better for hidden layers)")
    print("5. Still useful for output layers in binary classification")
    print("=" * 70)
    
    # Interactive test
    print("\n\nTest Your Understanding:")
    print("-" * 70)
    test_cases = [
        (0, "Zero input"),
        (5, "Large positive"),
        (-5, "Large negative"),
        (1, "Moderate positive"),
        (-1, "Moderate negative")
    ]
    
    for val, description in test_cases:
        result = sigmoid(val)
        interpretation = "uncertain" if 0.4 < result < 0.6 else ("likely" if result > 0.5 else "unlikely")
        print(f"{description}: sigmoid({val}) = {result:.4f} ({interpretation})")
    
    print("\n" + "=" * 70)
    print("Sigmoid Complete! Compare with ReLU and Softmax!")
    print("=" * 70)