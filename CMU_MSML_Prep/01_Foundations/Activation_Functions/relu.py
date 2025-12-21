"""
ReLU (Rectified Linear Unit) - The Easiest Deep ML Problem!

WHAT IS ReLU?
-------------
ReLU is the simplest activation function in neural networks.
It's literally just: "If the number is positive, keep it. If negative, make it zero."

Formula: f(x) = max(0, x)

Visual:
  Input:  -3  -2  -1   0   1   2   3
  Output:  0   0   0   0   1   2   3

WHY IS IT SO POPULAR?
---------------------
1. Simple: Just one line of code!
2. Fast: Very quick to compute
3. Effective: Works great in neural networks
4. Prevents "vanishing gradient" problem
5. Adds non-linearity (allows neural nets to learn complex patterns)

WHERE IS IT USED?
-----------------
- Every neural network layer
- Convolutional Neural Networks (CNNs)
- Deep learning models (GPT, ResNet, etc.)
- Almost all modern neural networks!
"""

def relu(z: float) -> float:
    """
    ReLU (Rectified Linear Unit) activation function.
    
    The simplest activation function: returns the input if positive, 0 if negative.
    
    Parameters:
    -----------
    z : float
        Input value (can be any number)
    
    Returns:
    --------
    float
        - z if z >= 0 (keep positive numbers)
        - 0 if z < 0  (zero out negative numbers)
    
    Formula: f(x) = max(0, x)
    
    Examples:
    --------
    >>> relu(5)
    5.0
    >>> relu(-3)
    0.0
    >>> relu(0)
    0.0
    """
    # That's it! Just one line:
    # If z is positive or zero, return z. Otherwise return 0.
    return max(0, z)


# ============================================================================
# COMPREHENSIVE EXAMPLES AND EXPLANATIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ReLU ACTIVATION FUNCTION - COMPLETE WALKTHROUGH")
    print("=" * 70)
    
    # Example 1: Basic understanding
    print("\nExample 1: Basic ReLU Behavior")
    print("-" * 70)
    
    test_values = [5, -3, 0, 2.5, -1.7, 10, -10]
    
    print("Input -> Output:")
    for val in test_values:
        result = relu(val)
        sign = "positive" if val >= 0 else "negative"
        print(f"  {val:6.1f} ({sign:8s}) -> {result:6.1f}")
    
    print("\nRule: Positive numbers stay the same, negative numbers become 0")
    
    # Example 2: Visual representation
    print("\n\nExample 2: Visual Graph")
    print("-" * 70)
    
    print("ReLU function graph:")
    print("     |")
    print("  4  |     *")
    print("  3  |    *")
    print("  2  |   *")
    print("  1  |  *")
    print("  0  |*--------*--------*--------*--------*")
    print("     -2  -1   0   1   2   3   4   5")
    print("     (negative side is flat at 0)")
    print("\nKey: Line goes up for positive x, stays at 0 for negative x")
    
    # Example 3: Why it's called "Rectified"
    print("\n\nExample 3: Why 'Rectified'?")
    print("-" * 70)
    
    print("'Rectified' means 'corrected' or 'made straight'")
    print("\nBefore ReLU: Output can be negative")
    print("  Input: -5 -> Output: -5 (negative)")
    print("\nAfter ReLU: Output is always >= 0")
    print("  Input: -5 -> Output: 0 (rectified to non-negative)")
    print("\nIt 'rectifies' negative values to zero!")
    
    # Example 4: In neural networks
    print("\n\nExample 4: How ReLU Works in Neural Networks")
    print("-" * 70)
    
    print("Neural network layer without ReLU:")
    print("  Input -> Linear transformation -> Output (can be any number)")
    print("  Example: 2.5 -> linear -> -1.3 (negative output)")
    
    print("\nNeural network layer WITH ReLU:")
    print("  Input -> Linear transformation -> ReLU -> Output (>= 0)")
    print("  Example: 2.5 -> linear -> -1.3 -> ReLU -> 0.0")
    print("  Example: 2.5 -> linear -> 3.7 -> ReLU -> 3.7")
    
    print("\nWhy this matters:")
    print("  - Allows network to 'turn off' neurons (output = 0)")
    print("  - Adds non-linearity (enables learning complex patterns)")
    print("  - Prevents vanishing gradients (keeps gradients flowing)")
    
    # Example 5: Step-by-step calculation
    print("\n\nExample 5: Step-by-Step Calculation")
    print("-" * 70)
    
    examples = [
        (5, "Positive number stays positive"),
        (-3, "Negative number becomes zero"),
        (0, "Zero stays zero"),
        (2.5, "Decimal positive stays the same"),
        (-1.7, "Decimal negative becomes zero")
    ]
    
    for input_val, explanation in examples:
        output = relu(input_val)
        print(f"\nInput: {input_val}")
        print(f"  Step 1: Check if {input_val} >= 0? {input_val >= 0}")
        if input_val >= 0:
            print(f"  Step 2: Since it's >= 0, return {input_val}")
        else:
            print(f"  Step 2: Since it's < 0, return 0")
        print(f"  Output: {output}")
        print(f"  Explanation: {explanation}")
    
    # Example 6: Comparison with other functions
    print("\n\nExample 6: ReLU vs Other Functions")
    print("-" * 70)
    
    test_val = 2.0
    print(f"For input = {test_val}:")
    print(f"  ReLU:     {relu(test_val)} (keeps positive values)")
    print(f"  Identity: {test_val} (no change)")
    print(f"  Sigmoid:  ~0.88 (squashes to 0-1 range)")
    print(f"  Tanh:     ~0.96 (squashes to -1 to 1 range)")
    
    test_val_neg = -2.0
    print(f"\nFor input = {test_val_neg}:")
    print(f"  ReLU:     {relu(test_val_neg)} (zeros out negative)")
    print(f"  Identity: {test_val_neg} (no change)")
    print(f"  Sigmoid:  ~0.12 (squashes to 0-1 range)")
    print(f"  Tanh:     ~-0.96 (squashes to -1 to 1 range)")
    
    print("\nReLU advantage: Simple, fast, and effective!")
    
    # Example 7: Real-world analogy
    print("\n\nExample 7: Real-World Analogy")
    print("-" * 70)
    
    print("Think of ReLU like a water valve:")
    print("  - Positive pressure (positive input) -> water flows (output = input)")
    print("  - Negative pressure (negative input) -> valve closes (output = 0)")
    print("  - No pressure (zero input) -> no flow (output = 0)")
    
    print("\nOr like a diode in electronics:")
    print("  - Current flows one way (positive) -> passes through")
    print("  - Current flows other way (negative) -> blocked (becomes 0)")
    
    # Example 8: Code implementation options
    print("\n\nExample 8: Different Ways to Implement ReLU")
    print("-" * 70)
    
    print("Method 1: Using max() (what we did)")
    print("  return max(0, z)")
    
    print("\nMethod 2: Using if-else")
    print("  if z > 0:")
    print("      return z")
    print("  else:")
    print("      return 0")
    
    print("\nMethod 3: Using ternary operator")
    print("  return z if z > 0 else 0")
    
    print("\nAll three are equivalent! max() is the most Pythonic.")
    
    # Final summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. ReLU is the SIMPLEST activation function")
    print("2. Formula: max(0, x) - that's it!")
    print("3. Positive numbers stay the same")
    print("4. Negative numbers become zero")
    print("5. Used in almost all modern neural networks")
    print("6. Adds non-linearity (enables learning complex patterns)")
    print("7. Prevents vanishing gradient problem")
    print("=" * 70)
    
    # Interactive test
    print("\n\nTry it yourself!")
    print("-" * 70)
    print("Test cases:")
    test_cases = [
        (5, 5),
        (-3, 0),
        (0, 0),
        (2.5, 2.5),
        (-1.7, 0),
        (100, 100),
        (-100, 0)
    ]
    
    all_passed = True
    for input_val, expected in test_cases:
        result = relu(input_val)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  [{status}] relu({input_val:6.1f}) = {result:6.1f} (expected {expected:6.1f})")
    
    if all_passed:
        print("\nAll tests passed! You understand ReLU!")
    else:
        print("\nSome tests failed. Check the implementation.")
    
    print("\n" + "=" * 70)
    print("Next steps: Try Leaky ReLU, Sigmoid, or Softmax!")
    print("=" * 70)
