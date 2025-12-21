"""
Matrix Times Vector Multiplication - Complete Walkthrough

WHAT IS MATRIX-VECTOR MULTIPLICATION?
-------------------------------------
Matrix-vector multiplication is THE fundamental operation in machine learning.
It's how we apply linear transformations to data.

Formula: Matrix A (m×n) × Vector b (n×1) = Vector c (m×1)

Visual Example:
    [1  2]     [3]     [1×3 + 2×4]     [11]
    [4  5]  ×  [4]  =  [4×3 + 5×4]  =  [32]
    [7  8]              [7×3 + 8×4]     [53]

Key Rule: Columns in matrix = Rows in vector (must match!)

WHY IS IT SO IMPORTANT?
-----------------------
- Neural networks: Every layer does matrix-vector multiplication
- Linear regression: Predictions = X @ theta (matrix times vector)
- Transformations: Rotating, scaling, projecting data
- Feature extraction: Combining features linearly
"""

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    """
    Multiply a matrix by a vector.
    
    Parameters:
    -----------
    a : list of lists
        Matrix with shape (m, n) - m rows, n columns
    b : list
        Vector with n elements
    
    Returns:
    --------
    c : list
        Result vector with m elements
        Returns -1 if dimensions don't match
    
    Example:
    --------
    >>> a = [[1, 2], [3, 4]]
    >>> b = [5, 6]
    >>> matrix_dot_vector(a, b)
    [17, 39]  # [1×5+2×6, 3×5+4×6]
    """
    # Step 1: Verify the dimensions
    # Matrix columns must equal vector length
    # Example: Matrix 3×2 needs vector of length 2
    if len(a[0]) != len(b):
        return -1  # Dimension mismatch!
    
    # Step 2: Compute the dot product for each row
    # For each row in the matrix, multiply with the vector
    c = []
    for row in a:
        # Dot product: sum of (row[i] * vector[i]) for all i
        # Example: row = [1, 2], b = [3, 4]
        #          dot_product = 1×3 + 2×4 = 3 + 8 = 11
        dot_product = sum(row[i] * b[i] for i in range(len(b)))
        c.append(dot_product)
    
    # Step 3: Return the result
    return c


# ============================================================================
# EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MATRIX TIMES VECTOR - COMPLETE WALKTHROUGH")
    print("=" * 70)
    
    # Example 1: Basic 2×2 matrix × 2-element vector
    print("\nExample 1: Basic Multiplication")
    print("-" * 70)
    
    a1 = [[1, 2],
          [3, 4]]
    b1 = [5, 6]
    
    print("Matrix A (2×2):")
    for row in a1:
        print(f"  {row}")
    print(f"\nVector b: {b1}")
    
    result1 = matrix_dot_vector(a1, b1)
    print(f"\nResult: {result1}")
    
    print("\nStep-by-step calculation:")
    print("  Row 0: [1, 2] × [5, 6] = 1×5 + 2×6 = 5 + 12 = 17")
    print("  Row 1: [3, 4] × [5, 6] = 3×5 + 4×6 = 15 + 24 = 39")
    print(f"  Final: {result1}")
    
    # Example 2: 3×2 matrix × 2-element vector
    print("\n\nExample 2: 3×2 Matrix × 2-Element Vector")
    print("-" * 70)
    
    a2 = [[1, 2],
          [4, 5],
          [7, 8]]
    b2 = [3, 4]
    
    print("Matrix A (3×2):")
    for i, row in enumerate(a2):
        print(f"  Row {i}: {row}")
    print(f"\nVector b: {b2}")
    
    result2 = matrix_dot_vector(a2, b2)
    print(f"\nResult: {result2}")
    
    print("\nCalculation:")
    for i, row in enumerate(a2):
        calc = f"{row[0]}×{b2[0]} + {row[1]}×{b2[1]}"
        result = row[0]*b2[0] + row[1]*b2[1]
        print(f"  Row {i}: {calc} = {result}")
    
    # Example 3: Dimension mismatch
    print("\n\nExample 3: Dimension Mismatch (Error Case)")
    print("-" * 70)
    
    a3 = [[1, 2, 3],
          [4, 5, 6]]  # 2×3 matrix
    b3 = [7, 8]  # 2-element vector (needs 3!)
    
    print("Matrix A (2×3):")
    for row in a3:
        print(f"  {row}")
    print(f"\nVector b: {b3} (length {len(b3)})")
    print("\nProblem: Matrix has 3 columns, but vector has only 2 elements!")
    
    result3 = matrix_dot_vector(a3, b3)
    print(f"Result: {result3} (error code -1)")
    
    # Example 4: Real-world - Linear Regression Prediction
    print("\n\nExample 4: Linear Regression Prediction (Real-World Use)")
    print("-" * 70)
    
    # Features for 3 houses: [bias=1, size, bedrooms]
    X = [[1, 1000, 2],  # House 1: 1000 sqft, 2 bedrooms
         [1, 1500, 3],  # House 2: 1500 sqft, 3 bedrooms
         [1, 2000, 4]]  # House 3: 2000 sqft, 4 bedrooms
    
    # Learned weights: [intercept, price_per_sqft, price_per_bedroom]
    theta = [50, 0.1, 20]  # $50k base + $0.1k/sqft + $20k/bedroom
    
    print("Features (X):")
    print("  House 1: [bias=1, 1000 sqft, 2 bedrooms]")
    print("  House 2: [bias=1, 1500 sqft, 3 bedrooms]")
    print("  House 3: [bias=1, 2000 sqft, 4 bedrooms]")
    print(f"\nWeights (theta): {theta}")
    print("  [base_price, price_per_sqft, price_per_bedroom]")
    
    predictions = matrix_dot_vector(X, theta)
    print(f"\nPredictions: {predictions}")
    
    print("\nPrice calculations:")
    for i, (house, pred) in enumerate(zip(X, predictions), 1):
        calc = f"{theta[0]} + {house[1]}×{theta[1]} + {house[2]}×{theta[2]}"
        print(f"  House {i}: {calc} = ${pred:.1f}k")
    
    # Example 5: Neural Network Layer (Simplified)
    print("\n\nExample 5: Neural Network Layer (Simplified)")
    print("-" * 70)
    
    # Input features: [feature1, feature2]
    input_features = [0.5, 0.8]
    
    # Weight matrix for a layer: 3 neurons, 2 inputs each
    weights = [[0.1, 0.2],  # Neuron 1 weights
               [0.3, 0.4],  # Neuron 2 weights
               [0.5, 0.6]]  # Neuron 3 weights
    
    print("Input features:", input_features)
    print("\nWeight matrix (3 neurons × 2 inputs):")
    for i, row in enumerate(weights, 1):
        print(f"  Neuron {i}: {row}")
    
    neuron_outputs = matrix_dot_vector(weights, input_features)
    print(f"\nNeuron outputs: {neuron_outputs}")
    
    print("\nWhat happened:")
    print("  Each neuron computed: (weight1 × input1) + (weight2 × input2)")
    for i, (neuron_weights, output) in enumerate(zip(weights, neuron_outputs), 1):
        calc = f"{neuron_weights[0]}×{input_features[0]} + {neuron_weights[1]}×{input_features[1]}"
        print(f"  Neuron {i}: {calc} = {output}")
    
    # Example 6: Visual representation
    print("\n\nExample 6: Visual Matrix Multiplication")
    print("-" * 70)
    
    a6 = [[2, 3],
          [1, 4]]
    b6 = [5, 6]
    
    print("Visual representation:")
    print("  [2  3]     [5]")
    print("  [1  4]  ×  [6]")
    print()
    print("  = [2×5 + 3×6]")
    print("    [1×5 + 4×6]")
    print()
    print("  = [10 + 18]")
    print("    [5  + 24]")
    print()
    
    result6 = matrix_dot_vector(a6, b6)
    print(f"  = {result6}")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Matrix columns MUST equal vector length")
    print("2. Result has same number of rows as matrix")
    print("3. Each row of matrix multiplies with entire vector")
    print("4. Used in: Linear regression, neural networks, transformations")
    print("5. Foundation for ALL linear operations in ML")
    print("=" * 70)
    
    # Interactive test
    print("\n\nQuick Test:")
    print("-" * 70)
    test_a = [[1, 0], [0, 1]]  # Identity matrix
    test_b = [7, 8]
    test_result = matrix_dot_vector(test_a, test_b)
    print(f"Identity matrix × vector [7, 8] = {test_result}")
    print("(Identity matrix doesn't change the vector!)")

