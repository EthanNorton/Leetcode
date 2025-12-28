"""
Exercise 2.1: Matrix Operations with NumPy
==========================================

✅ COMPLETED SOLUTION - Use as reference after attempting the template!

Goal: Master NumPy matrix operations and broadcasting.

Requirements:
1. Create matrices and vectors
2. Perform matrix multiplication
3. Use broadcasting
4. Avoid Python loops
5. Understand dimensions
"""

import numpy as np

def matrix_vector_multiplication():
    """
    Compute: y = X @ w + b
    
    Where:
    - X is a matrix (n_samples, n_features)
    - w is a weight vector (n_features,)
    - b is a bias scalar
    - y is the result (n_samples,)
    
    Use broadcasting to add bias to all samples.
    """
    # Create example data
    X = np.array([[1, 2], 
                  [3, 4], 
                  [5, 6]])  # (3, 2) - 3 samples, 2 features
    w = np.array([0.5, -1.0])  # (2,) - 2 weights
    b = 0.2  # scalar bias
    
    # Compute: y = X @ w + b
    # Matrix multiply, then broadcast bias
    y = X @ w + b
    
    return y


def vectorized_operations():
    """
    Replace loops with NumPy vectorized operations.
    
    Task: Given two arrays, compute element-wise operations.
    """
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    
    # All vectorized - no loops!
    sum_result = a + b           # Element-wise sum
    product_result = a * b        # Element-wise product
    power_result = a ** 2         # Square each element
    dot_product = a @ b           # Dot product (or np.dot(a, b))
    
    return {
        'sum': sum_result,
        'product': product_result,
        'power': power_result,
        'dot': dot_product
    }


def broadcasting_example():
    """
    Understand NumPy broadcasting rules.
    
    Broadcasting allows operations between arrays of different shapes.
    """
    # Matrix (3, 2)
    matrix = np.array([[1, 2],
                       [3, 4],
                       [5, 6]])
    
    # Vector (2,)
    vector = np.array([10, 20])
    
    # Broadcasting automatically expands vector to match matrix
    result = matrix + vector  # NumPy handles the expansion!
    
    return result


def matrix_operations():
    """
    Practice common matrix operations.
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    # Matrix operations
    matmul = A @ B                    # Matrix multiplication
    transpose = A.T                   # Transpose
    sum_cols = np.sum(A, axis=0)      # Sum along columns (axis=0)
    sum_rows = np.sum(A, axis=1)      # Sum along rows (axis=1)
    
    return {
        'matmul': matmul,
        'transpose': transpose,
        'sum_cols': sum_cols,
        'sum_rows': sum_rows
    }


# Tests
def test_matrix_vector_multiplication():
    """Test matrix-vector multiplication"""
    print("Testing matrix-vector multiplication...")
    y = matrix_vector_multiplication()
    
    X = np.array([[1, 2], [3, 4], [5, 6]])
    w = np.array([0.5, -1.0])
    b = 0.2
    expected = X @ w + b
    
    assert y is not None, "y should not be None"
    assert np.allclose(y, expected), f"Expected {expected}, got {y}"
    assert y.shape == (3,), f"Shape should be (3,), got {y.shape}"
    print("✓ Test passed!")
    print(f"  Result: {y}")
    print(f"  Expected: {expected}")


def test_vectorized_operations():
    """Test vectorized operations"""
    print("\nTesting vectorized operations...")
    results = vectorized_operations()
    
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    
    assert np.allclose(results['sum'], a + b), "Sum incorrect"
    assert np.allclose(results['product'], a * b), "Product incorrect"
    assert np.allclose(results['power'], a ** 2), "Power incorrect"
    assert results['dot'] == np.dot(a, b), "Dot product incorrect"
    print("✓ All vectorized operations correct!")


def test_broadcasting():
    """Test broadcasting"""
    print("\nTesting broadcasting...")
    result = broadcasting_example()
    
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    vector = np.array([10, 20])
    expected = matrix + vector
    
    assert result is not None, "Result should not be None"
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    assert result.shape == (3, 2), f"Shape should be (3, 2), got {result.shape}"
    print("✓ Broadcasting works correctly!")
    print(f"  Result:\n{result}")


if __name__ == "__main__":
    print("="*60)
    print("NumPy Matrix Operations Exercises")
    print("="*60)
    
    test_matrix_vector_multiplication()
    test_vectorized_operations()
    test_broadcasting()
    
    print("\n" + "="*60)
    print("Matrix Operations Practice:")
    print("="*60)
    results = matrix_operations()
    print(f"\nMatrix A:\n{np.array([[1, 2, 3], [4, 5, 6]])}")
    print(f"\nMatrix B:\n{np.array([[7, 8], [9, 10], [11, 12]])}")
    print(f"\nA @ B:\n{results['matmul']}")
    print(f"\nA.T:\n{results['transpose']}")
    print(f"\nSum along columns: {results['sum_cols']}")
    print(f"\nSum along rows: {results['sum_rows']}")
    
    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. Use @ for matrix multiplication")
    print("2. Broadcasting automatically handles dimension expansion")
    print("3. Avoid Python loops - NumPy is much faster!")
    print("4. Understand axis parameter (0=columns, 1=rows)")

