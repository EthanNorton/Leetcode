"""
Exercise 2.1: Matrix Operations with NumPy - BLANK TEMPLATE
===========================================================

📝 TEMPLATE FILE - Fill in the TODO sections below!

Goal: Master NumPy matrix operations and broadcasting.

Requirements:
1. Create matrices and vectors
2. Perform matrix multiplication
3. Use broadcasting
4. Avoid Python loops
5. Understand dimensions

INSTRUCTIONS:
- Fill in the TODO sections below
- Implement all functions
- Run the tests to verify your implementation
- Don't peek at the solution folder until you've tried!
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
    
    # TODO: Compute y = X @ w + b
    # Hint: Use @ for matrix multiplication, broadcasting handles the +b
    
    # Your code here
    y = X @ w + b  # Replace with: X @ w + b
    
    return y


def vectorized_operations():
    """
    Replace loops with NumPy vectorized operations.
    
    Task: Given two arrays, compute element-wise operations.
    """
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    
    # TODO: Compute the following WITHOUT loops:
    # 1. Element-wise sum: a + b
    # 2. Element-wise product: a * b
    # 3. Element-wise power: a ** 2
    # 4. Dot product: a @ b (or np.dot(a, b))
    
    sum_result = a + b  # Your code
    product_result = a * b  # Your code
    power_result = a ** 2 # Your code
    dot_product = a @ b # Your code
    
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
    
    # TODO: Add vector to each row of matrix using broadcasting
    # Result should be (3, 2)
    # Hint: Broadcasting will automatically expand vector to match matrix
    
    result = matrix + vector # Your code: matrix + vector (broadcasting!)
    
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
    
    # TODO: Compute the following:
    # 1. A @ B (matrix multiplication)
    # 2. A.T (transpose)
    # 3. np.sum(A, axis=0) (sum along columns)
    # 4. np.sum(A, axis=1) (sum along rows)
    
    matmul = A @ B # A @ B
    transpose = A.T  # A.T
    sum_cols = np.sum(A, axis=0) # np.sum(A, axis=0)
    sum_rows = np.sum(A, axis=1) # np.sum(A, axis=1)
    
    return {
        'matmul': matmul,
        'transpose': transpose,
        'sum_cols': sum_cols,
        'sum_rows': sum_rows
    }


# Tests (These will run after you implement the functions above)
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
    print("NumPy Matrix Operations Exercises - TEMPLATE")
    print("="*60)
    print("\nFill in the TODO sections and run the tests!")
    print("Don't look at the solution folder until you've tried!\n")
    
    # Uncomment these as you implement each function:
    test_matrix_vector_multiplication()
    test_vectorized_operations()
    test_broadcasting()
    
    # Uncomment this after implementing matrix_operations():
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

