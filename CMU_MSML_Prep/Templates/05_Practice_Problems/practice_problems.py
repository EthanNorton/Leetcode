"""
Practice Problems - CMU MSML Prep
==================================

A collection of practice problems to reinforce concepts from Exercises 1-4.
Work through these problems to build confidence!

Difficulty: Easy → Medium → Hard
"""

# TEMPLATE FILE - Fill in the TODO sections below!
# Compare with: practice_problems_solutions.py when you're done

import numpy as np


# ============================================================================
# PROBLEM 1: Vector Normalization (Easy)
# ============================================================================

def normalize_vector(v):
    """
    Normalize a vector to unit length (L2 norm).
    
    Formula: v_normalized = v / ||v||
    where ||v|| = sqrt(sum(v^2))
    
    Args:
        v: Input vector (n,)
        
    Returns:
        v_normalized: Normalized vector with unit length
        
    Example:
        >>> v = np.array([3, 4])
        >>> normalize_vector(v)
        array([0.6, 0.8])  # Length = 1.0
        
    TODO: Implement this function
    """
    # Your code here
    pass


# ============================================================================
# PROBLEM 2: Matrix-Vector Dot Product (Easy)
# ============================================================================

def matrix_vector_dot(A, v):
    """
    Compute matrix-vector product: A @ v
    
    Args:
        A: Matrix (m, n)
        v: Vector (n,)
        
    Returns:
        result: Vector (m,)
        
    Example:
        >>> A = np.array([[1, 2], [3, 4]])
        >>> v = np.array([1, 2])
        >>> matrix_vector_dot(A, v)
        array([ 5, 11])  # [1*1 + 2*2, 3*1 + 4*2]
        
    TODO: Implement without using @ operator (use loops or np.dot)
    """
    # Your code here
    pass


# ============================================================================
# PROBLEM 3: Compute Loss and Gradient (Medium)
# ============================================================================

def compute_loss_and_gradient(X, y, w):
    """
    Compute MSE loss and gradient for linear regression.
    
    Loss: L = (1/m) * sum((X @ w - y)^2)
    Gradient: grad = (2/m) * X.T @ (X @ w - y)
    
    Args:
        X: Feature matrix (m, n) with bias column
        y: Target values (m,)
        w: Weights (n,)
        
    Returns:
        loss: Scalar MSE loss
        gradient: Gradient vector (n,)
        
    Example:
        >>> X = np.array([[1, 1], [1, 2]])
        >>> y = np.array([2, 3])
        >>> w = np.array([0, 1])
        >>> loss, grad = compute_loss_and_gradient(X, y, w)
        
    TODO: Implement loss and gradient computation
    """
    # Your code here
    pass


# ============================================================================
# PROBLEM 4: Simple Linear Regression (Medium)
# ============================================================================

def simple_linear_regression(x, y):
    """
    Fit a simple linear regression: y = w0 + w1 * x
    
    Use closed-form solution for simple case:
    w1 = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
    w0 = y_mean - w1 * x_mean
    
    Args:
        x: Single feature (n,)
        y: Target values (n,)
        
    Returns:
        w0: Intercept (bias)
        w1: Slope
        
    Example:
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([2, 4, 6])
        >>> w0, w1 = simple_linear_regression(x, y)
        >>> w0  # Should be ~0
        >>> w1  # Should be ~2
        
    TODO: Implement simple linear regression using the formulas above
    """
    # Your code here
    pass


# ============================================================================
# PROBLEM 5: Predict Function (Easy)
# ============================================================================

def predict_linear_regression(X, w):
    """
    Make predictions using learned weights.
    
    Args:
        X: Feature matrix (m, n) with bias column
        w: Learned weights (n,)
        
    Returns:
        predictions: Predicted values (m,)
        
    Example:
        >>> X = np.array([[1, 2], [1, 3]])
        >>> w = np.array([1, 2])
        >>> predict_linear_regression(X, w)
        array([5, 7])  # [1*1 + 2*2, 1*1 + 2*3]
        
    TODO: Implement prediction function
    """
    # Your code here
    pass


# ============================================================================
# PROBLEM 6: Batch Gradient Descent Step (Medium)
# ============================================================================

def gradient_descent_step(X, y, w, learning_rate):
    """
    Perform one step of gradient descent.
    
    Args:
        X: Feature matrix (m, n)
        y: Target values (m,)
        w: Current weights (n,)
        learning_rate: Step size
        
    Returns:
        w_new: Updated weights (n,)
        loss: Current loss value
        
    TODO: Implement one gradient descent step
    """
    # Your code here
    pass


# ============================================================================
# PROBLEM 7: Check Convergence (Easy)
# ============================================================================

def check_convergence(loss_history, tolerance=1e-6, patience=10):
    """
    Check if gradient descent has converged.
    
    Convergence: Loss doesn't improve by more than tolerance for 'patience' iterations.
    
    Args:
        loss_history: List of loss values
        tolerance: Minimum change to consider improvement
        patience: Number of iterations to wait
        
    Returns:
        converged: Boolean indicating convergence
        final_loss: Final loss value
        
    TODO: Implement convergence check
    """
    # Your code here
    pass


# ============================================================================
# PROBLEM 8: Ridge Regression (Hard)
# ============================================================================

def ridge_regression_closed_form(X, y, lambda_reg):
    """
    Ridge Regression (L2 regularization) using closed-form solution.
    
    Formula: w = (X.T @ X + lambda * I)^(-1) @ X.T @ y
    
    Args:
        X: Feature matrix (m, n) with bias
        y: Target values (m,)
        lambda_reg: Regularization strength
        
    Returns:
        w: Learned weights (n,)
        
    TODO: Implement ridge regression closed-form solution
    """
    # Your code here
    pass


# ============================================================================
# TESTS (Uncomment and run as you complete each problem)
# ============================================================================

def test_problem_1():
    """Test vector normalization"""
    print("Testing Problem 1: Vector Normalization...")
    v = np.array([3, 4])
    v_norm = normalize_vector(v)
    assert np.isclose(np.linalg.norm(v_norm), 1.0), "Vector should have unit length"
    assert np.allclose(v_norm, np.array([0.6, 0.8])), "Normalization incorrect"
    print("✓ Problem 1 passed!")


def test_problem_2():
    """Test matrix-vector dot product"""
    print("\nTesting Problem 2: Matrix-Vector Dot Product...")
    A = np.array([[1, 2], [3, 4]])
    v = np.array([1, 2])
    result = matrix_vector_dot(A, v)
    expected = A @ v
    assert np.allclose(result, expected), "Dot product incorrect"
    print("✓ Problem 2 passed!")


def test_problem_3():
    """Test loss and gradient computation"""
    print("\nTesting Problem 3: Loss and Gradient...")
    X = np.array([[1, 1], [1, 2]])
    y = np.array([2, 3])
    w = np.array([0, 1])
    loss, grad = compute_loss_and_gradient(X, y, w)
    
    # Check loss is non-negative
    assert loss >= 0, "Loss should be non-negative"
    # Check gradient shape
    assert grad.shape == w.shape, "Gradient shape should match weights"
    print(f"✓ Problem 3 passed! Loss: {loss:.4f}")


def test_problem_4():
    """Test simple linear regression"""
    print("\nTesting Problem 4: Simple Linear Regression...")
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    w0, w1 = simple_linear_regression(x, y)
    assert np.isclose(w0, 0, atol=0.1), "Intercept should be ~0"
    assert np.isclose(w1, 2, atol=0.1), "Slope should be ~2"
    print(f"✓ Problem 4 passed! w0={w0:.2f}, w1={w1:.2f}")


def test_problem_5():
    """Test prediction function"""
    print("\nTesting Problem 5: Predict Function...")
    X = np.array([[1, 2], [1, 3]])
    w = np.array([1, 2])
    predictions = predict_linear_regression(X, w)
    expected = np.array([5, 7])
    assert np.allclose(predictions, expected), "Predictions incorrect"
    print("✓ Problem 5 passed!")


def test_problem_6():
    """Test gradient descent step"""
    print("\nTesting Problem 6: Gradient Descent Step...")
    X = np.array([[1, 1], [1, 2]])
    y = np.array([2, 3])
    w = np.array([0.0, 0.0])
    w_new, loss = gradient_descent_step(X, y, w, learning_rate=0.1)
    
    assert w_new.shape == w.shape, "Weight shape should not change"
    assert loss >= 0, "Loss should be non-negative"
    print(f"✓ Problem 6 passed! Loss: {loss:.4f}")


def test_problem_7():
    """Test convergence check"""
    print("\nTesting Problem 7: Convergence Check...")
    # Loss that converges
    loss_history = [10, 9, 8, 7, 6.5, 6.1, 6.05, 6.02, 6.01, 6.001, 6.0001]
    converged, final_loss = check_convergence(loss_history, tolerance=0.01, patience=3)
    assert converged, "Should detect convergence"
    print("✓ Problem 7 passed!")


def test_problem_8():
    """Test ridge regression"""
    print("\nTesting Problem 8: Ridge Regression...")
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([3, 5, 7])
    lambda_reg = 0.1
    w = ridge_regression_closed_form(X, y, lambda_reg)
    
    assert w is not None, "Weights should not be None"
    assert len(w) == 2, "Should have 2 weights"
    print(f"✓ Problem 8 passed! Weights: {w}")


if __name__ == "__main__":
    print("="*60)
    print("CMU MSML Practice Problems")
    print("="*60)
    print("\nWork through these problems to reinforce your understanding!")
    print("Uncomment the tests as you complete each problem.\n")
    
    # Uncomment tests as you complete each problem:
    # test_problem_1()
    # test_problem_2()
    # test_problem_3()
    # test_problem_4()
    # test_problem_5()
    # test_problem_6()
    # test_problem_7()
    # test_problem_8()
    
    print("\n" + "="*60)
    print("Tips:")
    print("="*60)
    print("1. Start with easier problems (1, 2, 5)")
    print("2. Use NumPy operations (avoid Python loops when possible)")
    print("3. Test each function independently")
    print("4. Check shapes of arrays before operations")
    print("5. Use np.allclose() to compare floating-point results")
