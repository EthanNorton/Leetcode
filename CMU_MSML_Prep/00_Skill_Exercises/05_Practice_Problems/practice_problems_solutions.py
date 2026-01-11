"""
Practice Problems - SOLUTIONS
==============================

Complete solutions for all practice problems.
Only look at this after you've tried solving them yourself!
"""

# SOLUTION FILE - Open this side-by-side with the template to compare your answers!
# File: practice_problems_solutions.py

import numpy as np


# ============================================================================
# PROBLEM 1: Vector Normalization (Easy)
# ============================================================================

def normalize_vector(v):
    """
    Normalize a vector to unit length (L2 norm).
    
    Formula: v_normalized = v / ||v||
    where ||v|| = sqrt(sum(v^2))
    """
    v = np.array(v)
    norm = np.linalg.norm(v)  # ||v|| = sqrt(sum(v^2))
    
    # Handle zero vector
    if norm == 0:
        return v
    
    return v / norm


# ============================================================================
# PROBLEM 2: Matrix-Vector Dot Product (Easy)
# ============================================================================

def matrix_vector_dot(A, v):
    """
    Compute matrix-vector product: A @ v
    """
    A = np.array(A)
    v = np.array(v)
    
    # Using loops (as requested)
    m, n = A.shape
    result = np.zeros(m)
    
    for i in range(m):
        for j in range(n):
            result[i] += A[i, j] * v[j]
    
    return result
    
    # Alternative (vectorized, but we'll use loops as practice):
    # return np.dot(A, v)


# ============================================================================
# PROBLEM 3: Compute Loss and Gradient (Medium)
# ============================================================================

def compute_loss_and_gradient(X, y, w):
    """
    Compute MSE loss and gradient for linear regression.
    
    Loss: L = (1/m) * sum((X @ w - y)^2)
    Gradient: grad = (2/m) * X.T @ (X @ w - y)
    """
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    
    # Reshape for consistency
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    
    m = X.shape[0]
    
    # Predictions
    predictions = X @ w
    
    # Error
    error = predictions - y
    
    # Loss (MSE)
    loss = np.mean(error**2)
    
    # Gradient
    gradient = (2.0 / m) * (X.T @ error)
    
    return loss, gradient.flatten()


# ============================================================================
# PROBLEM 4: Simple Linear Regression (Medium)
# ============================================================================

def simple_linear_regression(x, y):
    """
    Fit a simple linear regression: y = w0 + w1 * x
    
    Closed-form solution:
    w1 = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
    w0 = y_mean - w1 * x_mean
    """
    x = np.array(x)
    y = np.array(y)
    
    # Compute means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Compute deviations
    x_dev = x - x_mean
    y_dev = y - y_mean
    
    # Compute slope (w1)
    numerator = np.sum(x_dev * y_dev)
    denominator = np.sum(x_dev**2)
    
    # Handle division by zero
    if denominator == 0:
        w1 = 0
    else:
        w1 = numerator / denominator
    
    # Compute intercept (w0)
    w0 = y_mean - w1 * x_mean
    
    return w0, w1


# ============================================================================
# PROBLEM 5: Predict Function (Easy)
# ============================================================================

def predict_linear_regression(X, w):
    """
    Make predictions using learned weights.
    
    Formula: predictions = X @ w
    """
    X = np.array(X)
    w = np.array(w)
    
    # Ensure dimensions match
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    
    predictions = X @ w
    
    # Flatten if needed
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    
    return predictions


# ============================================================================
# PROBLEM 6: Batch Gradient Descent Step (Medium)
# ============================================================================

def gradient_descent_step(X, y, w, learning_rate):
    """
    Perform one step of gradient descent.
    """
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    
    # Reshape for consistency
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    
    m = X.shape[0]
    
    # Compute predictions
    predictions = X @ w
    
    # Compute error
    error = predictions - y
    
    # Compute gradient
    gradient = (X.T @ error) / m
    
    # Update weights
    w_new = w - learning_rate * gradient
    
    # Compute loss
    loss = np.mean(error**2)
    
    return w_new.flatten(), loss


# ============================================================================
# PROBLEM 7: Check Convergence (Easy)
# ============================================================================

def check_convergence(loss_history, tolerance=1e-6, patience=10):
    """
    Check if gradient descent has converged.
    
    Convergence: Loss doesn't improve by more than tolerance for 'patience' iterations.
    """
    if len(loss_history) < patience + 1:
        return False, loss_history[-1] if loss_history else None
    
    # Check if loss improved significantly in last 'patience' iterations
    recent_losses = loss_history[-patience:]
    
    # Compute improvement
    improvement = loss_history[-patience-1] - recent_losses[-1]
    
    # Check if improvement is below tolerance
    converged = improvement < tolerance
    
    return converged, loss_history[-1]


# ============================================================================
# PROBLEM 8: Ridge Regression (Hard)
# ============================================================================

def ridge_regression_closed_form(X, y, lambda_reg):
    """
    Ridge Regression (L2 regularization) using closed-form solution.
    
    Formula: w = (X.T @ X + lambda * I)^(-1) @ X.T @ y
    """
    X = np.array(X)
    y = np.array(y)
    
    m, n = X.shape
    
    # Create identity matrix
    I = np.eye(n)
    
    # Compute regularized matrix
    regularized_matrix = X.T @ X + lambda_reg * I
    
    # Solve for weights using pseudoinverse for numerical stability
    w = np.linalg.pinv(regularized_matrix) @ X.T @ y
    
    return w


# ============================================================================
# TESTS
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
    
    assert loss >= 0, "Loss should be non-negative"
    assert grad.shape == w.shape, "Gradient shape should match weights"
    print(f"✓ Problem 3 passed! Loss: {loss:.4f}, Gradient: {grad}")


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
    print(f"✓ Problem 6 passed! Loss: {loss:.4f}, New weights: {w_new}")


def test_problem_7():
    """Test convergence check"""
    print("\nTesting Problem 7: Convergence Check...")
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
    print("Practice Problems - SOLUTIONS")
    print("="*60)
    print("\nRunning all tests...\n")
    
    test_problem_1()
    test_problem_2()
    test_problem_3()
    test_problem_4()
    test_problem_5()
    test_problem_6()
    test_problem_7()
    test_problem_8()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
