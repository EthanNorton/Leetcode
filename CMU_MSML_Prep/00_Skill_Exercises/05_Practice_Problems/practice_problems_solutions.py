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
# PROBLEM 9: Sigmoid Function (Easy)
# ============================================================================

def sigmoid(z):
    """sigma(z) = 1 / (1 + exp(-z))"""
    z = np.array(z)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # Clip for stability


# ============================================================================
# PROBLEM 10: Softmax (Medium)
# ============================================================================

def softmax(z):
    """Numerically stable softmax"""
    z = np.array(z)
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    return exp_z / exp_z.sum()


# ============================================================================
# PROBLEM 11: Binary Cross-Entropy Loss (Medium)
# ============================================================================

def binary_cross_entropy_loss(y_true, y_pred):
    """BCE with clipping to avoid log(0)"""
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    y_true = np.array(y_true)
    m = len(y_true)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m


# ============================================================================
# PROBLEM 12: Add Bias Column (Easy)
# ============================================================================

def add_bias_column(X):
    """Add column of ones as first column"""
    X = np.array(X)
    m = X.shape[0]
    return np.hstack([np.ones((m, 1)), X])


# ============================================================================
# PROBLEM 13: Z-Score Standardization (Easy)
# ============================================================================

def standardize_features(X):
    """Z-score standardize each column"""
    X = np.array(X, dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    X_std = (X - mean) / std
    return X_std, mean, std


# ============================================================================
# PROBLEM 14: Lasso Regression (L1) Gradient (Hard)
# ============================================================================

def lasso_gradient_step(X, y, w, learning_rate, lambda_l1):
    """One gradient descent step with L1 penalty"""
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    m = X.shape[0]
    error = (X @ w - y).flatten()
    mse_grad = (X.T @ error) / m
    l1_subgrad = np.sign(w)
    l1_subgrad[w == 0] = 0
    gradient = mse_grad.reshape(-1, 1) + lambda_l1 * l1_subgrad
    w_new = w - learning_rate * gradient
    loss = np.mean(error**2) + lambda_l1 * np.sum(np.abs(w))
    return w_new.flatten(), loss


# ============================================================================
# PROBLEM 15: Polynomial Features (Medium)
# ============================================================================

def polynomial_features(x, degree=2):
    """Create [1, x, x^2, ..., x^degree]"""
    x = np.array(x).reshape(-1, 1)
    cols = [x**d for d in range(degree + 1)]
    return np.hstack(cols)


# ============================================================================
# PROBLEM 16: Precision, Recall, F1 (Medium)
# ============================================================================

def precision_recall_f1(y_true, y_pred):
    """Compute precision, recall, F1 for binary classification"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


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


def test_problem_9():
    """Test sigmoid"""
    print("\nTesting Problem 9: Sigmoid...")
    assert np.isclose(sigmoid(0), 0.5), "sigmoid(0) should be 0.5"
    assert sigmoid(100) > 0.99, "sigmoid(100) should be ~1"
    print("✓ Problem 9 passed!")


def test_problem_10():
    """Test softmax"""
    print("\nTesting Problem 10: Softmax...")
    z = np.array([1, 2, 3])
    p = softmax(z)
    assert np.isclose(p.sum(), 1.0), "Probabilities should sum to 1"
    assert np.all(p >= 0), "Probabilities should be non-negative"
    print("✓ Problem 10 passed!")


def test_problem_11():
    """Test BCE loss"""
    print("\nTesting Problem 11: Binary Cross-Entropy...")
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8])
    loss = binary_cross_entropy_loss(y_true, y_pred)
    assert loss >= 0, "Loss should be non-negative"
    print("✓ Problem 11 passed!")


def test_problem_12():
    """Test add bias column"""
    print("\nTesting Problem 12: Add Bias Column...")
    X = np.array([[1, 2], [3, 4]])
    X_b = add_bias_column(X)
    assert X_b.shape[1] == X.shape[1] + 1, "Should add one column"
    assert np.all(X_b[:, 0] == 1), "First column should be ones"
    print("✓ Problem 12 passed!")


def test_problem_13():
    """Test standardization"""
    print("\nTesting Problem 13: Z-Score Standardization...")
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_std, mean, std = standardize_features(X)
    assert np.allclose(X_std.mean(axis=0), 0, atol=1e-10), "Mean should be ~0"
    assert np.allclose(X_std.std(axis=0), 1, atol=1e-10), "Std should be ~1"
    print("✓ Problem 13 passed!")


def test_problem_14():
    """Test Lasso gradient step"""
    print("\nTesting Problem 14: Lasso Gradient Step...")
    X = np.array([[1, 1], [1, 2]])
    y = np.array([2, 3])
    w = np.array([0.0, 0.0])
    w_new, loss = lasso_gradient_step(X, y, w, 0.1, 0.01)
    assert w_new.shape == w.shape, "Shape should not change"
    print("✓ Problem 14 passed!")


def test_problem_15():
    """Test polynomial features"""
    print("\nTesting Problem 15: Polynomial Features...")
    x = np.array([1, 2, 3])
    X_poly = polynomial_features(x, degree=2)
    assert X_poly.shape == (3, 3), "Should be (3, 3)"
    assert np.allclose(X_poly[:, 2], x**2), "Third column should be x^2"
    print("✓ Problem 15 passed!")


def test_problem_16():
    """Test precision, recall, F1"""
    print("\nTesting Problem 16: Precision, Recall, F1...")
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0])
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    assert 0 <= p <= 1 and 0 <= r <= 1 and 0 <= f1 <= 1
    print("✓ Problem 16 passed!")


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
    test_problem_9()
    test_problem_10()
    test_problem_11()
    test_problem_12()
    test_problem_13()
    test_problem_14()
    test_problem_15()
    test_problem_16()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
