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
# PROBLEM 9: Sigmoid Function (Easy)
# ============================================================================

def sigmoid(z):
    """
    Sigmoid activation: sigma(z) = 1 / (1 + exp(-z))

    Args:
        z: Input (scalar or array)

    Returns:
        Output in range (0, 1)

    Example:
        >>> sigmoid(0)
        0.5

    TODO: Implement sigmoid
    """
    pass


# ============================================================================
# PROBLEM 10: Softmax (Medium)
# ============================================================================

def softmax(z):
    """
    Softmax: softmax(z)_i = exp(z_i) / sum(exp(z_j))
    Use numerical stability: subtract max(z) before exp.

    Args:
        z: Logits vector (n,)

    Returns:
        Probability distribution (n,) summing to 1

    TODO: Implement numerically stable softmax
    """
    pass


# ============================================================================
# PROBLEM 11: Binary Cross-Entropy Loss (Medium)
# ============================================================================

def binary_cross_entropy_loss(y_true, y_pred):
    """
    BCE loss: -(1/m) * sum(y*log(p) + (1-y)*log(1-p))
    Clip y_pred to avoid log(0).

    Args:
        y_true: Labels {0, 1}, shape (m,)
        y_pred: Predicted probabilities, shape (m,)

    Returns:
        Scalar loss value

    TODO: Implement BCE loss with clipping
    """
    pass


# ============================================================================
# PROBLEM 12: Add Bias Column (Easy)
# ============================================================================

def add_bias_column(X):
    """
    Add column of ones as the first column for bias term.

    Args:
        X: Feature matrix (m, n)

    Returns:
        X_with_bias: (m, n+1) with first column all 1s

    Example:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> add_bias_column(X)
        array([[1, 1, 2], [1, 3, 4]])

    TODO: Implement add_bias_column
    """
    pass


# ============================================================================
# PROBLEM 13: Z-Score Standardization (Easy)
# ============================================================================

def standardize_features(X):
    """
    Z-score standardize each column: (x - mean) / std
    Handle std=0 by returning zeros for that column.

    Args:
        X: Feature matrix (m, n)

    Returns:
        X_std: Standardized matrix (m, n)
        mean: Per-column means (n,)
        std: Per-column stds (n,)

    TODO: Implement standardization
    """
    pass


# ============================================================================
# PROBLEM 14: Lasso Regression (L1) Gradient (Hard)
# ============================================================================

def lasso_gradient_step(X, y, w, learning_rate, lambda_l1):
    """
    One gradient descent step with L1 penalty.
    Gradient of L1: sign(w) (subgradient at 0 is 0).

    Args:
        X: Feature matrix (m, n)
        y: Targets (m,)
        w: Weights (n,)
        learning_rate: Step size
        lambda_l1: L1 regularization strength

    Returns:
        w_new: Updated weights (n,)
        loss: Current loss (MSE + L1 penalty)

    TODO: Implement Lasso gradient step
    """
    pass


# ============================================================================
# PROBLEM 15: Polynomial Features (Medium)
# ============================================================================

def polynomial_features(x, degree=2):
    """
    Create polynomial features: [1, x, x^2, ..., x^degree]

    Args:
        x: 1D array (n,)
        degree: Max polynomial degree

    Returns:
        X_poly: Matrix (n, degree+1)

    Example:
        >>> x = np.array([1, 2, 3])
        >>> polynomial_features(x, degree=2)
        array([[1, 1, 1], [1, 2, 4], [1, 3, 9]])

    TODO: Implement polynomial features
    """
    pass


# ============================================================================
# PROBLEM 16: Precision, Recall, F1 (Medium)
# ============================================================================

def precision_recall_f1(y_true, y_pred):
    """
    Compute precision, recall, and F1 for binary classification.
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    Args:
        y_true: Ground truth {0, 1}, shape (m,)
        y_pred: Predicted {0, 1}, shape (m,)

    Returns:
        precision, recall, f1: Scalars

    TODO: Implement metrics
    """
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
    # test_problem_9()
    # test_problem_10()
    # test_problem_11()
    # test_problem_12()
    # test_problem_13()
    # test_problem_14()
    # test_problem_15()
    # test_problem_16()
    
    print("\n" + "="*60)
    print("Tips:")
    print("="*60)
    print("1. Start with easier problems (1, 2, 5)")
    print("2. Use NumPy operations (avoid Python loops when possible)")
    print("3. Test each function independently")
    print("4. Check shapes of arrays before operations")
    print("5. Use np.allclose() to compare floating-point results")
