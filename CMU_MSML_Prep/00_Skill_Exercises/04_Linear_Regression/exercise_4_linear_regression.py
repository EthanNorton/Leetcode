"""
Exercise 4: Linear Regression
==============================

Goal: Implement linear regression using both closed-form solution and gradient descent.

Requirements:
1. Closed-form solution (Normal Equation)
2. Gradient descent implementation
3. Compare both methods
4. Handle edge cases (bias term, matrix dimensions)
"""

import numpy as np


def linear_regression_closed_form(X, y):
    """
    Linear Regression using Normal Equation (Closed-Form Solution).
    
    Formula: w = (X.T @ X)^(-1) @ X.T @ y
    
    This is the analytical solution - no iteration needed!
    
    Args:
        X: Feature matrix (n_samples, n_features)
           NOTE: X should already include a column of ones for bias term
        y: Target values (n_samples,)
        
    Returns:
        w: Learned weights (n_features,) including bias term
        
    Example:
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])  # First column is bias
        >>> y = np.array([3, 5, 7])  # y = 2*x + 1
        >>> w = linear_regression_closed_form(X, y)
        >>> w  # Should be approximately [1, 2] (bias=1, slope=2)
    """
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # SOLUTION: Closed-form solution using Normal Equation
    # Formula: w = (X.T @ X)^(-1) @ X.T @ y
    # Use pseudoinverse for numerical stability
    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    return w


def linear_regression_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Linear Regression using Gradient Descent.
    
    Iteratively updates weights to minimize MSE loss.
    
    Args:
        X: Feature matrix (n_samples, n_features)
           NOTE: X should already include a column of ones for bias term
        y: Target values (n_samples,)
        learning_rate: Step size (alpha)
        iterations: Number of gradient descent steps
        
    Returns:
        w: Learned weights (n_features,) including bias term
        loss_history: List of MSE loss at each iteration
        
    Example:
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> y = np.array([3, 5, 7])
        >>> w, losses = linear_regression_gradient_descent(X, y, learning_rate=0.1, iterations=100)
        >>> w  # Should be approximately [1, 2]
    """
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape y to column vector if needed
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Initialize weights to zeros
    m, n = X.shape
    w = np.zeros((n, 1))
    
    # Store loss history for visualization
    loss_history = []
    
    # Gradient descent loop
    for i in range(iterations):
        # Compute predictions
        h = X @ w
        
        # Compute error
        error = h - y
        
        # Compute gradient
        gradient = (X.T @ error) / m
        
        # Update weights
        w = w - learning_rate * gradient
        
        # Compute and store loss
        loss = np.mean(error**2)
        loss_history.append(loss)
    
    return w.flatten(), loss_history


def add_bias_term(X):
    """
    Add bias term (column of ones) to feature matrix.
    
    This is needed for both methods to learn the intercept.
    
    Args:
        X: Feature matrix without bias (n_samples, n_features)
        
    Returns:
        X_with_bias: Feature matrix with bias column prepended (n_samples, n_features+1)
        
    Example:
        >>> X = np.array([[1], [2], [3]])
        >>> add_bias_term(X)
        array([[1, 1],
               [1, 2],
               [1, 3]])
    """
    X = np.array(X)
    
    # Add column of ones for bias term
    m = X.shape[0]
    ones = np.ones((m, 1))
    X_with_bias = np.hstack([ones, X])
    
    return X_with_bias


# Tests
def test_closed_form():
    """Test closed-form solution"""
    print("Testing closed-form solution...")
    
    # Simple example: y = 2*x + 1
    X = np.array([[1, 1],  # [bias, x]
                  [1, 2],
                  [1, 3]])
    y = np.array([3, 5, 7])  # y = 2*x + 1
    
    w = linear_regression_closed_form(X, y)
    
    assert w is not None, "Weights should not be None"
    assert len(w) == 2, f"Should have 2 weights (bias + slope), got {len(w)}"
    assert np.allclose(w, [1, 2], atol=0.1), f"Expected approximately [1, 2], got {w}"
    print(f"✓ Closed-form solution correct! Weights: {w}")


def test_gradient_descent():
    """Test gradient descent solution"""
    print("\nTesting gradient descent...")
    
    # Simple example: y = 2*x + 1
    X = np.array([[1, 1],
                  [1, 2],
                  [1, 3]])
    y = np.array([3, 5, 7])
    
    w, losses = linear_regression_gradient_descent(X, y, learning_rate=0.1, iterations=1000)
    
    assert w is not None, "Weights should not be None"
    assert len(w) == 2, f"Should have 2 weights, got {len(w)}"
    assert len(losses) == 1000, f"Should have 1000 loss values, got {len(losses)}"
    assert np.allclose(w, [1, 2], atol=0.1), f"Expected approximately [1, 2], got {w}"
    assert losses[-1] < losses[0], "Loss should decrease over iterations"
    print(f"✓ Gradient descent correct! Final weights: {w}")
    print(f"  Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")


def test_add_bias_term():
    """Test bias term addition"""
    print("\nTesting add_bias_term...")
    
    X = np.array([[1], [2], [3]])
    X_with_bias = add_bias_term(X)
    
    assert X_with_bias.shape == (3, 2), f"Shape should be (3, 2), got {X_with_bias.shape}"
    assert np.allclose(X_with_bias[:, 0], [1, 1, 1]), "First column should be ones"
    assert np.allclose(X_with_bias[:, 1], [1, 2, 3]), "Second column should be original X"
    print("✓ Bias term addition correct!")


def compare_methods():
    """Compare closed-form vs gradient descent"""
    print("\n" + "="*60)
    print("Comparing Closed-Form vs Gradient Descent")
    print("="*60)
    
    # Generate simple data: y = 2*x + 1 + noise
    np.random.seed(42)
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
    y = np.array([3, 5, 7, 9, 11]) + np.random.normal(0, 0.1, 5)
    
    # Closed-form
    w_closed = linear_regression_closed_form(X, y)
    
    # Gradient descent
    w_gd, losses = linear_regression_gradient_descent(X, y, learning_rate=0.01, iterations=1000)
    
    print(f"\nClosed-form weights: {w_closed}")
    print(f"Gradient descent weights: {w_gd}")
    print(f"\nDifference: {np.abs(w_closed - w_gd)}")
    print(f"Final GD loss: {losses[-1]:.6f}")
    
    # They should be very close!
    assert np.allclose(w_closed, w_gd, atol=0.1), "Methods should give similar results"
    print("\n✓ Both methods give similar results!")


if __name__ == "__main__":
    print("="*60)
    print("Linear Regression Exercises")
    print("="*60)
    
    test_add_bias_term()
    test_closed_form()
    test_gradient_descent()
    compare_methods()
    
    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. Closed-form: Fast, exact solution (if X.T@X is invertible)")
    print("2. Gradient descent: Iterative, works for large datasets")
    print("3. Both need bias term (column of ones) to learn intercept")
    print("4. Gradient descent requires tuning learning rate")
    print("5. Loss should decrease over iterations in GD")

