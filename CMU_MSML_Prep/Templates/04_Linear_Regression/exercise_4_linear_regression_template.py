"""
Exercise 4: Linear Regression - BLANK TEMPLATE
==============================================

Goal: Implement linear regression using both closed-form solution and gradient descent.

Requirements:
1. Closed-form solution (Normal Equation)
2. Gradient descent implementation
3. Compare both methods
4. Handle edge cases (bias term, matrix dimensions)

INSTRUCTIONS:
- Fill in the TODO sections below
- Implement all functions
- Run the tests to verify your implementation
- Don't peek at the solution folder until you've tried!
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
    
    # TODO: Implement closed-form solution
    # Formula: w = (X.T @ X)^(-1) @ X.T @ y
    # Steps:
    # 1. Compute X.T @ X (matrix multiplication)
    # 2. Take inverse: np.linalg.inv(...) or np.linalg.pinv(...) for numerical stability
    # 3. Multiply by X.T @ y
    # Hint: Use np.linalg.pinv() instead of inv() for better numerical stability
    
    # Your code here
    w = None  # Replace with: np.linalg.pinv(X.T @ X) @ X.T @ y
    
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
    
    # TODO: Initialize weights
    # Start with zeros: w = np.zeros((n_features, 1))
    # Get dimensions: m, n = X.shape
    
    m, n = X.shape
    w = None  # Replace with: np.zeros((n, 1))
    
    # Store loss history for visualization
    loss_history = []
    
    # TODO: Gradient descent loop
    # For each iteration:
    #   1. Compute predictions: h = X @ w
    #   2. Compute error: error = h - y
    #   3. Compute gradient: gradient = (X.T @ error) / m
    #   4. Update weights: w = w - learning_rate * gradient
    #   5. Compute and store loss: mse = np.mean(error**2)
    
    for i in range(iterations):
        # Your code here
        # h = None  # Predictions: X @ w
        # error = None  # Error: h - y
        # gradient = None  # Gradient: (X.T @ error) / m
        # w = None  # Update: w - learning_rate * gradient
        # loss = None  # MSE loss: np.mean(error**2)
        # loss_history.append(loss)
        pass
    
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
    
    # TODO: Add column of ones
    # Steps:
    # 1. Get number of samples: m = X.shape[0]
    # 2. Create column of ones: np.ones((m, 1))
    # 3. Concatenate: np.hstack([ones, X]) or np.c_[ones, X]
    
    # Your code here
    m = X.shape[0]
    ones = None  # Replace with: np.ones((m, 1))
    X_with_bias = None  # Replace with: np.hstack([ones, X])
    
    return X_with_bias


# Tests (These will run after you implement the functions above)
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
    print("Linear Regression Exercises - TEMPLATE")
    print("="*60)
    print("\nFill in the TODO sections and run the tests!")
    print("Don't look at the solution folder until you've tried!\n")
    
    # Uncomment these as you implement each function:
    # test_add_bias_term()
    # test_closed_form()
    # test_gradient_descent()
    # compare_methods()
    
    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. Closed-form: Fast, exact solution (if X.T@X is invertible)")
    print("2. Gradient descent: Iterative, works for large datasets")
    print("3. Both need bias term (column of ones) to learn intercept")
    print("4. Gradient descent requires tuning learning rate")
    print("5. Loss should decrease over iterations in GD")

