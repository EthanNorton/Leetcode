"""
Exercise 3.1: Mean Squared Error Loss - BLANK TEMPLATE
=======================================================

Goal: Implement MSE loss and its gradient from scratch.

Requirements:
1. Vectorized implementation (no loops)
2. Handle edge cases
3. Return scalar value
4. Understand gradient computation

INSTRUCTIONS:
- Fill in the TODO sections below
- Implement all functions
- Run the tests to verify your implementation
- Don't peek at the solution folder until you've tried!
"""

import numpy as np

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error: mean((y_true - y_pred)^2)
    
    Args:
        y_true: True target values (array-like)
        y_pred: Predicted values (array-like)
        
    Returns:
        Scalar MSE loss value
        
    Example:
        >>> y_true = [1, 2, 3]
        >>> y_pred = [1.1, 1.9, 3.2]
        >>> mse_loss(y_true, y_pred)
        0.013333...
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # TODO: Implement MSE
    # Formula: mean((y_true - y_pred)^2)
    # Hint: Use vectorized operations, no loops!
    # Steps:
    # 1. Compute difference: y_true - y_pred
    # 2. Square the difference: (y_true - y_pred) ** 2
    # 3. Take the mean: np.mean(...)
    
    # Your code here
    loss = None  # Replace with: np.mean((y_true - y_pred)**2)
    
    return loss


def mse_loss_gradient(y_true, y_pred, X):
    """
    Compute gradient of MSE with respect to weights.
    
    For linear regression: y_pred = X @ w
    Gradient: (2/m) * X.T @ (y_pred - y_true)
    
    Args:
        y_true: True target values (n_samples,)
        y_pred: Predicted values (n_samples,)
        X: Feature matrix (n_samples, n_features)
        
    Returns:
        Gradient vector (n_features,)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    X = np.array(X)
    
    # TODO: Compute gradient
    # Formula: (2/m) * X.T @ (y_pred - y_true)
    # Where m = number of samples
    # Steps:
    # 1. Compute error: y_pred - y_true
    # 2. Get number of samples: m = len(y_true) or y_true.shape[0]
    # 3. Compute: X.T @ error (matrix multiplication)
    # 4. Multiply by (2/m)
    
    # Your code here
    m = len(y_true)
    gradient = None  # Replace with: (2/m) * X.T @ (y_pred - y_true)
    
    return gradient


def mae_loss(y_true, y_pred):
    """
    Mean Absolute Error: mean(|y_true - y_pred|)
    
    Bonus: Implement MAE as well!
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # TODO: Implement MAE
    # Formula: mean(|y_true - y_pred|)
    # Steps:
    # 1. Compute absolute difference: np.abs(y_true - y_pred)
    # 2. Take the mean: np.mean(...)
    
    # Your code here
    loss = None  # Replace with: np.mean(np.abs(y_true - y_pred))
    
    return loss


# Tests (These will run after you implement the functions above)
def test_mse_loss():
    """Test MSE loss function"""
    print("Testing MSE loss...")
    
    # Test 1: Perfect predictions
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    loss = mse_loss(y_true, y_pred)
    assert abs(loss) < 1e-10, f"Perfect predictions should give loss ≈ 0, got {loss}"
    print("✓ Test 1 passed: Perfect predictions")
    
    # Test 2: Some error
    y_true = [1, 2, 3]
    y_pred = [1.1, 1.9, 3.2]
    loss = mse_loss(y_true, y_pred)
    expected = np.mean((np.array(y_true) - np.array(y_pred))**2)
    assert abs(loss - expected) < 1e-10, f"Expected {expected}, got {loss}"
    print(f"✓ Test 2 passed: Loss = {loss:.6f}")
    
    # Test 3: Single value
    loss = mse_loss([5], [6])
    assert abs(loss - 1.0) < 1e-10, f"Should be 1.0, got {loss}"
    print("✓ Test 3 passed: Single value")
    
    print("\nAll MSE tests passed! 🎉")


def test_mse_gradient():
    """Test MSE gradient computation"""
    print("\nTesting MSE gradient...")
    
    # Simple example: y = 2*x + 1
    X = np.array([[1, 1],  # [bias, x]
                  [1, 2],
                  [1, 3]])
    w_true = np.array([1, 2])  # [bias, weight]
    y_true = X @ w_true  # [3, 5, 7]
    
    # Wrong predictions (w = [0, 0])
    w_pred = np.array([0, 0])
    y_pred = X @ w_pred  # [0, 0, 0]
    
    gradient = mse_loss_gradient(y_true, y_pred, X)
    
    # Manual calculation
    m = len(y_true)
    expected_gradient = (2/m) * X.T @ (y_pred - y_true)
    
    assert gradient is not None, "Gradient should not be None"
    assert np.allclose(gradient, expected_gradient), \
        f"Expected {expected_gradient}, got {gradient}"
    assert gradient.shape == (2,), f"Shape should be (2,), got {gradient.shape}"
    print("✓ Gradient computation correct!")
    print(f"  Gradient: {gradient}")
    print(f"  Expected: {expected_gradient}")


def test_mae_loss():
    """Test MAE loss function"""
    print("\nTesting MAE loss...")
    
    y_true = [1, 2, 3]
    y_pred = [1.1, 1.9, 3.2]
    loss = mae_loss(y_true, y_pred)
    
    expected = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    assert abs(loss - expected) < 1e-10, f"Expected {expected}, got {loss}"
    print(f"✓ MAE loss = {loss:.6f}")


if __name__ == "__main__":
    print("="*60)
    print("Loss Functions and Gradients Exercises - TEMPLATE")
    print("="*60)
    print("\nFill in the TODO sections and run the tests!")
    print("Don't look at the solution folder until you've tried!\n")
    
    # Uncomment these as you implement each function:
    # test_mse_loss()
    # test_mse_gradient()
    # test_mae_loss()
    
    # Example usage (uncomment after implementing):
    # print("\n" + "="*60)
    # print("Example Usage:")
    # print("="*60)
    # 
    # y_true = np.array([10, 20, 30, 40, 50])
    # y_pred = np.array([12, 18, 32, 38, 52])
    # 
    # mse = mse_loss(y_true, y_pred)
    # mae = mae_loss(y_true, y_pred)
    # 
    # print(f"\nTrue values:  {y_true}")
    # print(f"Predictions:  {y_pred}")
    # print(f"\nMSE Loss: {mse:.4f}")
    # print(f"MAE Loss: {mae:.4f}")
    # print(f"\nRMSE (Root MSE): {np.sqrt(mse):.4f}")
    
    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. MSE penalizes large errors more (squared)")
    print("2. MAE is more robust to outliers")
    print("3. Gradients are computed using chain rule")
    print("4. Always use vectorized operations!")

