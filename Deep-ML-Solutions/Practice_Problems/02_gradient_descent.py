"""
CMU Practice Problem: Gradient Descent Implementation
10-725: Optimization for Machine Learning
10-701/715: Introduction to Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, alpha=0.01, iterations=1000, verbose=False):
    """
    Implement gradient descent for linear regression.
    
    Args:
        X: Feature matrix (m×n) - should include bias column
        y: Target vector (m×1)
        alpha: Learning rate
        iterations: Number of iterations
        verbose: Print cost every 100 iterations
    
    Returns:
        theta: Learned parameters (n×1)
        cost_history: Cost at each iteration
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(m, 1)
    cost_history = []
    
    for i in range(iterations):
        # Hypothesis: h = X @ theta
        h = X @ theta
        
        # Cost: J = (1/(2m)) * sum((h - y)^2)
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Gradient: ∇J = (1/m) * X^T @ (h - y)
        gradient = (1/m) * X.T @ (h - y)
        
        # Update: θ = θ - α * ∇J
        theta = theta - alpha * gradient
        
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.6f}")
    
    return theta, cost_history


def feature_scaling(X, method='standardize'):
    """
    Scale features for better gradient descent performance.
    
    Args:
        X: Feature matrix
        method: 'standardize' (mean=0, std=1) or 'minmax' ([0,1])
    
    Returns:
        X_scaled: Scaled features
        params: Scaling parameters (for inverse transform)
    """
    if method == 'standardize':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_scaled = (X - mean) / (std + 1e-8)  # Add small epsilon
        params = {'mean': mean, 'std': std, 'method': 'standardize'}
    elif method == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        X_scaled = (X - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    else:
        raise ValueError("Method must be 'standardize' or 'minmax'")
    
    return X_scaled, params


# Test cases
if __name__ == "__main__":
    print("=" * 60)
    print("GRADIENT DESCENT - PRACTICE PROBLEM")
    print("=" * 60)
    print()
    
    # Generate synthetic data: y = 2*x + 1 + noise
    np.random.seed(42)
    m = 100
    X = np.random.randn(m, 1) * 10  # Features
    y = 2 * X.flatten() + 1 + np.random.randn(m) * 2  # Targets with noise
    
    # Add bias column (column of ones)
    X_b = np.c_[np.ones((m, 1)), X]
    
    print("Data Info:")
    print(f"  Number of examples: {m}")
    print(f"  Features shape: {X.shape}")
    print(f"  True relationship: y = 2*x + 1 + noise")
    print()
    
    # Feature scaling (important for gradient descent!)
    X_scaled, scaling_params = feature_scaling(X, method='standardize')
    X_scaled_b = np.c_[np.ones((m, 1)), X_scaled]
    
    # Run gradient descent
    print("Running Gradient Descent...")
    print("  Learning rate: 0.01")
    print("  Iterations: 1000")
    print()
    
    theta, cost_history = gradient_descent(
        X_scaled_b, y, alpha=0.01, iterations=1000, verbose=True
    )
    
    print()
    print("Results:")
    print(f"  Learned parameters: θ₀ = {theta[0,0]:.4f}, θ₁ = {theta[1,0]:.4f}")
    print(f"  True parameters:    θ₀ = 1.0, θ₁ = 2.0")
    print()
    print(f"  Final cost: {cost_history[-1]:.6f}")
    print(f"  Initial cost: {cost_history[0]:.6f}")
    print()
    
    # Note: Parameters are in scaled space, need to transform back
    # For interpretation: y = theta[0] + theta[1] * (x_scaled)
    # where x_scaled = (x - mean) / std
    
    print("=" * 60)
    print("✅ Gradient descent completed!")
    print("=" * 60)
    print()
    print("Note: Parameters are learned in scaled feature space.")
    print("For prediction, scale new inputs the same way.")

