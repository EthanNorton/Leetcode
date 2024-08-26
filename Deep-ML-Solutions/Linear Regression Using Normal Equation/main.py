import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    
    # Step 1: Convert Lists (X and Y vectors) to NumPy Arrays 
    X = np.array(X)
    y = np.array(y)
    
    # Step 2: Add a Bias Term (Intercept)
    X_b = np.c_[np.ones((X.shape[0], 1)), X[:, 1:]]  # Adjusted to remove the first column duplication, duplicates first otherwise 

    # Step 3: Compute Theta Using the Pseudoinverse
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y  # Using np.linalg.pinv
    
    # Step 4: Round the Coefficients 
    theta_rounded = np.round(theta, 4)

    # Step 5: Convert Theta to a List and Return 
    return theta_rounded.tolist()
