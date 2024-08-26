import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))

    # Reshape y to be a column vector if it is not already
    y = y.reshape(m, 1)
    
    for i in range(iterations): 
        h = X @ theta
        error = h - y 
        gradient = (X.T @ error) / m  # study gradient meaning a bit more 
        theta = theta - alpha * gradient  # Update theta

    # Round the coefficients to four decimal places
    theta_rounded = np.round(theta, 4)

    return theta_rounded.flatten()  # Flatten to 1D array for the output, optional due to pre-reshaped above 
