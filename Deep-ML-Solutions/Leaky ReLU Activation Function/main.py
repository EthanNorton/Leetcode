def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	 # Apply the Leaky ReLU function
    if z >= 0:
        return z
    else:
        return alpha * z