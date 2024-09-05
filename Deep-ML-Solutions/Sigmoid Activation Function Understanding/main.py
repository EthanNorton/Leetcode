import math
from math import exp 
import numpy as np 

def sigmoid(z: float) -> float:
	z = 1 / (1 + exp(-z))
	z = np.round(z,4)
	return z