# didn't fully understand this one 

import numpy as np

def to_categorical(x, n_col=None):
    
    if n_col is None:
        n_col = np.max(x) + 1  
    
    
    one_hot_matrix = np.zeros((len(x), n_col)) 
    one_hot_matrix[np.arange(len(x)), x] = 1  
    return one_hot_matrix

	