import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    # Step 1: Convert the list to a NumPy array
    np_array = np.array(a)
    # Step 2: Reshape the NumPy array 
    reshaped_array = np_array.reshape(new_shape)
    # Step 3: Convert the reshaped array back to a Python list
    reshaped_matrix = reshaped_array.tolist()
    # Step 4: Return the reshaped matrix 
    return reshaped_matrix 