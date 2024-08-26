def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Step 1: Verify the dimensions
    if len(a[0]) != len(b):
        return -1
    
    # Step 2: Compute the dot product
    c = []
    for row in a:
        dot_product = sum(row[i] * b[i] for i in range(len(b)))
        c.append(dot_product)
    
    # Step 3: Return the result
    return c
