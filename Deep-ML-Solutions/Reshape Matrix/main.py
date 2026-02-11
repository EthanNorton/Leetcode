"""
RESHAPE MATRIX - COMPLETE WALKTHROUGH
=====================================

WHAT IS RESHAPING?
------------------
Reshaping changes the dimensions of a matrix while keeping all the same elements.
Think of it like rearranging items in a box - same items, different arrangement.

Key Rule: original_rows × original_cols = new_rows × new_cols
(Must have same total number of elements!)

Example:
  Original: 2×6 = 12 elements
  Reshaped: 3×4 = 12 elements ✓
  
  Original: 2×6 = 12 elements
  Reshaped: 3×5 = 15 elements ✗ (Can't do this!)

WHY IS IT IMPORTANT?
--------------------
1. Neural networks need specific input shapes
2. Image processing (28×28 images → 784×1 for fully connected layer)
3. Batch processing (multiple samples)
4. Data preprocessing

WHERE IS IT USED?
-----------------
- CNN to fully connected layer (flatten images)
- Batch processing in neural networks
- Data preprocessing pipelines
- Tensor operations in deep learning
"""

import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    """
    Reshape a matrix to new dimensions while preserving all elements.
    
    Parameters:
    -----------
    a : list of lists
        Original matrix (2D)
    new_shape : tuple (rows, cols)
        Desired new dimensions
    
    Returns:
    --------
    list of lists
        Reshaped matrix
    
    Key Rule: original_rows × original_cols = new_rows × new_cols
    
    Example:
    --------
    >>> a = [[1, 2, 3, 4], [5, 6, 7, 8]]
    >>> reshape_matrix(a, (4, 2))
    [[1, 2], [3, 4], [5, 6], [7, 8]]
    """
    # Step 1: Convert the list to a NumPy array
    # NumPy makes reshaping easy and efficient
    np_array = np.array(a)
    
    # Step 2: Reshape the NumPy array 
    # Elements are read row-by-row (row-major order)
    # Example: [[1,2,3], [4,5,6]] reads as: 1, 2, 3, 4, 5, 6
    reshaped_array = np_array.reshape(new_shape)
    
    # Step 3: Convert the reshaped array back to a Python list
    reshaped_matrix = reshaped_array.tolist()
    
    # Step 4: Return the reshaped matrix 
    return reshaped_matrix


# Test cases - Try running these to see how reshaping works!
if __name__ == "__main__":
    print("=" * 60)
    print("RESHAPE MATRIX - TEST CASES")
    print("=" * 60)
    print()
    
    # Test 1: 2×6 → 3×4
    print("Test 1: Reshape 2×6 to 3×4")
    a1 = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    result1 = reshape_matrix(a1, (3, 4))
    print(f"Original (2×6): {a1}")
    print(f"Reshaped (3×4): {result1}")
    expected1 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    assert result1 == expected1, f"Expected {expected1}, got {result1}"
    print("✅ Test 1 passed!")
    print()
    
    # Test 2: 3×4 → 2×6
    print("Test 2: Reshape 3×4 to 2×6")
    a2 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    result2 = reshape_matrix(a2, (2, 6))
    print(f"Original (3×4): {a2}")
    print(f"Reshaped (2×6): {result2}")
    expected2 = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    assert result2 == expected2, f"Expected {expected2}, got {result2}"
    print("✅ Test 2 passed!")
    print()
    
    # Test 3: Flatten (2×3 → 1×6)
    print("Test 3: Flatten 2×3 to 1×6")
    a3 = [[1, 2, 3], [4, 5, 6]]
    result3 = reshape_matrix(a3, (1, 6))
    print(f"Original (2×3): {a3}")
    print(f"Reshaped (1×6): {result3}")
    expected3 = [[1, 2, 3, 4, 5, 6]]
    assert result3 == expected3, f"Expected {expected3}, got {result3}"
    print("✅ Test 3 passed!")
    print()
    
    # Test 4: Expand (1×9 → 3×3)
    print("Test 4: Expand 1×9 to 3×3")
    a4 = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    result4 = reshape_matrix(a4, (3, 3))
    print(f"Original (1×9): {a4}")
    print(f"Reshaped (3×3): {result4}")
    expected4 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert result4 == expected4, f"Expected {expected4}, got {result4}"
    print("✅ Test 4 passed!")
    print()
    
    # Example 5: Why Reshaping Matters in Neural Networks
    print("\n\nExample 5: Neural Network Application")
    print("-" * 70)
    
    print("CNN to Fully Connected Layer:")
    print("  CNN output: 100 images, each 7×7 pixels = (100, 7, 7)")
    print("  Fully connected layer needs: (100, 49) - flattened!")
    print("  Solution: Reshape (100, 7, 7) → (100, 49)")
    
    print("\nStep-by-step:")
    print("  1. Original: 100 images × 7×7 = 100 × 49 elements")
    print("  2. Reshape to: 100 rows × 49 columns")
    print("  3. Each row = one flattened image")
    print("  4. Ready for fully connected layer!")
    
    # Example 6: Visual Understanding
    print("\n\nExample 6: Visual Understanding")
    print("-" * 70)
    
    print("Original (2×6):")
    print("  [1,  2,  3,  4,  5,  6]")
    print("  [7,  8,  9, 10, 11, 12]")
    print("\nReading order (row-major):")
    print("  1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12")
    
    print("\nReshaped to (3×4):")
    print("  [1,  2,  3,  4]")
    print("  [5,  6,  7,  8]")
    print("  [9, 10, 11, 12]")
    print("\nSame elements, different arrangement!")
    
    # Example 7: Common Use Cases
    print("\n\nExample 7: Common Use Cases")
    print("-" * 70)
    
    print("1. Flattening Images for Neural Networks:")
    print("   Image: 28×28 pixels = 784 elements")
    print("   Reshape: (28, 28) → (1, 784) or (784, 1)")
    print("   Why: Fully connected layers need 1D input")
    
    print("\n2. Batch Processing:")
    print("   Batch of 32 images, each 28×28")
    print("   Reshape: (32, 28, 28) → (32, 784)")
    print("   Why: Process entire batch at once")
    
    print("\n3. Preparing Data:")
    print("   Data: (1000, 10) features")
    print("   Need: (100, 10) batches")
    print("   Reshape: Split into batches")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Reshaping changes dimensions, keeps all elements")
    print("2. Key rule: original_rows × cols = new_rows × cols")
    print("3. Elements read row-by-row (row-major order)")
    print("4. Essential for neural networks (CNN → FC layers)")
    print("5. Used in batch processing and data preprocessing")
    print("=" * 70)
    
    print("\nAll tests passed!")
    print("Key insight: Reshaping preserves all elements, just reorganizes them!")
    print("Elements are read row-by-row (row-major order).") 