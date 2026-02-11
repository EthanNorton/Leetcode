"""
Interactive Guide to Understanding Matrix Reshaping
Run this file to see reshaping in action with visual examples!
"""

import numpy as np

def reshape_matrix(a, new_shape):
    """Reshape a matrix to new dimensions"""
    np_array = np.array(a)
    reshaped_array = np_array.reshape(new_shape)
    return reshaped_array.tolist()


def print_matrix(matrix, label):
    """Pretty print a matrix"""
    print(f"{label}:")
    for row in matrix:
        print(f"  {row}")
    print()


def demonstrate_reshape():
    """Walk through reshaping with examples"""
    
    print("=" * 70)
    print("MATRIX RESHAPING - INTERACTIVE GUIDE")
    print("=" * 70)
    print()
    
    print("📚 What is Reshaping?")
    print("   Reshaping changes a matrix's dimensions (rows × columns)")
    print("   while keeping all elements in the same order!")
    print()
    print("🔑 Key Rule: original_rows × original_cols = new_rows × new_cols")
    print()
    
    print("=" * 70)
    print("EXAMPLE 1: Reshape 2×6 → 3×4")
    print("=" * 70)
    print()
    
    a1 = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    print_matrix(a1, "Original (2×6)")
    
    print("Reading order (row-major): 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12")
    print()
    
    result1 = reshape_matrix(a1, (3, 4))
    print_matrix(result1, "Reshaped (3×4)")
    
    print("✅ Check: 2×6 = 12 elements, 3×4 = 12 elements ✓")
    print()
    
    print("=" * 70)
    print("EXAMPLE 2: Reshape 3×4 → 2×6")
    print("=" * 70)
    print()
    
    a2 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    print_matrix(a2, "Original (3×4)")
    
    result2 = reshape_matrix(a2, (2, 6))
    print_matrix(result2, "Reshaped (2×6)")
    
    print("✅ Check: 3×4 = 12 elements, 2×6 = 12 elements ✓")
    print()
    
    print("=" * 70)
    print("EXAMPLE 3: Flattening (2×3 → 1×6)")
    print("=" * 70)
    print()
    
    a3 = [[1, 2, 3], [4, 5, 6]]
    print_matrix(a3, "Original (2×3)")
    
    result3 = reshape_matrix(a3, (1, 6))
    print_matrix(result3, "Flattened (1×6)")
    
    print("💡 This is commonly used to flatten images for neural networks!")
    print()
    
    print("=" * 70)
    print("EXAMPLE 4: Expanding (1×9 → 3×3)")
    print("=" * 70)
    print()
    
    a4 = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    print_matrix(a4, "Original (1×9)")
    
    result4 = reshape_matrix(a4, (3, 3))
    print_matrix(result4, "Expanded (3×3)")
    
    print("💡 This creates a square matrix from a flat vector!")
    print()
    
    print("=" * 70)
    print("WHY IS RESHAPING IMPORTANT IN ML?")
    print("=" * 70)
    print()
    print("1. 🖼️  Image Processing:")
    print("   - Images: 28×28 pixels → Flatten to 784×1 for neural networks")
    print("   - Batch processing: Reshape for batch operations")
    print()
    print("2. 🧠 Neural Networks:")
    print("   - CNNs output: (batch, height, width, channels)")
    print("   - Dense layers need: (batch, features)")
    print("   - Reshape to convert between formats!")
    print()
    print("3. 📊 Data Preprocessing:")
    print("   - Prepare data for different model architectures")
    print("   - Convert between row and column formats")
    print()
    
    print("=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print()
    print("✅ Row-Major Order: Elements are read row-by-row")
    print("   [1, 2, 3] → [4, 5, 6] reads as: 1, 2, 3, 4, 5, 6")
    print()
    print("✅ Element Count Must Match:")
    print("   Original: 2×3 = 6 elements")
    print("   New shape: 3×2 = 6 elements ✓")
    print("   New shape: 4×2 = 8 elements ✗ (ERROR!)")
    print()
    print("✅ Data Preservation: All elements stay the same, just reorganized")
    print()
    
    print("=" * 70)
    print("TRY IT YOURSELF:")
    print("=" * 70)
    print()
    print("Test 1: Reshape [[1, 2], [3, 4], [5, 6]] to (2, 3)")
    test1 = [[1, 2], [3, 4], [5, 6]]
    result_test1 = reshape_matrix(test1, (2, 3))
    print(f"  Result: {result_test1}")
    print()
    
    print("Test 2: Flatten [[1, 2, 3, 4], [5, 6, 7, 8]] to (1, 8)")
    test2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
    result_test2 = reshape_matrix(test2, (1, 8))
    print(f"  Result: {result_test2}")
    print()
    
    print("=" * 70)
    print("🎉 You now understand matrix reshaping!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_reshape()

