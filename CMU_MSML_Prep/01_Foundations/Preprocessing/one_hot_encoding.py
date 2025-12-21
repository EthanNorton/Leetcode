"""
One-Hot Encoding: Converting categorical data to numerical format

WHAT IS ONE-HOT ENCODING?
-------------------------
One-hot encoding converts categorical (nominal) values into binary vectors.
Each category becomes a binary column where only one position is "hot" (1) and rest are "cold" (0).

Example:
  Categories: ["red", "blue", "green"]
  "red"   -> [1, 0, 0]
  "blue"  -> [0, 1, 0]
  "green" -> [0, 0, 1]

WHY DO WE NEED IT?
------------------
- Machine learning algorithms need numbers, not text
- We can't use labels like 0, 1, 2 directly because that implies ordering (0 < 1 < 2)
- One-hot encoding treats each category as equally important, no ordering
"""

import numpy as np

def to_categorical(x, n_col=None):
    """
    Convert integer labels to one-hot encoded matrix.
    
    Parameters:
    -----------
    x : array-like
        Integer labels (e.g., [0, 1, 2, 1, 0])
    n_col : int, optional
        Number of columns (categories). If None, inferred from max value in x.
    
    Returns:
    --------
    one_hot_matrix : numpy array
        One-hot encoded matrix where each row has exactly one 1
    
    Example:
    --------
    >>> x = [0, 1, 2, 1, 0]
    >>> to_categorical(x)
    array([[1., 0., 0.],  # 0 -> [1, 0, 0]
           [0., 1., 0.],  # 1 -> [0, 1, 0]
           [0., 0., 1.],  # 2 -> [0, 0, 1]
           [0., 1., 0.],  # 1 -> [0, 1, 0]
           [1., 0., 0.]]) # 0 -> [1, 0, 0]
    """
    # Step 1: Determine number of categories (columns)
    # If not specified, find the maximum value and add 1
    # Example: if max is 2, we need columns 0, 1, 2 (so 3 columns)
    if n_col is None:
        n_col = np.max(x) + 1  # +1 because we include 0
    
    # Step 2: Create a matrix of zeros
    # Shape: (number of samples, number of categories)
    # Example: 5 samples, 3 categories -> (5, 3) matrix of zeros
    one_hot_matrix = np.zeros((len(x), n_col))
    
    # Step 3: Set the "hot" position to 1 for each sample
    # This is the tricky part! Let's break it down:
    # 
    # np.arange(len(x)) creates row indices: [0, 1, 2, 3, 4]
    # x contains column indices: [0, 1, 2, 1, 0]
    # 
    # one_hot_matrix[row, col] = 1 sets position (row, col) to 1
    # 
    # Example:
    #   Row 0, Col 0 -> one_hot_matrix[0, 0] = 1
    #   Row 1, Col 1 -> one_hot_matrix[1, 1] = 1
    #   Row 2, Col 2 -> one_hot_matrix[2, 2] = 1
    #   Row 3, Col 1 -> one_hot_matrix[3, 1] = 1
    #   Row 4, Col 0 -> one_hot_matrix[4, 0] = 1
    one_hot_matrix[np.arange(len(x)), x] = 1
    
    return one_hot_matrix


# ============================================================================
# EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ONE-HOT ENCODING - STEP BY STEP EXPLANATION")
    print("=" * 70)
    
    # Example 1: Simple case
    print("\nExample 1: Basic One-Hot Encoding")
    print("-" * 70)
    
    x1 = np.array([0, 1, 2, 1, 0])
    print(f"Input labels: {x1}")
    print("Meaning: Sample 0 is category 0, Sample 1 is category 1, etc.")
    
    result1 = to_categorical(x1)
    print(f"\nOne-hot encoded matrix:")
    print(result1)
    print("\nExplanation:")
    print("  Row 0: [1, 0, 0] -> Sample 0 belongs to category 0")
    print("  Row 1: [0, 1, 0] -> Sample 1 belongs to category 1")
    print("  Row 2: [0, 0, 1] -> Sample 2 belongs to category 2")
    print("  Row 3: [0, 1, 0] -> Sample 3 belongs to category 1")
    print("  Row 4: [1, 0, 0] -> Sample 4 belongs to category 0")
    
    # Example 2: Real-world example - Animal types
    print("\n\nExample 2: Animal Classification")
    print("-" * 70)
    
    # Let's say: 0=cat, 1=dog, 2=bird
    animals = np.array([0, 1, 2, 0, 1, 2, 0])  # cat, dog, bird, cat, dog, bird, cat
    animal_names = ["cat", "dog", "bird"]
    
    print("Animal labels:", animals)
    print("Mapping: 0=cat, 1=dog, 2=bird")
    
    result2 = to_categorical(animals)
    print(f"\nOne-hot encoded:")
    print(result2)
    
    print("\nInterpretation:")
    for i, animal_idx in enumerate(animals):
        one_hot = result2[i]
        animal_name = animal_names[animal_idx]
        print(f"  Sample {i}: {animal_name:4s} -> {one_hot} (category {animal_idx})")
    
    # Example 3: Step-by-step breakdown of the indexing trick
    print("\n\nExample 3: Understanding the Indexing Trick")
    print("-" * 70)
    
    x3 = np.array([2, 0, 1])
    print(f"Input: {x3}")
    print("\nStep-by-step:")
    print("1. Create zeros matrix: shape (3, 3)")
    print("   [[0, 0, 0],")
    print("    [0, 0, 0],")
    print("    [0, 0, 0]]")
    
    print("\n2. Row indices: np.arange(3) = [0, 1, 2]")
    print("   Column indices: x = [2, 0, 1]")
    
    print("\n3. Set positions:")
    print("   one_hot_matrix[0, 2] = 1  (row 0, col 2)")
    print("   one_hot_matrix[1, 0] = 1  (row 1, col 0)")
    print("   one_hot_matrix[2, 1] = 1  (row 2, col 1)")
    
    result3 = to_categorical(x3)
    print(f"\n4. Final result:")
    print(result3)
    
    # Example 4: Specifying number of columns explicitly
    print("\n\nExample 4: Specifying Number of Columns")
    print("-" * 70)
    
    x4 = np.array([0, 1, 0])
    print(f"Input: {x4}")
    print("Max value is 1, so normally we'd create 2 columns (0 and 1)")
    
    result4a = to_categorical(x4)
    print(f"\nAuto-detected (2 columns):")
    print(result4a)
    
    # But what if we want 4 columns? (maybe for future categories)
    result4b = to_categorical(x4, n_col=4)
    print(f"\nExplicit (4 columns):")
    print(result4b)
    print("Note: Categories 2 and 3 are unused but columns exist for them")
    
    # Example 5: Why one-hot encoding matters
    print("\n\nExample 5: Why Not Just Use 0, 1, 2 Directly?")
    print("-" * 70)
    
    print("Problem with using labels directly:")
    print("  If we use: cat=0, dog=1, bird=2")
    print("  The model might think: cat < dog < bird (ordering!)")
    print("  But categories have NO natural ordering!")
    
    print("\nOne-hot encoding solves this:")
    print("  cat  -> [1, 0, 0]")
    print("  dog  -> [0, 1, 0]")
    print("  bird -> [0, 0, 1]")
    print("  Each category is equally distant from others!")
    
    # Example 6: Visual representation
    print("\n\nExample 6: Visual Matrix Representation")
    print("-" * 70)
    
    x6 = np.array([0, 1, 2, 0, 2])
    result6 = to_categorical(x6)
    
    print("Input labels:", x6)
    print("\nOne-hot matrix (each row = one sample):")
    print("     Col0  Col1  Col2")
    for i, row in enumerate(result6):
        print(f"Row{i}: {row}  <- Sample {i} is category {x6[i]}")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. One-hot encoding converts categories to binary vectors")
    print("2. Each sample gets exactly one '1' in its row")
    print("3. Number of columns = number of unique categories")
    print("4. Used when categories have no natural ordering")
    print("5. Essential for neural networks and many ML algorithms")
    print("=" * 70)
