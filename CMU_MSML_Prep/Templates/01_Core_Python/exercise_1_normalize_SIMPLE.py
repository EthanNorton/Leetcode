"""
Exercise 1.1: Normalization Function - SIMPLE VERSION
=====================================================

This is a simplified version with step-by-step guidance.
Follow along and fill in each step!

📝 TEMPLATE FILE - Fill in the TODO sections below!

What does normalization do?
- Takes numbers like [1, 2, 3, 4]
- Makes them have mean = 0 and standard deviation = 1
- Result: [-1.34, -0.45, 0.45, 1.34] (approximately)

Formula: normalized_value = (value - mean) / standard_deviation
"""

def normalize(xs):
    """
    Normalize a list of numbers.
    
    Step 1: Check if list is empty
    Step 2: Handle single value (special case)
    Step 3: Calculate the mean (average)
    Step 4: Calculate the variance
    Step 5: Calculate the standard deviation
    Step 6: Apply the formula to each number
    """
    
    # ============================================
    # STEP 1: Check if the list is empty
    # ============================================
    # If the list is empty, we can't calculate mean/std
    # So we should raise an error
    
    # TODO: Check if xs is empty
    # Hint: Empty list is "falsy" in Python
    # Write: if not xs:
    # Then: raise ValueError("Input list cannot be empty")
    
    # Your code here:
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    
    # ============================================
    # STEP 2: Handle single value
    # ============================================
    # If there's only one number, we can't really normalize it
    # Convention: return [0.0]
    
    # TODO: Check if list has only one value
    # Hint: Use len(xs) == 1
    # Then: return [0.0]
    
    # Your code here:
    if len(xs) == 1:
        return [0.0]
    
    
    # ============================================
    # STEP 3: Calculate the MEAN (average)
    # ============================================
    # Mean = sum of all numbers / how many numbers
    # Example: [1, 2, 3, 4] → mean = (1+2+3+4)/4 = 2.5
    
    # TODO: Calculate mean
    # Hint: sum(xs) gives the sum, len(xs) gives the count
    # Write: mean = sum(xs) / len(xs)
    
    # Your code here:
    import numpy as np
    xs_array = np.array(xs)
    mean = np.mean(xs_array)
    return ((xs_array - mean) / std).tolist()

    
    
    # ============================================
    # STEP 4: Calculate the VARIANCE
    # ============================================
    # Variance = average of (each number - mean) squared
    # Example: For [1, 2, 3, 4] with mean=2.5:
    #   (1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²
    #   = (-1.5)² + (-0.5)² + (0.5)² + (1.5)²
    #   = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
    #   variance = 5.0 / 4 = 1.25
    
    # TODO: Calculate variance
    # Hint: Use a list comprehension: [(x - mean) ** 2 for x in xs]
    # Then take the sum and divide by len(xs)
    # Write: variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    
    # Your code here:
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    
    
    # ============================================
    # STEP 5: Calculate the STANDARD DEVIATION
    # ============================================
    # Standard deviation = square root of variance
    # Example: std = sqrt(1.25) ≈ 1.118
    
    # TODO: Calculate standard deviation
    # Hint: Square root = raise to power 0.5, or use math.sqrt()
    # Write: std = variance ** 0.5
    
    # Your code here:
    std = np.std(xs_array, ddof=0)
    if std == 0:
        return np.zeros_like(xs_array).tolist()
    
    # ============================================
    # STEP 6: Handle division by zero
    # ============================================
    # If all numbers are the same, variance = 0, so std = 0
    # We can't divide by zero!
    # Solution: If std is 0, return list of zeros
    
    # TODO: Check if std is 0
    # Hint: if std == 0:
    # Then: return [0.0] * len(xs)
    
    # Your code here:
    if std == 0:
        return [0.0] * len(xs)
    
    
    # ============================================
    # STEP 7: Apply the normalization formula
    # ============================================
    # For each number: normalized = (number - mean) / std
    # Use a list comprehension!
    # Example: [(x - mean) / std for x in xs]
    
    # TODO: Apply formula to each number
    # Hint: List comprehension: [formula for x in xs]
    # Write: return [(x - mean) / std for x in xs]
    
    # Your code here:
    return [(x - mean) / std for x in xs]


# ============================================
# TESTING YOUR FUNCTION
# ============================================
# Uncomment these tests one at a time as you implement each step

if __name__ == "__main__":
    print("="*60)
    print("Testing normalize() function")
    print("="*60)
    
    # Test 1: Basic case
    print("\nTest 1: Basic normalization")
    print("Input: [1, 2, 3, 4]")
    try:
        result = normalize([1, 2, 3, 4])
        print(f"Result: {result}")
        
        # Check if mean is approximately 0
        mean_result = sum(result) / len(result)
        print(f"Mean of result: {mean_result:.6f} (should be ~0)")
        
        # Check if std is approximately 1
        variance_result = sum((x - mean_result) ** 2 for x in result) / len(result)
        std_result = variance_result ** 0.5
        print(f"Std of result: {std_result:.6f} (should be ~1)")
        
        if abs(mean_result) < 0.1 and abs(std_result - 1.0) < 0.1:
            print("✓ Test 1 PASSED!")
        else:
            print("✗ Test 1 FAILED - check your implementation")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("   Make sure you've implemented all steps!")
    
    # Test 2: Single value
    print("\nTest 2: Single value")
    print("Input: [5]")
    try:
        result = normalize([5])
        print(f"Result: {result}")
        if result == [0.0]:
            print("✓ Test 2 PASSED!")
        else:
            print(f"✗ Test 2 FAILED - expected [0.0], got {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Empty list
    print("\nTest 3: Empty list")
    print("Input: []")
    try:
        result = normalize([])
        print(f"✗ Test 3 FAILED - should have raised ValueError!")
    except ValueError:
        print("✓ Test 3 PASSED! (correctly raised ValueError)")
    except Exception as e:
        print(f"✗ Test 3 FAILED - raised {type(e).__name__} instead of ValueError")
    
    # Test 4: All same values
    print("\nTest 4: All same values")
    print("Input: [5, 5, 5, 5]")
    try:
        result = normalize([5, 5, 5, 5])
        print(f"Result: {result}")
        print("✓ Test 4 PASSED! (handled division by zero)")
    except ZeroDivisionError:
        print("✗ Test 4 FAILED - got division by zero error")
        print("   Hint: Check if std == 0 before dividing!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Done! If all tests passed, you're done! 🎉")
    print("="*60)

