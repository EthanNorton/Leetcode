"""
Exercise 1.1: Normalization Function - BLANK TEMPLATE
======================================================

Goal: Implement z-score normalization from scratch using clean Python.

Requirements:
1. Use list comprehensions (no explicit loops)
2. Handle edge cases (empty list, single value)
3. Add error handling
4. Write unit tests
5. Add docstrings

Formula: normalized = (x - mean) / std

INSTRUCTIONS:
- Fill in the TODO sections below
- Implement the normalize function
- Run the tests to verify your implementation
- Don't peek at the solution folder until you've tried!
"""

def normalize(xs):
    """
    Z-score normalization: (x - mean) / std
    
    Args:
        xs: List of numbers to normalize
        
    Returns:
        List of normalized values
        
    Raises:
        ValueError: If input is empty or invalid
        
    Example:
        >>> normalize([1, 2, 3, 4])
        [-1.3416407864998738, -0.4472135954999579, 0.4472135954999579, 1.3416407864998738]
    """
    # TODO: Implement this function
    # Hints:
    # 1. Check if xs is empty -> raise ValueError
    # 2. If len(xs) == 1, return [0.0]
    # 3. Calculate mean: sum(xs) / len(xs)
    # 4. Calculate variance: sum((x-mean)**2 for x in xs) / len(xs)
    # 5. Calculate std: variance**0.5
    # 6. Handle division by zero (if std == 0)
    # 7. Return list comprehension: [(x-mean)/std for x in xs]
    
    # Your implementation here
    pass


# Unit Tests
def test_normalize():
    """Test the normalize function"""
    print("Testing normalize()...")
    
    # Test 1: Basic case
    result = normalize([1, 2, 3, 4])
    expected_mean = sum(result) / len(result)
    expected_std = (sum((x - expected_mean)**2 for x in result) / len(result))**0.5
    assert abs(expected_mean) < 1e-10, "Mean should be ~0"
    assert abs(expected_std - 1.0) < 1e-10, "Std should be ~1"
    print("✓ Test 1 passed: Basic normalization")
    
    # Test 2: Edge case - single value
    result = normalize([5])
    assert result == [0.0], "Single value should normalize to 0"
    print("✓ Test 2 passed: Single value")
    
    # Test 3: Edge case - empty list
    try:
        normalize([])
        assert False, "Should raise ValueError"
    except ValueError:
        print("✓ Test 3 passed: Empty list raises error")
    
    # Test 4: Negative numbers
    result = normalize([-2, -1, 0, 1, 2])
    mean = sum(result) / len(result)
    assert abs(mean) < 1e-10, "Mean should be ~0"
    print("✓ Test 4 passed: Negative numbers")
    
    # Test 5: All same values
    result = normalize([5, 5, 5, 5])
    # Should handle division by zero (std = 0)
    print("✓ Test 5 passed: All same values")
    
    print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    # Run tests
    test_normalize()
    
    # Example usage
    print("\n" + "="*50)
    print("Example Usage:")
    print("="*50)
    
    data = [1, 2, 3, 4, 5]
    normalized = normalize(data)
    
    print(f"Original: {data}")
    print(f"Normalized: {[round(x, 4) for x in normalized]}")
    print(f"\nCheck: Mean ≈ 0, Std ≈ 1")
    print(f"Mean: {sum(normalized)/len(normalized):.6f}")
    print(f"Std: {(sum((x-sum(normalized)/len(normalized))**2 for x in normalized)/len(normalized))**0.5:.6f}")

