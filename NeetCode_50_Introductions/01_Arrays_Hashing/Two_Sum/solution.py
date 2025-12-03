def twoSum(nums, target):
    """
    Find two numbers that add up to target.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Your solution here
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    result = twoSum([2,7,11,15], 9)
    assert set(result) == {0, 1}
    print("Test 1 passed!")
    
    # Test 2
    result = twoSum([3,2,4], 6)
    assert set(result) == {1, 2}
    print("Test 2 passed!")
    
    # Test 3
    result = twoSum([3,3], 6)
    assert set(result) == {0, 1}
    print("Test 3 passed!")
    
    print("All tests passed!")

