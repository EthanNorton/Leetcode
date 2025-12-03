def search(nums, target):
    """
    Binary search for target in sorted array.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    # Your solution here
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    assert search([-1,0,3,5,9,12], 9) == 4
    print("Test 1 passed!")
    
    # Test 2
    assert search([-1,0,3,5,9,12], 2) == -1
    print("Test 2 passed!")
    
    print("All tests passed!")

