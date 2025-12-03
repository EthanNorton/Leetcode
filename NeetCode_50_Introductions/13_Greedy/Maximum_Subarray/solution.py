def maxSubArray(nums):
    """
    Find maximum sum of contiguous subarray (Kadane's algorithm).
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Your solution here
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    assert maxSubArray([-2,1,-3,4,-1,2,1,-5,4]) == 6
    print("Test 1 passed!")
    
    # Test 2
    assert maxSubArray([1]) == 1
    print("Test 2 passed!")
    
    print("All tests passed!")

