def runningSum(nums):
    """
    Calculate running sum of array.
    
    Time Complexity: O(n)
    Space Complexity: O(n) for output array
    
    Example: [1,2,3,4] → [1,3,6,10]
    - result[0] = 1
    - result[1] = 1 + 2 = 3
    - result[2] = 1 + 2 + 3 = 6
    - result[3] = 1 + 2 + 3 + 4 = 10
    """
    # Hint: Create a result array
    # The first element is just nums[0]
    # Each next element is: previous result + current num
    
    # Option 1: Build result array step by step
    result = []
    current_sum = 0
    for num in nums:
        current_sum += num
        result.append(current_sum)
    return result
    
    # Option 2: Modify nums in place (if allowed)
    # for i in range(1, len(nums)):
    #     nums[i] = nums[i] + nums[i-1]
    # return nums

# Test cases
if __name__ == "__main__":
    # Test 1
    result = runningSum([1,2,3,4])
    assert result == [1,3,6,10]
    print("Test 1 passed!")
    
    # Test 2
    result = runningSum([1,1,1,1,1])
    assert result == [1,2,3,4,5]
    print("Test 2 passed!")
    
    # Test 3
    result = runningSum([3,1,2,10,1])
    assert result == [3,4,6,16,17]
    print("Test 3 passed!")
    
    print("All tests passed!")

