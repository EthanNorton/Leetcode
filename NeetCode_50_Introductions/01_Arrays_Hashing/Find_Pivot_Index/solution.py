def pivotIndex(nums):
    """
    Find the pivot index where left sum equals right sum.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Example: [1,7,3,6,5,6]
    - Total sum = 28
    - At index 3: left_sum = 11, right_sum = 28 - 11 - 6 = 11 ✓
    """
    # Hint: Calculate total sum first
    # total_sum = sum(nums)
    
    # Then iterate through array
    # Keep track of left_sum as you go
    # right_sum = total_sum - left_sum - current_element
    # If left_sum == right_sum, return current index
    
    # Step 1: Calculate total sum
    # total_sum = sum(nums)
    
    # Step 2: Initialize left_sum to 0
    # left_sum = 0
    
    # Step 3: Loop through array
    # for i, num in enumerate(nums):
    #     right_sum = total_sum - left_sum - num
    #     if left_sum == right_sum:
    #         return i
    #     left_sum += num
    
    # Step 4: If no pivot found, return -1
    # return -1
    
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    result = pivotIndex([1,7,3,6,5,6])
    assert result == 3
    print("Test 1 passed!")
    
    # Test 2
    result = pivotIndex([1,2,3])
    assert result == -1
    print("Test 2 passed!")
    
    # Test 3
    result = pivotIndex([2,1,-1])
    assert result == 0
    print("Test 3 passed!")
    
    # Test 4
    result = pivotIndex([-1,-1,-1,-1,-1,0])
    assert result == 2
    print("Test 4 passed!")
    
    print("All tests passed!")

