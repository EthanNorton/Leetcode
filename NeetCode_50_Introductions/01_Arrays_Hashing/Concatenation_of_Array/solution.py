def getConcatenation(nums):
    """
    Concatenate array with itself.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example: [1,2,1] → [1,2,1,1,2,1]
    """
    # Hint: You need to create a new array that contains nums twice
    
    # Option 1: Use list concatenation with +
    # return nums + nums
    
    # Option 2: Use list multiplication
    # return nums * 2
    
    # Option 3: Build it manually
    # result = []
    # for num in nums:
    #     result.append(num)
    # for num in nums:
    #     result.append(num)
    # return result
    
    # Option 4: Use extend method
    # result = nums.copy()  # or list(nums)
    # result.extend(nums)
    # return result
    
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    result = getConcatenation([1,2,1])
    assert result == [1,2,1,1,2,1]
    print("Test 1 passed!")
    
    # Test 2
    result = getConcatenation([1,3,2,1])
    assert result == [1,3,2,1,1,3,2,1]
    print("Test 2 passed!")
    
    # Test 3
    result = getConcatenation([5])
    assert result == [5,5]
    print("Test 3 passed!")
    
    print("All tests passed!")

