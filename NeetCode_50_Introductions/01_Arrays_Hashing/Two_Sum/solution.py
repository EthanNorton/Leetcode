def twoSum(nums, target):
    """
    Find two numbers that add up to target.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Key insight: For each number, calculate what its "partner" would be.
    If we've seen the partner before, we found our pair!
    
    Example with nums = [2,7,11,15], target = 9:
    - num = 2, index = 0: Need 9-2=7. Is 7 in map? No. Store {2: 0}
    - num = 7, index = 1: Need 9-7=2. Is 2 in map? YES! Return [0, 1]
    """
    # Hint: Use a dictionary to store {number: index}
    # This lets us quickly look up if we've seen a number and where
    
    # Create an empty dictionary
    # Syntax: {} creates an empty dict, or dict()
    seen = {}  # Will store {number: index}
    
    # Loop through nums with both value and index
    # Syntax: for index, num in enumerate(nums):
    #   enumerate gives you (index, value) pairs
    #   i = index, num = value at that index
    for i, num in enumerate(nums):
        # For each number:
        #   1. Calculate what number we need: complement = target - num
        complement = target - num
        
        #   2. Check if complement is in our dictionary
        #   Syntax: if key in dict_name:
        if complement in seen:
            #   3. If yes, return [index_of_complement, current_index]
            #   seen[complement] gives us the index where we saw complement
            return [seen[complement], i]
        else:
            #   4. If no, store current number and its index in dictionary
            #   Syntax: dict_name[key] = value
            seen[num] = i
    
    # If we get here (shouldn't happen per problem constraints)
    return []

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

