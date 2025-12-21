def containsDuplicate(nums):
    """
    Check if array contains duplicates.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example walkthrough with nums = [1, 2, 3, 1]:
    - Start: seen = set()  (empty)
    - num = 1: 1 not in seen → seen.add(1) → seen = {1}
    - num = 2: 2 not in seen → seen.add(2) → seen = {1, 2}
    - num = 3: 3 not in seen → seen.add(3) → seen = {1, 2, 3}
    - num = 1: 1 IS in seen → return True (duplicate found!)
    """
    # Hint: Use a set to track numbers we've seen
    # Create an empty set to store seen numbers
    # Syntax: set() creates an empty set, or you can use {}
    seen = set()  # Initially empty: set()
    
    # Loop through each number in nums
    # Syntax: for variable_name in list_name:
    for num in nums:
        # Check if we've seen this number before
        # Syntax: "in" checks if something exists in a set/list
        if num in seen:
            # We found a duplicate!
            return True
        else:
            # Add this number to our set so we remember we've seen it
            # Syntax: set_name.add(value) adds to a set
            # This line MODIFIES the seen set by adding num to it
            seen.add(num)  # After this, seen now contains num
    
    # If we finish the loop without finding duplicates, return False
    return False

# Test cases
if __name__ == "__main__":
    # Test 1
    assert containsDuplicate([1,2,3,1]) == True
    print("Test 1 passed!")
    
    # Test 2
    assert containsDuplicate([1,2,3,4]) == False
    print("Test 2 passed!")
    
    # Test 3
    assert containsDuplicate([1,1,1,3,3,4,3,2,4,2]) == True
    print("Test 3 passed!")
    
    print("All tests passed!")

