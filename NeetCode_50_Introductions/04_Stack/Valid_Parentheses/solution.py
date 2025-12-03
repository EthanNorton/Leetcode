def isValid(s):
    """
    Check if parentheses are valid.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Your solution here
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    assert isValid("()") == True
    print("Test 1 passed!")
    
    # Test 2
    assert isValid("()[]{}") == True
    print("Test 2 passed!")
    
    # Test 3
    assert isValid("(]") == False
    print("Test 3 passed!")
    
    print("All tests passed!")

