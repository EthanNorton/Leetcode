def isPalindrome(s):
    """
    Check if string is a palindrome (ignoring case and non-alphanumeric).
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Your solution here
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    assert isPalindrome("A man, a plan, a canal: Panama") == True
    print("Test 1 passed!")
    
    # Test 2
    assert isPalindrome("race a car") == False
    print("Test 2 passed!")
    
    # Test 3
    assert isPalindrome(" ") == True
    print("Test 3 passed!")
    
    print("All tests passed!")

