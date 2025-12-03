def isAnagram(s, t):
    """
    Check if two strings are anagrams.
    
    Time Complexity: O(n)
    Space Complexity: O(1) - at most 26 characters
    """
    # Your solution here
    pass

# Test cases
if __name__ == "__main__":
    # Test 1
    assert isAnagram("anagram", "nagaram") == True
    print("Test 1 passed!")
    
    # Test 2
    assert isAnagram("rat", "car") == False
    print("Test 2 passed!")
    
    print("All tests passed!")

