def isAnagram(s, t):
    """
    Check if two strings are anagrams.
    
    Time Complexity: O(n)
    Space Complexity: O(1) - at most 26 characters
    
    An anagram means both strings have the same characters with the same frequencies.
    Example: "anagram" and "nagaram" are anagrams.
    """
    # Hint: Two strings are anagrams if they have the same character counts
    
    # Option 1: Use a dictionary to count characters
    # Count characters in s, then subtract counts from t
    # If all counts are 0, they're anagrams
    
    # Option 2: Sort both strings and compare
    # return sorted(s) == sorted(t)
    
    # Option 3: Use Counter from collections
    # from collections import Counter
    # return Counter(s) == Counter(t)
    
    # Step-by-step for Option 1:
    # 1. If lengths differ, return False
    # if len(s) != len(t):
    #     return False
    
    # 2. Create a dictionary to count characters
    # count = {}
    
    # 3. Count characters in s (add 1 for each)
    # for char in s:
    #     count[char] = count.get(char, 0) + 1
    
    # 4. Count characters in t (subtract 1 for each)
    # for char in t:
    #     count[char] = count.get(char, 0) - 1
    
    # 5. Check if all counts are 0
    # for value in count.values():
    #     if value != 0:
    #         return False
    # return True
    
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

