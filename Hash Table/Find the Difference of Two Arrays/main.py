from typing import List

class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        """
        Find the difference of two arrays.
        
        Args:
            nums1: First array of integers
            nums2: Second array of integers
            
        Returns:
            List containing two sublists:
            - First sublist: distinct integers in nums1 but not in nums2
            - Second sublist: distinct integers in nums2 but not in nums1
        """
        
        # TODO: Implement your solution here
        # Hint: Consider using sets for efficient membership testing
        
        # Step 1: Convert arrays to sets (this removes duplicates automatically)
        # set1 = set(nums1)
        # set2 = set(nums2)
        
        # Step 2: Find elements in nums1 but not in nums2
        # diff1 = set1 - set2  # or use set1.difference(set2)
        
        # Step 3: Find elements in nums2 but not in nums1  
        # diff2 = set2 - set1  # or use set2.difference(set1)
        
        # Step 4: Convert back to lists and return
        # return [list(diff1), list(diff2)]
        
        pass  # Remove this when you implement your solution

# Test cases to verify your solution
def test_solution():
    solution = Solution()
    
    # Test case 1: Example from problem
    nums1 = [1, 2, 3]
    nums2 = [2, 4, 6]
    result = solution.findDifference(nums1, nums2)
    print(f"Test 1 - nums1: {nums1}, nums2: {nums2}")
    print(f"Expected: [[1, 3], [4, 6]], Got: {result}")
    print()
    
    # Test case 2: Arrays with duplicates
    nums1 = [1, 2, 2, 3]
    nums2 = [2, 4, 4, 6]
    result = solution.findDifference(nums1, nums2)
    print(f"Test 2 - nums1: {nums1}, nums2: {nums2}")
    print(f"Expected: [[1, 3], [4, 6]], Got: {result}")
    print()
    
    # Test case 3: No common elements
    nums1 = [1, 2, 3]
    nums2 = [4, 5, 6]
    result = solution.findDifference(nums1, nums2)
    print(f"Test 3 - nums1: {nums1}, nums2: {nums2}")
    print(f"Expected: [[1, 2, 3], [4, 5, 6]], Got: {result}")
    print()
    
    # Test case 4: Identical arrays
    nums1 = [1, 2, 3]
    nums2 = [1, 2, 3]
    result = solution.findDifference(nums1, nums2)
    print(f"Test 4 - nums1: {nums1}, nums2: {nums2}")
    print(f"Expected: [[], []], Got: {result}")

# Uncomment the line below to run tests after implementing your solution
# test_solution()