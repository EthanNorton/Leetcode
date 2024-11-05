from typing import List

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # Initialize pointers for nums1, nums2, and the end of the merged array
        i = m - 1
        j = n - 1
        end = m + n - 1
        
        # Iterate from the end of nums1
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[end] = nums1[i]
                i -= 1
            else:
                nums1[end] = nums2[j]
                j -= 1
            end -= 1
        
        # If there are remaining elements in nums2, copy them
        while j >= 0:
            nums1[end] = nums2[j]
            j -= 1
            end -= 1