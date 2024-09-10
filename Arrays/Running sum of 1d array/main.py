# most favorite so far 

class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        runningsum = []
        for j in range(len(nums)):
            if j == 0: 
                runningsum.append(nums[j])
            if j > 0:
                runningsum.append(runningsum[j-1]+nums[j])
        return runningsum
        