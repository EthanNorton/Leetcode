# come back to this, a little bit more complex as it handles dynammic programming 

class Solution:
    def climbStairs(self, n: int) -> int:
        ways(n) = ways(n-1) + ways(n-2)