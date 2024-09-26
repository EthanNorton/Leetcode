# working through this one 

class Solution:
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x 
        mid = (left + right) //2 
        if mid * mid == x