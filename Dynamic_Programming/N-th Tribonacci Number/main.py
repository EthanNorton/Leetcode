class Solution:
    def tribonacci(self, n: int) -> int:
        # Handle base cases
        if n == 0: return 0
        if n <= 2: return 1
        
        # Initialize first three numbers
        dp = [0, 1, 1]
        
        # Calculate next numbers using previous three
        for i in range(3, n + 1):
            dp.append(dp[i-1] + dp[i-2] + dp[i-3])
            
        return dp[n]