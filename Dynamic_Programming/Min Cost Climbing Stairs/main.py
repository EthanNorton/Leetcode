class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # Handle edge cases
        if not cost:
            return 0
        if len(cost) == 1:
            return cost[0]
            
        # Initialize dp array with first two costs
        dp = [0] * len(cost)
        dp[0] = cost[0]
        dp[1] = cost[1]
        
        # For each step, calculate minimum cost to reach it
        for i in range(2, len(cost)):
            dp[i] = cost[i] + min(dp[i-1], dp[i-2])
            
        # Return minimum between last two steps to reach top
        return min(dp[-1], dp[-2])