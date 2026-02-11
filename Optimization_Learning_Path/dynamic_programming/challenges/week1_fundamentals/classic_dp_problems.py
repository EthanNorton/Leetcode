"""
Lecture 3 Challenge: Classic DP Problem Patterns
Dynamic Programming Learning Path - MIT & CS50 Lectures

This module implements solutions to three classic DP problems that represent different patterns:
1. 1D DP: Coin Change Problem
2. 2D DP: Longest Common Subsequence  
3. Interval DP: Matrix Chain Multiplication

TODO: Complete all the methods below following the hints provided in the lecture notes.
"""

from typing import List, Tuple, Dict, Any
import sys

class ClassicDPProblems:
    """
    Implementation of classic DP problems representing different patterns.
    """
    
    def __init__(self):
        """Initialize the classic DP problems solver."""
        pass
    
    # Problem 1: Coin Change (1D DP)
    def coin_change_min_coins(self, coins: List[int], amount: int) -> int:
        """
        You are given coins of different denominations and a total amount.
        Find the minimum number of coins needed to make up that amount.
        Return -1 if it's impossible to make the amount.
        
        THINK ABOUT IT:
        - If amount = 0, we need 0 coins (base case)
        - For amount = 1, we try each coin and see if we can make it
        - For amount = 2, we try each coin and see if we can make (2-coin) + 1 coin
        - This builds up: dp[amount] = minimum coins needed for that amount
        
        EXAMPLE: coins = [1, 3, 4], amount = 6
        - dp[0] = 0 (0 coins for amount 0)
        - dp[1] = 1 (use coin 1)
        - dp[2] = 2 (use two coins of 1)
        - dp[3] = 1 (use coin 3)
        - dp[4] = 1 (use coin 4)
        - dp[5] = 2 (use coin 4 + coin 1)
        - dp[6] = 2 (use coin 3 + coin 3)
        
        STEPS TO IMPLEMENT:
        1. Handle edge case (amount = 0)
        2. Create dp array initialized to infinity
        3. Set dp[0] = 0 (base case)
        4. For each amount from 1 to target:
           - For each coin:
             - If coin <= amount: dp[amount] = min(dp[amount], dp[amount-coin] + 1)
        5. Return dp[amount] if it's not infinity, else -1
        """
        # TODO: Implement coin change solution
        pass
    
    def coin_change_ways(self, coins: List[int], amount: int) -> int:
        """
        Same setup, but find the number of ways to make the amount.
        
        THINK ABOUT IT DIFFERENTLY:
        - This is NOT the same as minimum coins!
        - We want to COUNT how many different ways we can make the amount
        - dp[i] = number of ways to make amount i
        
        EXAMPLE: coins = [1, 3, 4], amount = 6
        - dp[0] = 1 (one way: use no coins)
        - dp[1] = 1 (one way: use coin 1)
        - dp[2] = 1 (one way: use two coins of 1)
        - dp[3] = 2 (two ways: use coin 3 OR use three coins of 1)
        - dp[4] = 3 (three ways: use coin 4 OR use coin 3+coin 1 OR use four coins of 1)
        - dp[5] = 4 (four ways: use coin 4+coin 1 OR use coin 3+two coins of 1 OR ...)
        - dp[6] = 3 (three ways: use coin 3+coin 3 OR use coin 4+two coins of 1 OR use six coins of 1)
        
        KEY INSIGHT: For each coin, we add the number of ways to make (amount - coin)
        
        STEPS TO IMPLEMENT:
        1. Handle edge case (amount = 0)
        2. Create dp array initialized to 0
        3. Set dp[0] = 1 (base case: one way to make 0)
        4. For each coin:
           - For each amount from coin to target:
             - dp[amount] += dp[amount-coin]
        5. Return dp[amount]
        """
        # TODO: Implement number of ways solution
        pass
    
    # Problem 2: Longest Common Subsequence (2D DP)
    def longest_common_subsequence(self, text1: str, text2: str) -> int:
        """
        Given two strings, find the length of their longest common subsequence.
        A subsequence is a sequence that appears in the same relative order.
        
        THINK ABOUT IT:
        - This is a 2D DP problem because we have TWO strings to compare
        - dp[i][j] = LCS length of text1[0:i] and text2[0:j]
        - We need to consider all possible combinations of prefixes
        
        EXAMPLE: text1 = "abcde", text2 = "ace"
        - dp[0][0] = 0 (empty strings)
        - dp[1][1] = 1 (both have 'a')
        - dp[2][2] = 1 (both have 'a', but 'b' != 'c')
        - dp[3][2] = 2 (both have 'a' and 'c')
        - dp[4][3] = 2 (both have 'a' and 'c', but 'd' != 'e')
        - dp[5][3] = 3 (both have 'a', 'c', and 'e')
        
        RECURRENCE:
        - If characters match: dp[i][j] = dp[i-1][j-1] + 1
        - If characters don't match: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        STEPS TO IMPLEMENT:
        1. Get lengths of both strings
        2. Create 2D dp array (m+1) x (n+1)
        3. Initialize first row and column to 0
        4. Fill dp table:
           - If characters match: dp[i][j] = dp[i-1][j-1] + 1
           - Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        5. Return dp[m][n]
        """
        # TODO: Implement LCS solution
        pass
    
    def longest_common_subsequence_optimized(self, text1: str, text2: str) -> int:
        """
        Same problem, but optimize space to O(min(m,n)).
        
        SPACE OPTIMIZATION TRICK:
        - We only need the previous row to compute the current row
        - So instead of storing the entire 2D table, we use just 2 rows
        - We alternate between "previous" and "current" row
        
        EXAMPLE: text1 = "abc", text2 = "ac"
        - Instead of 4x3 table, we use 2 rows of length 3
        - Row 0: [0, 0, 0] (empty string)
        - Row 1: [0, 1, 1] (after processing 'a')
        - Row 2: [0, 1, 1] (after processing 'b')
        - Row 3: [0, 1, 2] (after processing 'c')
        
        STEPS TO IMPLEMENT:
        1. Ensure text1 is the shorter string
        2. Create two rows: prev and curr
        3. Initialize prev row to 0
        4. For each character in text1:
           - For each character in text2:
             - If characters match: curr[j] = prev[j-1] + 1
             - Else: curr[j] = max(prev[j], curr[j-1])
           - Swap prev and curr rows
        5. Return prev[n] (the last element of the final prev row)
        """
        # TODO: Implement space-optimized LCS solution
        pass
    
    # Problem 3: Matrix Chain Multiplication (Interval DP)
    def matrix_chain_multiplication(self, dimensions: List[int]) -> int:
        """
        Given dimensions of matrices, find the minimum number of scalar multiplications
        needed to compute the product of all matrices.
        
        THINK ABOUT IT:
        - This is INTERVAL DP - we work on intervals of matrices
        - dp[i][j] = minimum cost to multiply matrices from i to j
        - We try all possible ways to split the interval
        
        EXAMPLE: dimensions = [1, 2, 3, 4] (matrices: 1x2, 2x3, 3x4)
        - Matrix A: 1x2, Matrix B: 2x3, Matrix C: 3x4
        - We want to find: min cost of (A*B)*C vs A*(B*C)
        
        COST CALCULATION:
        - To multiply A (p×q) and B (q×r), cost = p × q × r
        - (A*B)*C: cost = (1×2×3) + (1×3×4) = 6 + 12 = 18
        - A*(B*C): cost = (2×3×4) + (1×2×4) = 24 + 8 = 32
        - So (A*B)*C is better with cost 18
        
        INTERVAL DP PATTERN:
        - For each interval length from 2 to n
        - For each starting position i
        - Try all split points k in the interval
        
        STEPS TO IMPLEMENT:
        1. Get number of matrices (n = len(dimensions) - 1)
        2. Create 2D dp table n x n
        3. Initialize diagonal to 0 (single matrix costs 0)
        4. For each interval length from 2 to n:
           - For each starting position i:
             - Calculate ending position j = i + length - 1
             - For each split point k from i to j-1:
               - Calculate cost = dimensions[i] * dimensions[k+1] * dimensions[j+1]
               - dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + cost)
        5. Return dp[0][n-1]
        """
        # TODO: Implement matrix chain multiplication solution
        pass
    
    def matrix_chain_parenthesization(self, dimensions: List[int]) -> Tuple[int, str]:
        """
        Same problem, but also return the optimal parenthesization.
        
        HINTS:
        - Use a separate table to track the optimal split points
        - parent[i][j] = optimal split point for matrices i to j
        - Use this to reconstruct the optimal parenthesization
        - Return both minimum cost and the parenthesization string
        """
        # TODO: Implement with parenthesization tracking
        # Step 1: Implement matrix chain multiplication with parent table
        # Step 2: Create helper function to reconstruct parenthesization
        # Step 3: Use recursive approach to build the string
        # Step 4: Return (min_cost, parenthesization_string)
        pass
    
    def _reconstruct_parenthesization(self, parent: List[List[int]], i: int, j: int) -> str:
        """
        Helper function to reconstruct the optimal parenthesization.
        
        HINTS:
        - If i == j: return "A" + str(i)
        - Else: return "(" + left_part + right_part + ")"
        - Use parent[i][j] to find the optimal split point
        """
        # TODO: Implement parenthesization reconstruction
        # Step 1: Handle base case (single matrix)
        # Step 2: Get optimal split point from parent table
        # Step 3: Recursively construct left and right parts
        # Step 4: Combine with parentheses
        pass
    
    def analyze_patterns(self) -> str:
        """
        Analyze the patterns in these classic DP problems.
        
        HINTS:
        - Identify what makes each problem 1D, 2D, or interval DP
        - Compare the recurrence relations
        - Think about when you'd use each pattern
        - Consider space optimization opportunities
        """
        # TODO: Write pattern analysis
        analysis = """
        CLASSIC DP PATTERN ANALYSIS
        ===========================
        
        1D DP (Coin Change):
        - TODO: Describe what makes this 1D DP
        - TODO: Explain the recurrence relation
        - TODO: Discuss when to use 1D DP
        
        2D DP (Longest Common Subsequence):
        - TODO: Describe what makes this 2D DP
        - TODO: Explain the recurrence relation
        - TODO: Discuss when to use 2D DP
        
        Interval DP (Matrix Chain Multiplication):
        - TODO: Describe what makes this interval DP
        - TODO: Explain the recurrence relation
        - TODO: Discuss when to use interval DP
        
        Space Optimization:
        - TODO: Discuss opportunities for space optimization
        - TODO: Compare time vs space trade-offs
        
        Pattern Recognition:
        - TODO: How to identify which pattern applies to new problems
        - TODO: Common characteristics of each pattern
        """
        return analysis
    
    def run_tests(self):
        """Run all test cases to verify correctness."""
        print("Running Test Cases...")
        print("=" * 30)
        
        # Test coin change
        print("Testing Coin Change:")
        # Test minimum coins
        assert self.coin_change_min_coins([1, 3, 4], 6) == 2, "Should be 2 coins (3+3)"
        assert self.coin_change_min_coins([2], 3) == -1, "Should be impossible"
        assert self.coin_change_min_coins([1], 0) == 0, "Should be 0 coins for amount 0"
        print("  ✓ Minimum coins tests passed")
        
        # Test number of ways
        assert self.coin_change_ways([1, 3, 4], 6) == 3, "Should be 3 ways"
        assert self.coin_change_ways([2], 3) == 0, "Should be 0 ways"
        assert self.coin_change_ways([1], 0) == 1, "Should be 1 way for amount 0"
        print("  ✓ Number of ways tests passed")
        
        # Test LCS
        print("\nTesting Longest Common Subsequence:")
        assert self.longest_common_subsequence("abcde", "ace") == 3, "Should be 3"
        assert self.longest_common_subsequence("abc", "def") == 0, "Should be 0"
        assert self.longest_common_subsequence("", "") == 0, "Should be 0"
        print("  ✓ LCS tests passed")
        
        # Test LCS optimized
        assert self.longest_common_subsequence_optimized("abcde", "ace") == 3, "Should be 3"
        assert self.longest_common_subsequence_optimized("abc", "def") == 0, "Should be 0"
        print("  ✓ LCS optimized tests passed")
        
        # Test matrix chain
        print("\nTesting Matrix Chain Multiplication:")
        assert self.matrix_chain_multiplication([1, 2, 3, 4]) == 18, "Should be 18"
        assert self.matrix_chain_multiplication([1, 2]) == 0, "Should be 0"
        print("  ✓ Matrix chain tests passed")
        
        print("\nAll tests passed! ✅")
    
    def demonstrate_solutions(self):
        """Demonstrate solutions to all problems."""
        print("Demonstrating Classic DP Solutions")
        print("=" * 40)
        
        # Coin Change Demo
        print("1. Coin Change Problem:")
        coins = [1, 3, 4]
        amount = 6
        min_coins = self.coin_change_min_coins(coins, amount)
        num_ways = self.coin_change_ways(coins, amount)
        print(f"   Coins: {coins}, Amount: {amount}")
        print(f"   Minimum coins: {min_coins}")
        print(f"   Number of ways: {num_ways}")
        
        # LCS Demo
        print("\n2. Longest Common Subsequence:")
        text1 = "abcde"
        text2 = "ace"
        lcs_length = self.longest_common_subsequence(text1, text2)
        lcs_opt_length = self.longest_common_subsequence_optimized(text1, text2)
        print(f"   Text1: '{text1}', Text2: '{text2}'")
        print(f"   LCS length: {lcs_length}")
        print(f"   LCS length (optimized): {lcs_opt_length}")
        
        # Matrix Chain Demo
        print("\n3. Matrix Chain Multiplication:")
        dimensions = [1, 2, 3, 4]  # Matrices: 1x2, 2x3, 3x4
        min_cost = self.matrix_chain_multiplication(dimensions)
        min_cost_with_parent, parenthesization = self.matrix_chain_parenthesization(dimensions)
        print(f"   Dimensions: {dimensions}")
        print(f"   Minimum cost: {min_cost}")
        print(f"   Minimum cost (with parent): {min_cost_with_parent}")
        print(f"   Optimal parenthesization: {parenthesization}")


def main():
    """
    Main function to demonstrate the classic DP problems.
    """
    print("Classic DP Problems - Lecture 3 Challenge")
    print("=" * 50)
    
    # Initialize solver
    solver = ClassicDPProblems()
    
    # Run tests
    solver.run_tests()
    
    # Demonstrate solutions
    solver.demonstrate_solutions()
    
    # Analyze patterns
    print("\nPattern Analysis:")
    print("-" * 30)
    analysis = solver.analyze_patterns()
    print(analysis)
    
    print("\nChallenge completed! 🎉")
    print("You've successfully implemented classic DP patterns!")


if __name__ == "__main__":
    main()
