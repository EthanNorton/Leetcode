# Lecture 3: Classic DP Problems

**Video**: [MIT - Classic DP Problems](https://www.youtube.com/watch?v=OQ5jsbhAv_M)

## 📝 Learning Objectives

After watching this lecture, you should understand:
- The three classic DP problem patterns: 1D, 2D, and interval DP
- How to identify DP patterns in new problems
- Common recurrence relations and their applications
- Problem-solving strategies for DP

## 🎯 Challenge: Implement Classic DP Problem Patterns

### Problem Statement

Implement solutions to three classic DP problems that represent different patterns:
1. **1D DP**: Coin Change Problem
2. **2D DP**: Longest Common Subsequence
3. **Interval DP**: Matrix Chain Multiplication

### Requirements

Create a `ClassicDPProblems` class with the following structure:

```python
class ClassicDPProblems:
    def __init__(self):
        pass
    
    # Problem 1: Coin Change (1D DP)
    def coin_change_min_coins(self, coins, amount):
        """
        You are given coins of different denominations and a total amount.
        Find the minimum number of coins needed to make up that amount.
        Return -1 if it's impossible to make the amount.
        
        HINTS:
        - This is a 1D DP problem: dp[i] = minimum coins for amount i
        - Initialize dp[0] = 0 (0 coins for amount 0)
        - For each amount from 1 to target, try each coin
        - dp[i] = min(dp[i], dp[i-coin] + 1) for each valid coin
        - Return dp[amount] or -1 if impossible
        """
        # TODO: Implement coin change solution
        pass
    
    def coin_change_ways(self, coins, amount):
        """
        Same setup, but find the number of ways to make the amount.
        
        HINTS:
        - dp[i] = number of ways to make amount i
        - Initialize dp[0] = 1 (one way to make amount 0: use no coins)
        - For each coin, update dp[i] += dp[i-coin] for all i >= coin
        - This is different from the min coins problem!
        """
        # TODO: Implement number of ways solution
        pass
    
    # Problem 2: Longest Common Subsequence (2D DP)
    def longest_common_subsequence(self, text1, text2):
        """
        Given two strings, find the length of their longest common subsequence.
        A subsequence is a sequence that appears in the same relative order.
        
        HINTS:
        - This is a 2D DP problem: dp[i][j] = LCS of text1[0:i] and text2[0:j]
        - Initialize dp[0][j] = 0 and dp[i][0] = 0 (empty strings)
        - If text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
        - Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        - Return dp[m][n] where m, n are lengths of text1, text2
        """
        # TODO: Implement LCS solution
        pass
    
    def longest_common_subsequence_optimized(self, text1, text2):
        """
        Same problem, but optimize space to O(min(m,n)).
        
        HINTS:
        - Use only two rows of the DP table
        - Alternate between current and previous row
        - This reduces space from O(m*n) to O(min(m,n))
        """
        # TODO: Implement space-optimized LCS solution
        pass
    
    # Problem 3: Matrix Chain Multiplication (Interval DP)
    def matrix_chain_multiplication(self, dimensions):
        """
        Given dimensions of matrices, find the minimum number of scalar multiplications
        needed to compute the product of all matrices.
        
        HINTS:
        - This is an interval DP problem
        - dp[i][j] = minimum multiplications for matrices i to j
        - For each interval length from 2 to n
        - For each starting position i, ending position j = i + length - 1
        - Try all possible split points k: dp[i][j] = min(dp[i][k] + dp[k+1][j] + cost)
        - Cost = dimensions[i] * dimensions[k+1] * dimensions[j+1]
        """
        # TODO: Implement matrix chain multiplication solution
        pass
    
    def matrix_chain_parenthesization(self, dimensions):
        """
        Same problem, but also return the optimal parenthesization.
        
        HINTS:
        - Use a separate table to track the optimal split points
        - parent[i][j] = optimal split point for matrices i to j
        - Use this to reconstruct the optimal parenthesization
        - Return both minimum cost and the parenthesization string
        """
        # TODO: Implement with parenthesization tracking
        pass
    
    def analyze_patterns(self):
        """
        Analyze the patterns in these classic DP problems.
        
        HINTS:
        - Identify what makes each problem 1D, 2D, or interval DP
        - Compare the recurrence relations
        - Think about when you'd use each pattern
        - Consider space optimization opportunities
        """
        # TODO: Write pattern analysis
        pass
```

### Test Cases

```python
# Test cases for you to use
def test_coin_change():
    calc = ClassicDPProblems()
    assert calc.coin_change_min_coins([1, 3, 4], 6) == 2  # 3 + 3
    assert calc.coin_change_ways([1, 3, 4], 6) == 3  # [1,1,1,1,1,1], [3,3], [1,1,4]

def test_lcs():
    calc = ClassicDPProblems()
    assert calc.longest_common_subsequence("abcde", "ace") == 3  # "ace"
    assert calc.longest_common_subsequence("abc", "def") == 0  # no common subsequence

def test_matrix_chain():
    calc = ClassicDPProblems()
    # Test with dimensions [1, 2, 3, 4] representing matrices 1x2, 2x3, 3x4
    result = calc.matrix_chain_multiplication([1, 2, 3, 4])
    assert result == 18  # Optimal: ((A*B)*C) = 1*2*3 + 1*3*4 = 6 + 12 = 18
```

### Deliverables

1. **Complete implementations** of all 6 methods
2. **Test results** demonstrating correctness
3. **Pattern analysis** explaining the differences between DP types
4. **Space optimization** for applicable problems

## 🧠 Key Concepts to Master

- **1D DP**: Problems with one state variable (usually position or value)
- **2D DP**: Problems with two state variables (usually two sequences or positions)
- **Interval DP**: Problems involving intervals or ranges
- **Recurrence Relations**: Mathematical relationships between subproblems
- **Space Optimization**: Reducing space complexity while maintaining correctness

## 📊 Expected Results

- **Coin Change**: O(amount * coins) time, O(amount) space
- **LCS**: O(m * n) time, O(m * n) space (or O(min(m,n)) with optimization)
- **Matrix Chain**: O(n³) time, O(n²) space

## 🎯 Learning Outcomes

After completing this challenge, you will:
- Understand the three classic DP patterns
- Be able to identify which pattern applies to new problems
- Know how to optimize space complexity
- Have experience with interval DP (the most complex pattern)

---

**Ready to master classic DP patterns? Let's start coding!** 🚀
