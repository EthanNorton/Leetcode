# Lecture 2: Memoization vs Tabulation

**Video**: [CS50 - Dynamic Programming](https://www.youtube.com/watch?v=OQ5jsbhAv_M)

## 📝 Learning Objectives

After watching this lecture, you should understand:
- The difference between top-down (memoization) and bottom-up (tabulation) approaches
- When to use memoization vs tabulation
- Time and space complexity trade-offs
- Implementation patterns for both approaches

## 🎯 Challenge: Compare Top-Down vs Bottom-Up Approaches

### Problem Statement

Implement solutions to classic DP problems using both memoization and tabulation, then compare their performance and characteristics.

### Requirements

Create a `DPComparison` class with the following structure:

```python
class DPComparison:
    def __init__(self):
        pass
    
    # Problem 1: Climbing Stairs
    def climbing_stairs_memoized(self, n):
        """
        You are climbing a staircase. It takes n steps to reach the top.
        Each time you can either climb 1 or 2 steps.
        Return the number of distinct ways to climb to the top.
        
        HINTS:
        - Think about the base cases: what happens when n=0, n=1, n=2?
        - The recurrence relation is: ways(n) = ways(n-1) + ways(n-2)
        - Use a dictionary to store computed results
        - Don't forget to check if the result is already computed!
        """
        # TODO: Implement memoized solution
        pass
    
    def climbing_stairs_tabulated(self, n):
        """
        Same problem as above, but using tabulation.
        
        HINTS:
        - Create a table/dp array to store results
        - Fill the table bottom-up (from 0 to n)
        - Initialize base cases first
        - Use the recurrence relation to fill remaining cells
        """
        # TODO: Implement tabulated solution
        pass
    
    # Problem 2: House Robber
    def house_robber_memoized(self, nums):
        """
        You are a robber planning to rob houses along a street.
        Each house has a certain amount of money stashed.
        Adjacent houses have security systems connected, so you can't rob two adjacent houses.
        Return the maximum amount of money you can rob.
        
        HINTS:
        - Think about the decision at each house: rob it or skip it
        - If you rob house i, you can't rob house i-1
        - If you skip house i, you can rob house i-1
        - The recurrence: rob(i) = max(rob(i-1), rob(i-2) + nums[i])
        - Use memoization to avoid recomputing subproblems
        """
        # TODO: Implement memoized solution
        pass
    
    def house_robber_tabulated(self, nums):
        """
        Same problem as above, but using tabulation.
        
        HINTS:
        - Create dp array where dp[i] = max money from houses 0 to i
        - Initialize dp[0] = nums[0] and dp[1] = max(nums[0], nums[1])
        - For each house i >= 2: dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        - Return dp[n-1] where n is the length of nums
        """
        # TODO: Implement tabulated solution
        pass
    
    # Problem 3: Longest Increasing Subsequence
    def lis_memoized(self, nums):
        """
        Given an array of integers, find the length of the longest increasing subsequence.
        
        HINTS:
        - For each position i, find the LIS ending at position i
        - LIS(i) = 1 + max(LIS(j)) for all j < i where nums[j] < nums[i]
        - Use memoization to store results for each position
        - The answer is the maximum LIS(i) for all i
        """
        # TODO: Implement memoized solution
        pass
    
    def lis_tabulated(self, nums):
        """
        Same problem as above, but using tabulation.
        
        HINTS:
        - Create dp array where dp[i] = LIS ending at position i
        - Initialize all dp[i] = 1 (each element is a subsequence of length 1)
        - For each i, check all previous positions j < i
        - If nums[j] < nums[i], then dp[i] = max(dp[i], dp[j] + 1)
        - Return the maximum value in dp array
        """
        # TODO: Implement tabulated solution
        pass
    
    def compare_performance(self, test_cases):
        """
        Compare the performance of memoized vs tabulated solutions.
        
        HINTS:
        - Test each problem with different input sizes
        - Measure execution time for both approaches
        - Compare space usage (if possible)
        - Create a nice table showing the results
        """
        # TODO: Implement performance comparison
        pass
    
    def analyze_tradeoffs(self):
        """
        Analyze the trade-offs between memoization and tabulation.
        
        HINTS:
        - Think about time complexity (usually the same)
        - Consider space complexity differences
        - Think about when you might prefer one over the other
        - Consider implementation complexity
        """
        # TODO: Write analysis of trade-offs
        pass
```

### Test Cases

```python
# Test cases for you to use
def test_climbing_stairs():
    calc = DPComparison()
    assert calc.climbing_stairs_memoized(2) == 2  # [1,1] or [2]
    assert calc.climbing_stairs_memoized(3) == 3  # [1,1,1], [1,2], [2,1]
    assert calc.climbing_stairs_tabulated(2) == 2
    assert calc.climbing_stairs_tabulated(3) == 3

def test_house_robber():
    calc = DPComparison()
    assert calc.house_robber_memoized([1,2,3,1]) == 4  # Rob houses 0 and 2
    assert calc.house_robber_memoized([2,7,9,3,1]) == 12  # Rob houses 0, 2, 4
    assert calc.house_robber_tabulated([1,2,3,1]) == 4
    assert calc.house_robber_tabulated([2,7,9,3,1]) == 12

def test_lis():
    calc = DPComparison()
    assert calc.lis_memoized([10,9,2,5,3,7,101,18]) == 4  # [2,3,7,18]
    assert calc.lis_tabulated([10,9,2,5,3,7,101,18]) == 4
```

### Deliverables

1. **Complete implementations** of all 6 methods (3 problems × 2 approaches)
2. **Performance comparison** showing timing results
3. **Analysis report** explaining when to use each approach
4. **Test results** demonstrating correctness

## 🧠 Key Concepts to Master

- **Memoization**: Top-down approach, recursive with caching
- **Tabulation**: Bottom-up approach, iterative with table filling
- **Time Complexity**: Usually O(n) for both approaches
- **Space Complexity**: O(n) for memoization (recursion stack + cache), O(n) for tabulation (table)
- **Implementation**: Memoization is often more intuitive, tabulation is more efficient

## 📊 Expected Results

For each problem, both approaches should:
- Have the same time complexity
- Have similar space complexity
- Produce identical results
- Show different performance characteristics in practice

## 🎯 Learning Outcomes

After completing this challenge, you will:
- Understand when to use memoization vs tabulation
- Be able to implement both approaches for any DP problem
- Know the trade-offs between the two methods
- Have experience with 3 classic DP problems

---

**Ready to implement both approaches? Let's start coding!** 🚀
