# Lecture 1: Introduction to Dynamic Programming

**Video**: [MIT 6.0002 - Dynamic Programming](https://www.youtube.com/watch?v=OQ5jsbhAv_M)

## 📝 Learning Objectives

After watching this lecture, you should understand:
- What dynamic programming is and when to use it
- The two key properties: overlapping subproblems and optimal substructure
- The difference between memoization and tabulation
- How DP can dramatically improve time complexity

## 🎯 Challenge: Fibonacci Comparison

### Problem Statement

Implement and compare different approaches to computing Fibonacci numbers:
1. **Naive Recursive**: Basic recursive implementation
2. **Memoized Recursive**: Top-down DP with memoization
3. **Tabulated Iterative**: Bottom-up DP with tabulation
4. **Space-Optimized**: DP with O(1) space complexity

### Requirements

Create a `FibonacciCalculator` class with the following methods:

```python
class FibonacciCalculator:
    def __init__(self):
        self.memo = {}
        self.call_count = 0
    
    def naive_fibonacci(self, n):
        """Basic recursive implementation - O(2^n) time"""
        pass
    
    def memoized_fibonacci(self, n):
        """Top-down DP with memoization - O(n) time, O(n) space"""
        pass
    
    def tabulated_fibonacci(self, n):
        """Bottom-up DP with tabulation - O(n) time, O(n) space"""
        pass
    
    def space_optimized_fibonacci(self, n):
        """DP with O(1) space - O(n) time, O(1) space"""
        pass
    
    def compare_performance(self, n_values):
        """Compare all methods for different values of n"""
        pass
    
    def visualize_performance(self, max_n=30):
        """Create performance comparison charts"""
        pass
```

### Advanced Challenge

Extend your implementation to handle:
- **Matrix Exponentiation**: O(log n) time complexity
- **Large Numbers**: Handle Fibonacci numbers with 100+ digits
- **Modular Arithmetic**: Compute Fibonacci numbers modulo a large prime
- **Generalized Fibonacci**: Support custom recurrence relations

### Test Cases

```python
# Basic test cases
assert fibonacci_calc.naive_fibonacci(10) == 55
assert fibonacci_calc.memoized_fibonacci(10) == 55
assert fibonacci_calc.tabulated_fibonacci(10) == 55
assert fibonacci_calc.space_optimized_fibonacci(10) == 55

# Edge cases
assert fibonacci_calc.memoized_fibonacci(0) == 0
assert fibonacci_calc.memoized_fibonacci(1) == 1
assert fibonacci_calc.memoized_fibonacci(2) == 1
```

### Performance Analysis

Your implementation should demonstrate:
- **Time Complexity**: Show exponential vs linear growth
- **Space Complexity**: Compare memory usage patterns
- **Call Count**: Track recursive calls for naive vs memoized
- **Scalability**: Test with n = 40, 50, 100

### Deliverables

1. **Complete implementation** of all Fibonacci methods
2. **Performance comparison** with timing and memory analysis
3. **Visualization** showing time/space complexity differences
4. **Analysis report** explaining when to use each approach
5. **Bonus implementations** for advanced challenges

## 🧠 Key Concepts to Master

- **Overlapping Subproblems**: Same subproblems solved multiple times
- **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
- **Memoization**: Store results of expensive function calls
- **Tabulation**: Build solutions bottom-up using tables
- **Time-Space Tradeoff**: Balance between time and memory usage

## 📊 Expected Results

For n = 40:
- **Naive**: ~1 billion recursive calls, several seconds
- **Memoized**: ~40 recursive calls, milliseconds
- **Tabulated**: ~40 iterations, milliseconds
- **Space-Optimized**: ~40 iterations, constant space

## 🎯 Learning Outcomes

After completing this challenge, you will:
- Understand why DP is powerful for optimization problems
- Know when to use memoization vs tabulation
- Be able to optimize DP solutions for space complexity
- Have a foundation for more complex DP problems

## 📚 Additional Resources

- [MIT 6.006 - Dynamic Programming](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/)
- [CS50 - Dynamic Programming](https://cs50.harvard.edu/college/2023/fall/weeks/7/)
- [LeetCode DP Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)

---

**Ready to optimize Fibonacci? Let's start coding!** 🚀
