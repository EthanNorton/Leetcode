# Week 1 Quiz: Dynamic Programming Fundamentals

**Learning Path**: Dynamic Programming (MIT OCW & CS50)  
**Week**: 1 - Fundamentals  
**Time Limit**: 30 minutes  
**Total Points**: 100

## Instructions

- Answer all questions to the best of your ability
- Show your work for algorithm analysis problems
- Use the provided space for explanations
- Submit your answers in the format specified

---

## Part A: Conceptual Understanding (40 points)

### Question 1 (10 points)
**What are the two key properties that make a problem suitable for dynamic programming?**

**Answer:**
```
[Your answer here - explain overlapping subproblems and optimal substructure]
```

### Question 2 (10 points)
**Explain the difference between memoization and tabulation. When would you use each approach?**

**Answer:**
```
[Your answer here - compare top-down vs bottom-up approaches]
```

### Question 3 (10 points)
**Why does the naive recursive Fibonacci algorithm have exponential time complexity? Show the recurrence relation.**

**Answer:**
```
[Your answer here - recurrence relation and complexity analysis]
```

### Question 4 (10 points)
**Give three examples of problems that can be solved using dynamic programming. Explain why each has the DP properties.**

**Answer:**
```
[Your answer here - problems and DP property justification]
```

---

## Part B: Algorithm Analysis (35 points)

### Question 5 (15 points)
**Analyze the time and space complexity of your Fibonacci implementations:**

| Method | Time Complexity | Space Complexity | Explanation |
|--------|----------------|------------------|-------------|
| Naive Recursive | | | |
| Memoized | | | |
| Tabulated | | | |
| Space Optimized | | | |
| Matrix Exponentiation | | | |

**Answer:**
```
[Fill in the table above with complexity analysis]
```

### Question 6 (10 points)
**For the 0/1 Knapsack problem with n items and capacity W:**
- What is the time complexity of the DP solution?
- What is the space complexity?
- How would you optimize the space complexity?

**Answer:**
```
[Your analysis here]
```

### Question 7 (10 points)
**Consider the problem: "Find the minimum number of coins needed to make change for amount n using coins of denominations [1, 3, 4]."**
- Formulate this as a DP problem
- Write the recurrence relation
- What is the time and space complexity?

**Answer:**
```
[Your DP formulation here]
```

---

## Part C: Implementation Understanding (25 points)

### Question 8 (15 points)
**In your Fibonacci comparison implementation, explain:**
- How does memoization eliminate redundant calculations?
- Why does the call count differ between naive and memoized versions?
- What are the trade-offs between different approaches?

**Answer:**
```
[Your explanation here]
```

### Question 9 (10 points)
**Describe how you would modify your Fibonacci calculator to handle:**
- Very large numbers (100+ digits)
- Modular arithmetic (Fibonacci mod 10^9 + 7)
- Multiple queries efficiently

**Answer:**
```
[Your approach here]
```

---

## Bonus Question (10 points)

### Question 10 (10 points)
**Implement a generic DP solver that can handle any recurrence relation. The solver should:**
- Take a recurrence function as input
- Support both memoization and tabulation
- Handle base cases automatically
- Return the result and performance metrics

**Answer:**
```python
# Your code here
class GenericDPSolver:
    def __init__(self, recurrence_func, base_cases):
        """
        Initialize the DP solver.
        
        Args:
            recurrence_func: Function that computes f(n) from f(n-1), f(n-2), etc.
            base_cases: Dictionary mapping indices to values for base cases
        """
        pass
    
    def solve_memoized(self, n):
        """Solve using memoization."""
        pass
    
    def solve_tabulated(self, n):
        """Solve using tabulation."""
        pass
    
    def compare_methods(self, n_values):
        """Compare performance of both methods."""
        pass
```

**Explanation:**
```
[Your explanation here]
```

---

## Submission Instructions

1. Complete all questions above
2. Save your answers in a file named `quiz_week1_answers.md`
3. For the bonus question, save your code in `generic_dp_solver.py`
4. Submit both files to the assessments folder

## Grading Rubric

- **Conceptual Understanding (40%)**: Accuracy and depth of theoretical knowledge
- **Algorithm Analysis (35%)**: Correctness of complexity analysis and problem formulation
- **Implementation Understanding (25%)**: Understanding of practical applications
- **Bonus (10%)**: Advanced implementation and analysis

**Total Possible Points**: 110 (100 + 10 bonus)

---

**Good luck! Remember to show your work and explain your reasoning.** 🎯
