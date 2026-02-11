# Week 1 Quiz: Convex Optimization Fundamentals

**Learning Path**: Convex Optimization (Stephen Boyd Lectures)  
**Week**: 1 - Foundations  
**Time Limit**: 30 minutes  
**Total Points**: 100

## Instructions

- Answer all questions to the best of your ability
- Show your work for calculation problems
- Use the provided space for explanations
- Submit your answers in the format specified

---

## Part A: Conceptual Understanding (40 points)

### Question 1 (10 points)
**What are the two key properties that make a problem suitable for convex optimization?**

**Answer:**
```
[Your answer here - 2-3 sentences explaining the properties]
```

### Question 2 (10 points)
**Explain the difference between a convex set and a convex function. Provide one example of each.**

**Answer:**
```
[Your answer here - include definitions and examples]
```

### Question 3 (10 points)
**What is Jensen's inequality and why is it important for convex optimization?**

**Answer:**
```
[Your answer here - mathematical statement and significance]
```

### Question 4 (10 points)
**Give three real-world applications where convex optimization is used. Explain why each is convex.**

**Answer:**
```
[Your answer here - applications and convexity justification]
```

---

## Part B: Mathematical Analysis (35 points)

### Question 5 (15 points)
**Determine if the following function is convex on the domain x > 0:**
**f(x) = -log(x) + x²**

**Show your work using the second-order condition.**

**Answer:**
```
[Your work here - calculate second derivative and analyze]
```

### Question 6 (10 points)
**Check if the set S = {(x,y) | x² + y² ≤ 1, x ≥ 0} is convex.**

**Answer:**
```
[Your work here - use definition of convex set]
```

### Question 7 (10 points)
**For the function f(x,y) = x² + 2xy + y², determine:**
- Is it convex?
- What is its Hessian matrix?
- What are its eigenvalues?

**Answer:**
```
[Your work here - Hessian calculation and analysis]
```

---

## Part C: Implementation Understanding (25 points)

### Question 8 (15 points)
**In your convex function checker implementation, explain:**
- What are the advantages of the analytical method?
- What are the limitations of the numerical method?
- When would you use each approach?

**Answer:**
```
[Your answer here - compare the two methods]
```

### Question 9 (10 points)
**Describe a scenario where your convex function checker might give incorrect results. How would you improve it?**

**Answer:**
```
[Your answer here - identify limitations and improvements]
```

---

## Bonus Question (10 points)

### Question 10 (10 points)
**Implement a function to check if a multivariate function f(x₁, x₂, ..., xₙ) is convex using the Hessian matrix approach. Provide the code and explain your implementation.**

**Answer:**
```python
# Your code here
def multivariate_convex_checker(func, domain, num_points=100):
    """
    Check if a multivariate function is convex using Hessian analysis.
    
    Args:
        func: Function that takes a numpy array and returns a scalar
        domain: List of tuples (min, max) for each dimension
        num_points: Number of points to test
    
    Returns:
        bool: True if function appears convex
    """
    # Your implementation here
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
3. For the bonus question, save your code in `multivariate_convex_checker.py`
4. Submit both files to the assessments folder

## Grading Rubric

- **Conceptual Understanding (40%)**: Accuracy and depth of theoretical knowledge
- **Mathematical Analysis (35%)**: Correctness of calculations and proofs
- **Implementation Understanding (25%)**: Understanding of practical applications
- **Bonus (10%)**: Advanced implementation and analysis

**Total Possible Points**: 110 (100 + 10 bonus)

---

**Good luck! Remember to show your work and explain your reasoning.** 🎯
