# Lecture 1: Introduction to Convex Optimization

**Video**: [Convex Optimization - Introduction](https://www.youtube.com/watch?v=McLq1hEq3UY)

## 📝 Learning Objectives

After watching this lecture, you should understand:
- What makes a problem convex vs non-convex
- Why convex optimization is important
- Basic examples of convex optimization problems
- Applications in machine learning and operations research

## 🎯 Challenge: Convex Function Checker

### Problem Statement

Implement a comprehensive convex function checker that can:
1. Test if a given function is convex using multiple criteria
2. Visualize convex vs non-convex functions
3. Handle both analytical and numerical approaches

### Requirements

Create a `ConvexFunctionChecker` class with the following methods:

```python
class ConvexFunctionChecker:
    def __init__(self):
        pass
    
    def is_convex_analytical(self, func, domain):
        """
        Check convexity using analytical methods (first/second order conditions)
        """
        pass
    
    def is_convex_numerical(self, func, domain, num_points=1000):
        """
        Check convexity using numerical methods (Jensen's inequality)
        """
        pass
    
    def visualize_function(self, func, domain, title="Function Plot"):
        """
        Plot the function and highlight convex/non-convex regions
        """
        pass
    
    def test_common_functions(self):
        """
        Test a collection of common convex and non-convex functions
        """
        pass
```

### Test Functions

Implement tests for these functions:

**Convex Functions:**
- f(x) = x²
- f(x) = |x|
- f(x) = e^x
- f(x) = -log(x) (for x > 0)
- f(x,y) = x² + y²

**Non-Convex Functions:**
- f(x) = -x²
- f(x) = sin(x)
- f(x) = x³
- f(x,y) = x² - y²

### Advanced Challenge

Extend your checker to handle:
- Multivariate functions
- Constrained domains
- Piecewise functions
- Functions with discontinuities

### Deliverables

1. **Complete implementation** of `ConvexFunctionChecker`
2. **Visualization** showing convex vs non-convex functions
3. **Test results** for all provided functions
4. **Analysis** of why each function is convex or not
5. **Performance comparison** between analytical and numerical methods

### Bonus Points

- Implement automatic differentiation for gradient/hessian computation
- Add support for checking convexity of constraint sets
- Create interactive visualizations using Plotly or Bokeh

## 🧠 Key Concepts to Master

- **Convex Set**: A set where the line segment between any two points lies entirely within the set
- **Convex Function**: A function where the line segment between any two points on the graph lies above the graph
- **Jensen's Inequality**: For convex f: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
- **First-order Condition**: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
- **Second-order Condition**: ∇²f(x) ⪰ 0 (positive semidefinite)

## 📚 Additional Resources

- [Convex Optimization Book - Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/)
- [Convex Optimization Course Notes](https://see.stanford.edu/Course/EE364A)
- [CVXPY Documentation](https://www.cvxpy.org/) - Python library for convex optimization

---

**Ready to implement? Start coding!** 💻
