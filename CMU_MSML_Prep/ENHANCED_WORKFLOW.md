# CMU MSML Enhanced Workflow - Complete Skill Path

**Based on:** CMU MSML expectations and industry best practices

---

## 🎯 Overview

This workflow takes you from Python basics to PyTorch proficiency, organized into 10 core skill areas with practice exercises.

**Timeline:** 12-16 weeks (adjustable based on your schedule)

---

## 📋 Skill Progression Map

```
Week 1-2: Core Python + NumPy
Week 3-4: Vectorized Operations + Loss Functions
Week 5-6: Linear Regression (Closed-form + Gradient Descent)
Week 7-8: Logistic Regression + Probability
Week 9-10: Pandas + Data Wrangling
Week 11-12: Visualization + Debugging
Week 13-14: PyTorch Fundamentals
Week 15-16: Integration + Algorithms
```

---

## 1️⃣ Core Python (Weeks 1-2)

### Must Master:
- ✅ Loops (for, while)
- ✅ List comprehensions
- ✅ Functions (def, return, parameters)
- ✅ Classes (basic OOP)
- ✅ Error handling (try/except)
- ✅ Clean coding practices

### Practice Exercises:
**Exercise 1.1: Normalization Function**
```python
def normalize(xs):
    """
    Z-score normalization: (x - mean) / std
    
    Requirements:
    - Use list comprehensions
    - Handle edge cases (empty list, single value)
    - Add error handling
    """
    # Your code here
    pass
```

**Exercise 1.2: Data Structure Operations**
- Implement a simple statistics class
- Practice with dictionaries, sets, tuples
- File I/O operations

**Exercise 1.3: Clean Code Refactoring**
- Refactor messy code
- Add docstrings
- Write unit tests

### Checkpoint:
- [ ] Can write clean, readable Python functions
- [ ] Comfortable with list comprehensions
- [ ] Can handle errors gracefully
- [ ] Code is well-documented

---

## 2️⃣ NumPy for Linear Algebra (Weeks 1-2, continued)

### Must Master:
- ✅ Array creation and manipulation
- ✅ Matrix operations (@, .T, .dot())
- ✅ Broadcasting
- ✅ Slicing and indexing
- ✅ Avoid Python loops (vectorize!)

### Practice Exercises:
**Exercise 2.1: Matrix Operations**
```python
import numpy as np

# Create matrices
X = np.array([[1, 2], [3, 4]])
w = np.array([0.5, -1.0])
b = 0.2

# Compute: y = X @ w + b (using broadcasting)
# Your code here
```

**Exercise 2.2: Vectorized Operations**
- Replace loops with NumPy operations
- Practice broadcasting rules
- Matrix multiplication from scratch

**Exercise 2.3: Common Patterns**
- Stack, concatenate, reshape
- Random number generation
- Linear algebra operations (inv, eig, svd)

### Checkpoint:
- [ ] Can do matrix math without loops
- [ ] Understand broadcasting
- [ ] Comfortable with NumPy slicing
- [ ] Can implement common operations

---

## 3️⃣ Vectorized Loss + Gradients (Weeks 3-4)

### Must Master:
- ✅ Mean Squared Error (MSE)
- ✅ Cross-Entropy Loss
- ✅ Gradient computation (manual)
- ✅ Vectorized implementations

### Practice Exercises:
**Exercise 3.1: MSE Loss**
```python
def mse_loss(y_true, y_pred):
    """
    Mean Squared Error: mean((y_true - y_pred)^2)
    
    Requirements:
    - Vectorized (no loops)
    - Handle edge cases
    - Return scalar
    """
    # Your code here
    pass
```

**Exercise 3.2: Gradient Computation**
```python
def compute_gradient(X, y, w):
    """
    Compute gradient for linear regression:
    grad = (2/m) * X.T @ (X @ w - y)
    
    Requirements:
    - Vectorized
    - Correct dimensions
    """
    # Your code here
    pass
```

**Exercise 3.3: Multiple Loss Functions**
- Implement MSE, MAE, Huber loss
- Compare gradients
- Visualize loss surfaces

### Checkpoint:
- [ ] Can derive gradients manually
- [ ] Implement losses from scratch
- [ ] All operations vectorized
- [ ] Understand gradient shapes

---

## 4️⃣ Linear Regression (Weeks 5-6)

### Must Master:
- ✅ Closed-form solution (Normal Equation)
- ✅ Gradient descent implementation
- ✅ Compare both methods
- ✅ Understand when to use each

### Practice Exercises:
**Exercise 4.1: Closed-Form Solution**
```python
def linear_regression_closed_form(X, y):
    """
    Normal Equation: w = (X.T @ X)^(-1) @ X.T @ y
    
    Requirements:
    - Handle singular matrices
    - Add bias term
    - Return weights
    """
    # Your code here
    pass
```

**Exercise 4.2: Gradient Descent**
```python
def linear_regression_gd(X, y, lr=0.01, epochs=1000):
    """
    Gradient Descent for Linear Regression
    
    Requirements:
    - Initialize weights
    - Update rule: w -= lr * grad
    - Track loss over iterations
    - Return final weights and loss history
    """
    # Your code here
    pass
```

**Exercise 4.3: Comparison Study**
- Compare closed-form vs GD
- Analyze convergence
- Test on different datasets
- Measure performance

### Checkpoint:
- [ ] Can implement both methods
- [ ] Understand trade-offs
- [ ] Can debug convergence issues
- [ ] Know when to use each

---

## 5️⃣ Logistic Regression (Weeks 7-8)

### Must Master:
- ✅ Sigmoid function
- ✅ Binary classification
- ✅ Gradient descent for logistic regression
- ✅ Decision boundary

### Practice Exercises:
**Exercise 5.1: Sigmoid Implementation**
```python
def sigmoid(z):
    """
    Sigmoid: 1 / (1 + exp(-z))
    
    Requirements:
    - Handle numerical stability
    - Vectorized
    - Works with arrays
    """
    # Your code here
    pass
```

**Exercise 5.2: Logistic Regression Training**
```python
def train_logistic_regression(X, y, lr=0.01, epochs=500):
    """
    Train logistic regression with gradient descent
    
    Requirements:
    - Initialize weights
    - Compute predictions: sigmoid(X @ w)
    - Gradient: X.T @ (preds - y) / len(y)
    - Update weights
    - Track loss (cross-entropy)
    """
    # Your code here
    pass
```

**Exercise 5.3: End-to-End Pipeline**
- Load data
- Preprocess
- Train model
- Evaluate (accuracy, precision, recall)
- Visualize decision boundary

### Checkpoint:
- [ ] Can implement sigmoid correctly
- [ ] Understand binary classification
- [ ] Can train logistic regression
- [ ] Know how to evaluate

---

## 6️⃣ Basic Probability & Simulation (Weeks 7-8, continued)

### Must Master:
- ✅ Random sampling
- ✅ Monte Carlo methods
- ✅ Probability distributions
- ✅ Statistical inference

### Practice Exercises:
**Exercise 6.1: Monte Carlo Simulation**
```python
import numpy as np

# Estimate π using Monte Carlo
def estimate_pi(n_samples=10000):
    """
    Use random sampling to estimate π
    
    Requirements:
    - Generate random points
    - Count points in circle
    - Return estimate
    """
    # Your code here
    pass
```

**Exercise 6.2: Distribution Sampling**
- Sample from normal, uniform, exponential
- Compute mean, variance
- Visualize distributions

**Exercise 6.3: Statistical Tests**
- Hypothesis testing basics
- Confidence intervals
- Bootstrap sampling

### Checkpoint:
- [ ] Can use random sampling
- [ ] Understand Monte Carlo
- [ ] Comfortable with distributions
- [ ] Can compute statistics

---

## 7️⃣ Pandas + Data Wrangling (Weeks 9-10)

### Must Master:
- ✅ Reading/writing CSV
- ✅ Data cleaning (dropna, fillna)
- ✅ Feature engineering
- ✅ Groupby operations
- ✅ Merging/joining

### Practice Exercises:
**Exercise 7.1: Data Cleaning**
```python
import pandas as pd
import numpy as np

# Load and clean data
df = pd.read_csv("data.csv")

# Requirements:
# - Handle missing values
# - Remove duplicates
# - Convert data types
# - Create new features
```

**Exercise 7.2: Feature Engineering**
- Create log transformations
- Bin continuous variables
- Create interaction features
- Handle categorical data

**Exercise 7.3: Data Analysis**
- Groupby aggregations
- Pivot tables
- Time series operations
- Export cleaned data

### Checkpoint:
- [ ] Can load and clean data
- [ ] Comfortable with pandas operations
- [ ] Can engineer features
- [ ] Can handle messy data

---

## 8️⃣ Visualization (Weeks 11-12)

### Must Master:
- ✅ matplotlib basics
- ✅ Plotting loss curves
- ✅ Scatter plots, histograms
- ✅ Debugging with plots

### Practice Exercises:
**Exercise 8.1: Loss Visualization**
```python
import matplotlib.pyplot as plt

def plot_loss_curve(losses):
    """
    Plot training loss over iterations
    
    Requirements:
    - Clear labels
    - Proper formatting
    - Save figure
    """
    # Your code here
    pass
```

**Exercise 8.2: Data Exploration**
- Plot distributions
- Scatter plots with regression line
- Correlation heatmaps
- Feature importance plots

**Exercise 8.3: Model Debugging**
- Plot predictions vs actual
- Residual plots
- Decision boundaries
- Learning curves

### Checkpoint:
- [ ] Can create basic plots
- [ ] Use plots for debugging
- [ ] Professional-looking figures
- [ ] Can save/export plots

---

## 9️⃣ PyTorch Fundamentals (Weeks 13-14)

### Must Master:
- ✅ Tensors (creation, operations)
- ✅ Computation graphs
- ✅ Models (nn.Module)
- ✅ Loss functions
- ✅ Optimizers
- ✅ Training loop
- ✅ Backpropagation (backward())

### Practice Exercises:
**Exercise 9.1: Basic Tensors**
```python
import torch

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Operations
z = x + y
z = x * y
z = torch.dot(x, y)
```

**Exercise 9.2: Simple Model**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
model = nn.Linear(5, 1)

# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Training loop
for x, y in dataloader:
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
```

**Exercise 9.3: Custom Model**
- Define custom nn.Module
- Implement forward pass
- Train on real data
- Evaluate performance

### Checkpoint:
- [ ] Understand tensors
- [ ] Can define models
- [ ] Understand computation graphs
- [ ] Can write training loops
- [ ] Know overfitting vs generalization

---

## 🔟 Algorithms & Complexity (Weeks 15-16)

### Must Master:
- ✅ Time complexity (O notation)
- ✅ Sorting algorithms
- ✅ Searching algorithms
- ✅ Hash tables vs lists
- ✅ When to use what

### Practice Exercises:
**Exercise 10.1: Complexity Analysis**
- Analyze your ML code
- Identify bottlenecks
- Optimize slow operations

**Exercise 10.2: Data Structure Choice**
- When to use list vs dict vs set
- Implement simple hash table
- Compare search times

**Exercise 10.3: Algorithm Implementation**
- Implement quicksort
- Binary search
- Graph algorithms (if time)

### Checkpoint:
- [ ] Can analyze complexity
- [ ] Choose right data structure
- [ ] Understand trade-offs
- [ ] Can optimize code

---

## 🌟 Bonus Skills (Throughout)

### Writing Unit Tests
```python
import unittest

class TestNormalize(unittest.TestCase):
    def test_normalize(self):
        result = normalize([1, 2, 3, 4])
        self.assertEqual(len(result), 4)
        # Add more tests
```

### Good Naming & Comments
- Descriptive variable names
- Clear function names
- Helpful docstrings
- Inline comments for complex logic

### Reproducibility
```python
import numpy as np
import random

# Set seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
```

### Git Workflow
- Commit frequently
- Meaningful commit messages
- Branch for experiments
- Clean history

### Jupyter Discipline
- Clear markdown cells
- Organized structure
- Save outputs
- Version control notebooks

---

## 📅 Weekly Schedule Template

### Monday: Learn
- Read concept explanation
- Watch videos (if available)
- Take notes
- Understand theory

### Tuesday-Wednesday: Practice
- Work on exercises
- Code from scratch
- Test your solutions
- Debug issues

### Thursday: Review
- Compare with solutions
- Refactor code
- Write tests
- Document learnings

### Friday: Apply
- Mini-project using skills
- Real dataset
- End-to-end pipeline
- Write report

### Weekend: Integrate
- Connect to other skills
- Review CMU materials
- Plan next week
- Rest!

---

## ✅ Mastery Checklist

### Must-Have (Core):
- [ ] Python fluency (loops, functions, classes)
- [ ] NumPy linear algebra (matrices, broadcasting)
- [ ] Implement linear regression (closed-form + GD)
- [ ] Implement logistic regression
- [ ] Gradient descent & vectorization
- [ ] Basic probability & simulation

### Strongly Recommended:
- [ ] PyTorch basics (tensors, models, training)
- [ ] Pandas wrangling (cleaning, features)
- [ ] Plotting & debugging (matplotlib)
- [ ] Writing clean reusable code

### Nice-to-Have:
- [ ] Regularization (L1, L2)
- [ ] Cross-validation
- [ ] KNN, SVM intuition
- [ ] Advanced optimization (momentum, Adam)

### Professional Skills:
- [ ] Unit tests
- [ ] Good naming & comments
- [ ] Reproducibility (seeds)
- [ ] Git workflow
- [ ] Jupyter discipline

---

## 🎯 Practice Path Options

### Option A: Small Coding Drills (Recommended for Beginners)
- **Week 1-2:** 3 NumPy problems/day
- **Week 3-4:** 2 loss function implementations/day
- **Week 5-6:** 1 regression problem/day
- **Focus:** Master fundamentals first

### Option B: Full Mini-Projects (Recommended for Experienced)
- **Week 1-2:** NumPy mini-project (matrix operations library)
- **Week 3-4:** Loss functions library
- **Week 5-6:** Complete regression project
- **Focus:** Build portfolio pieces

### Option C: Hybrid (Balanced)
- **Mon-Wed:** Small drills
- **Thu-Fri:** Mini-project integration
- **Weekend:** Review and plan
- **Focus:** Balance practice and application

---

## 📚 Resources

### Learning Materials:
- NumPy documentation
- PyTorch tutorials
- Pandas user guide
- CMU course materials

### Practice:
- This repository (all exercises)
- Kaggle Learn courses
- CMU practice problems
- LeetCode (for algorithms)

### Community:
- CMU MSML Discord
- Stack Overflow
- Reddit (r/MachineLearning)
- Study groups

---

## 🚀 Getting Started

1. **Assess Your Level:**
   - Take the skills assessment
   - Identify gaps
   - Choose practice path

2. **Set Up Environment:**
   ```bash
   pip install numpy pandas matplotlib torch jupyter
   ```

3. **Start with Week 1:**
   - Core Python review
   - NumPy basics
   - First exercises

4. **Track Progress:**
   - Use checklist
   - Commit code regularly
   - Document learnings

---

## 💡 Tips for Success

1. **Code Daily:** Even 30 minutes helps
2. **From Scratch:** Don't copy-paste, type it out
3. **Understand Why:** Not just how
4. **Test Everything:** Write unit tests
5. **Document:** Comments and docstrings
6. **Review:** Revisit old code
7. **Connect:** Link concepts together
8. **Practice:** More practice = more confidence

---

**Remember:** This is a marathon, not a sprint. Consistency beats intensity. Good luck! 🎓

