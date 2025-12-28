# How to Solve These Exercises: A Thinking Guide

**Goal:** Learn how to approach and solve ML/coding problems step-by-step.

---

## 🧠 General Problem-Solving Framework

### The 5-Step Process

1. **Understand** → What is the problem asking?
2. **Plan** → How will you solve it?
3. **Break Down** → What are the smaller pieces?
4. **Implement** → Code it step-by-step
5. **Test & Debug** → Verify it works

---

## 📋 Step-by-Step: Exercise 1 (Normalize Function)

### Step 1: Understand the Problem

**What is normalization?**
- Normalization transforms data to have mean=0 and std=1
- Formula: `normalized = (x - mean) / std`
- This makes different features comparable

**What does the function need to do?**
- Take a list of numbers
- Calculate mean and standard deviation
- Transform each number using the formula
- Return the normalized list

**Example:**
- Input: `[1, 2, 3, 4]`
- Mean: `(1+2+3+4)/4 = 2.5`
- Std: `sqrt(sum((x-2.5)^2)/4) ≈ 1.118`
- Output: `[~-1.34, ~-0.45, ~0.45, ~1.34]`

### Step 2: Plan Your Approach

**Mental checklist:**
- [ ] Handle empty list → raise error
- [ ] Handle single value → return [0.0]
- [ ] Calculate mean
- [ ] Calculate standard deviation
- [ ] Handle division by zero (std=0)
- [ ] Apply formula to each element

### Step 3: Break Down into Smaller Steps

**Sub-problems:**
1. **Input validation** - Check if list is empty
2. **Edge case** - Single value returns [0.0]
3. **Mean calculation** - `sum(xs) / len(xs)`
4. **Variance calculation** - `sum((x-mean)^2) / len(xs)`
5. **Std calculation** - `sqrt(variance)`
6. **Handle std=0** - Prevent division by zero
7. **Apply formula** - Use list comprehension

### Step 4: Implement Step-by-Step

```python
def normalize(xs):
    # Step 1: Input validation
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    # Step 2: Edge case - single value
    if len(xs) == 1:
        return [0.0]
    
    # Step 3: Calculate mean
    mean = sum(xs) / len(xs)
    
    # Step 4: Calculate variance
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    
    # Step 5: Calculate standard deviation
    std = variance ** 0.5  # or: math.sqrt(variance)
    
    # Step 6: Handle division by zero
    if std == 0:
        # All values are the same, return zeros
        return [0.0] * len(xs)
    
    # Step 7: Apply formula using list comprehension
    return [(x - mean) / std for x in xs]
```

### Step 5: Think About Edge Cases

**What could go wrong?**
1. Empty list → Handle with ValueError
2. Single value → Return [0.0] (can't normalize one value)
3. All same values → std=0, handle division by zero
4. Negative numbers → Should work fine
5. Very large numbers → Should work, but watch precision

### 💡 Key Insights

- **List comprehensions** are your friend: `[formula for x in xs]`
- **Break it down**: Mean → Variance → Std → Formula
- **Handle edge cases** early (empty, single, same values)
- **Test incrementally**: Test each piece as you build

---

## 📋 Step-by-Step: Exercise 2 (Matrix Operations)

### Step 1: Understand NumPy Basics

**Key concepts:**
- Arrays are multi-dimensional
- Shape: `(rows, columns)` or `(n_samples, n_features)`
- `@` is matrix multiplication (different from `*`)
- Broadcasting: automatic dimension expansion

**Matrix multiplication:**
- `A @ B`: (m×n) × (n×p) → (m×p)
- Columns of A must equal rows of B
- Element at (i,j) = dot product of row i of A and column j of B

### Step 2: Matrix-Vector Multiplication

**Problem:** `y = X @ w + b`

**How to think:**
1. `X` is (3, 2) - 3 samples, 2 features
2. `w` is (2,) - 2 weights
3. `X @ w` gives (3,) - one value per sample
4. `b` is scalar - broadcasts to all samples
5. Result: (3,) - predictions for 3 samples

**Step-by-step:**
```python
# Given:
X = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])  # (3, 2)
w = np.array([0.5, -1.0])  # (2,)
b = 0.2  # scalar

# Step 1: Matrix multiplication
# X @ w = [1*0.5 + 2*(-1.0), 3*0.5 + 4*(-1.0), 5*0.5 + 6*(-1.0)]
#       = [-1.5, -2.5, -3.5]  (shape: 3,)

# Step 2: Add bias (broadcasting)
# y = [-1.5, -2.5, -3.5] + 0.2
#   = [-1.3, -2.3, -3.3]

# Implementation:
y = X @ w + b  # NumPy handles everything!
```

### Step 3: Vectorized Operations

**Key insight:** Don't use loops, use NumPy operations!

**Element-wise operations:**
- `a + b` → adds element by element
- `a * b` → multiplies element by element
- `a ** 2` → squares each element
- `a @ b` → dot product (sum of element-wise product)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise (vectorized):
sum_result = a + b        # [5, 7, 9]
product_result = a * b    # [4, 10, 18]
power_result = a ** 2     # [1, 4, 9]

# Dot product:
dot_product = a @ b       # 1*4 + 2*5 + 3*6 = 32
# OR: np.dot(a, b)
```

### Step 4: Broadcasting

**How broadcasting works:**
- NumPy automatically expands smaller arrays
- Rule: dimensions must be compatible
- Works when shapes align from the right

**Example:**
```python
matrix = np.array([[1, 2],    # (3, 2)
                   [3, 4],
                   [5, 6]])
vector = np.array([10, 20])   # (2,)

# Broadcasting automatically does:
# vector becomes: [[10, 20],
#                  [10, 20],
#                  [10, 20]]

result = matrix + vector
# Result: [[11, 22],
#          [13, 24],
#          [15, 26]]
```

### 💡 Key Insights

- **Use `@` for matrix multiplication**, not `*`
- **`*` is element-wise** multiplication
- **Broadcasting is automatic** - trust NumPy!
- **Avoid loops** - NumPy operations are much faster
- **Check shapes** if something goes wrong: `array.shape`

---

## 📋 Step-by-Step: Exercise 3 (MSE Loss)

### Step 1: Understand Loss Functions

**What is MSE?**
- Mean Squared Error measures prediction error
- Formula: `mean((y_true - y_pred)^2)`
- Penalizes large errors more (squared term)
- Always non-negative (squared)

**Why use it?**
- Differentiable (smooth)
- Penalizes outliers heavily
- Works well for regression

### Step 2: Break Down the Formula

**Formula:** `MSE = mean((y_true - y_pred)^2)`

**Step-by-step thinking:**
1. Compute difference: `error = y_true - y_pred`
2. Square the difference: `squared_error = error^2`
3. Take the mean: `mse = mean(squared_error)`

**Example:**
```python
y_true = [1, 2, 3]
y_pred = [1.1, 1.9, 3.2]

# Step 1: Difference
error = [1-1.1, 2-1.9, 3-3.2] = [-0.1, 0.1, -0.2]

# Step 2: Square
squared_error = [0.01, 0.01, 0.04]

# Step 3: Mean
mse = (0.01 + 0.01 + 0.04) / 3 = 0.02
```

### Step 3: Vectorized Implementation

**Key: Use NumPy, no loops!**

```python
def mse_loss(y_true, y_pred):
    # Convert to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Step 1: Difference (vectorized)
    error = y_true - y_pred
    
    # Step 2: Square (vectorized)
    squared_error = error ** 2
    
    # Step 3: Mean
    mse = np.mean(squared_error)
    
    return mse
```

**Or in one line:**
```python
return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
```

### Step 4: Gradient Computation

**What is a gradient?**
- Gradient tells us how to change parameters to reduce loss
- For MSE: `grad = (2/m) * X.T @ (y_pred - y_true)`
- Points in direction of steepest increase (we move opposite)

**How to think about it:**
1. Error: `y_pred - y_true` (how wrong we are)
2. Multiply by X.T: projects error onto each feature
3. Scale by `2/m`: normalization and derivative factor

**Step-by-step:**
```python
def mse_loss_gradient(y_true, y_pred, X):
    # Convert to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    X = np.array(X)
    
    # Step 1: Number of samples
    m = len(y_true)  # or: y_true.shape[0]
    
    # Step 2: Compute error
    error = y_pred - y_true  # (m,)
    
    # Step 3: Project error onto features
    # X.T is (n_features, m)
    # error is (m,)
    # X.T @ error gives (n_features,) - gradient for each feature
    gradient = X.T @ error
    
    # Step 4: Scale by 2/m (from derivative)
    gradient = (2 / m) * gradient
    
    return gradient
```

### 💡 Key Insights

- **MSE formula**: Mean of squared differences
- **Vectorize everything**: Use NumPy, avoid loops
- **Gradient formula**: `(2/m) * X.T @ error`
- **Shapes matter**: Check dimensions carefully
- **Think step-by-step**: Error → Square → Mean

---

## 🎯 General Problem-Solving Strategies

### Strategy 1: Start Simple, Build Up

1. **Get something working first** (even if it's wrong)
2. **Make it correct**
3. **Make it elegant**
4. **Handle edge cases**

**Example for normalize:**
```python
# Version 1: Basic (might not handle edge cases)
def normalize(xs):
    mean = sum(xs) / len(xs)
    std = (sum((x-mean)**2 for x in xs) / len(xs))**0.5
    return [(x-mean)/std for x in xs]

# Version 2: Add edge cases
def normalize(xs):
    if not xs:
        raise ValueError("Empty list")
    if len(xs) == 1:
        return [0.0]
    # ... rest of code
```

### Strategy 2: Use Test Cases to Guide You

**Write test cases BEFORE implementing:**
```python
# Think: What should happen?
normalize([1, 2, 3, 4])  # Should give mean=0, std=1
normalize([5])           # Should return [0.0]
normalize([])            # Should raise error
normalize([1, 1, 1, 1])  # Should handle std=0
```

### Strategy 3: Draw It Out (For Math Problems)

**For matrix operations:**
- Draw the shapes
- Draw the operation
- Verify dimensions match

```
X:      w:      X@w:    b:    result:
[1 2]   [0.5]   [-1.5]   0.2   [-1.3]
[3 4] @ [-1.0] =[-2.5] + 0.2 = [-2.3]
[5 6]           [-3.5]   0.2   [-3.3]
(3,2)   (2,)    (3,)    scalar (3,)
```

### Strategy 4: Check Your Understanding

**Ask yourself:**
- What is the input?
- What is the output?
- What should happen step-by-step?
- What could go wrong?
- Does this make mathematical sense?

### Strategy 5: Debug Incrementally

**Test each piece:**
```python
def normalize(xs):
    # Test mean calculation
    mean = sum(xs) / len(xs)
    print(f"Mean: {mean}")  # Debug: check if correct
    
    # Test variance
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    print(f"Variance: {variance}")  # Debug
    
    # Test std
    std = variance ** 0.5
    print(f"Std: {std}")  # Debug
    
    # Test result
    result = [(x - mean) / std for x in xs]
    print(f"Result: {result}")  # Debug
    
    return result
```

---

## 📚 Resources for When You're Stuck

### For Python Basics:
- Python list comprehensions: `[x*2 for x in range(10)]`
- Error handling: `try/except`
- Built-in functions: `sum()`, `len()`, `abs()`

### For NumPy:
- NumPy array creation: `np.array([1, 2, 3])`
- Matrix multiplication: `A @ B`
- Broadcasting rules: Check NumPy docs
- Shape checking: `array.shape`, `array.ndim`

### For Math:
- Mean: `sum(x) / len(x)`
- Variance: `sum((x-mean)^2) / n`
- Standard deviation: `sqrt(variance)`
- Matrix multiplication rules

---

## ✅ Checklist: Before You Start Coding

- [ ] Do I understand what the function should do?
- [ ] Do I know the formula/method?
- [ ] Can I work through an example by hand?
- [ ] Do I know what edge cases to handle?
- [ ] Do I have a plan for implementation?

---

## 🎓 Remember

1. **It's okay to be stuck** - that's when you learn!
2. **Break problems into smaller pieces** - tackle one at a time
3. **Test as you go** - catch errors early
4. **Start simple** - get it working, then improve
5. **Ask for help** - after you've tried yourself
6. **Practice makes perfect** - each problem gets easier!

---

**Now you're ready! Pick a template and start solving. Good luck! 🚀**

