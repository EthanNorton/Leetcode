# Side-by-Side: Template vs Solution

**Use this to compare your work with the solution!**

---

## 📁 File Locations

### Template (Your Starting Point):
```
Templates/02_NumPy/exercise_2_matrix_ops_template.py
```
- Has `None` placeholders
- Has TODO comments
- Tests are commented out

### Solution (Reference):
```
00_Skill_Exercises/02_NumPy/exercise_2_matrix_ops.py
```
- Fully implemented
- All functions complete
- Tests run automatically

---

## 🔍 Function-by-Function Comparison

### Function 1: `matrix_vector_multiplication()`

**Template (What you fill in):**
```python
def matrix_vector_multiplication():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    w = np.array([0.5, -1.0])  # (2,)
    b = 0.2  # scalar
    
    # TODO: Compute y = X @ w + b
    y = None  # Replace with: X @ w + b
    
    return y
```

**Solution (Complete):**
```python
def matrix_vector_multiplication():
    X = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    w = np.array([0.5, -1.0])  # (2,)
    b = 0.2  # scalar
    
    # Compute: y = X @ w + b
    y = X @ w + b  # Matrix multiply, then broadcast bias
    
    return y
```

**Key:** Just use `X @ w + b` - NumPy handles broadcasting automatically!

---

### Function 2: `vectorized_operations()`

**Template:**
```python
def vectorized_operations():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    
    # TODO: Compute WITHOUT loops
    sum_result = None  # Your code
    product_result = None  # Your code
    power_result = None  # Your code
    dot_product = None  # Your code
    
    return {
        'sum': sum_result,
        'product': product_result,
        'power': power_result,
        'dot': dot_product
    }
```

**Solution:**
```python
def vectorized_operations():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    
    # All vectorized - no loops!
    sum_result = a + b           # Element-wise sum
    product_result = a * b        # Element-wise product
    power_result = a ** 2         # Square each element
    dot_product = a @ b           # Dot product (or np.dot(a, b))
    
    return {
        'sum': sum_result,
        'product': product_result,
        'power': power_result,
        'dot': dot_product
    }
```

**Key:** NumPy operations are element-wise by default. Use `@` for dot product!

---

### Function 3: `broadcasting_example()`

**Template:**
```python
def broadcasting_example():
    matrix = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    vector = np.array([10, 20])  # (2,)
    
    # TODO: Add vector to each row using broadcasting
    result = None  # Your code: matrix + vector
    
    return result
```

**Solution:**
```python
def broadcasting_example():
    matrix = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    vector = np.array([10, 20])  # (2,)
    
    # Broadcasting automatically expands vector to match matrix
    result = matrix + vector  # NumPy handles the expansion!
    
    return result
```

**Key:** Just add them! NumPy automatically broadcasts the vector to each row.

---

### Function 4: `matrix_operations()`

**Template:**
```python
def matrix_operations():
    A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    B = np.array([[7, 8], [9, 10], [11, 12]])  # (3, 2)
    
    # TODO: Compute
    matmul = None  # A @ B
    transpose = None  # A.T
    sum_cols = None  # np.sum(A, axis=0)
    sum_rows = None  # np.sum(A, axis=1)
    
    return {
        'matmul': matmul,
        'transpose': transpose,
        'sum_cols': sum_cols,
        'sum_rows': sum_rows
    }
```

**Solution:**
```python
def matrix_operations():
    A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    B = np.array([[7, 8], [9, 10], [11, 12]])  # (3, 2)
    
    matmul = A @ B                    # Matrix multiplication
    transpose = A.T                   # Transpose
    sum_cols = np.sum(A, axis=0)      # Sum along columns (axis=0)
    sum_rows = np.sum(A, axis=1)      # Sum along rows (axis=1)
    
    return {
        'matmul': matmul,
        'transpose': transpose,
        'sum_cols': sum_cols,
        'sum_rows': sum_rows
    }
```

**Key:** 
- `@` for matrix multiply
- `.T` for transpose
- `axis=0` = columns, `axis=1` = rows

---

## 💡 Quick Reference

### NumPy Operations Cheat Sheet

| Operation | Syntax | Example |
|-----------|--------|---------|
| Matrix multiply | `A @ B` | `X @ w` |
| Element-wise multiply | `A * B` | `a * b` |
| Element-wise add | `A + B` | `a + b` |
| Transpose | `A.T` | `X.T` |
| Sum along axis | `np.sum(A, axis=0)` | Sum columns |
| Dot product | `a @ b` | `np.array([1,2]) @ np.array([3,4])` |

### Shape Rules

- Matrix multiply: `(m, n) @ (n, p) → (m, p)`
- Broadcasting: Shapes must be compatible from the right
- Transpose: `(m, n).T → (n, m)`

---

## 🎯 How to Use This

1. **Open both files side-by-side in your editor**
   - Template: `Templates/02_NumPy/exercise_2_matrix_ops_template.py`
   - Solution: `00_Skill_Exercises/02_NumPy/exercise_2_matrix_ops.py`

2. **Try implementing in the template first**
   - Don't peek at solution until you've tried!

3. **Compare your solution**
   - See if you got it right
   - Learn from differences
   - Understand the approach

4. **Test your code**
   - Uncomment tests in template
   - Run and verify

---

## ✅ Checklist

After completing each function, check:

- [ ] Function 1: `matrix_vector_multiplication()` - Uses `@` and `+`
- [ ] Function 2: `vectorized_operations()` - No loops, all NumPy ops
- [ ] Function 3: `broadcasting_example()` - Simple addition, broadcasting works
- [ ] Function 4: `matrix_operations()` - All operations correct

---

**Remember:** Try first, then compare! That's how you learn! 🚀



