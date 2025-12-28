# Walkthrough Example: Normalize Function

**This is a complete walkthrough showing how to think through and solve Exercise 1.**

---

## 🎯 The Problem

**Implement a function that normalizes a list of numbers.**

Normalization means:
- Transform data to have mean = 0
- Transform data to have standard deviation = 1
- Formula: `normalized = (x - mean) / std`

---

## 🧠 Step 1: Understand What We Need

### What does "normalize" mean?

Let's work through an example by hand:

**Input:** `[1, 2, 3, 4]`

1. **Calculate mean:**
   ```
   mean = (1 + 2 + 3 + 4) / 4 = 10 / 4 = 2.5
   ```

2. **Calculate variance:**
   ```
   variance = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4
            = ((-1.5)² + (-0.5)² + (0.5)² + (1.5)²) / 4
            = (2.25 + 0.25 + 0.25 + 2.25) / 4
            = 5.0 / 4 = 1.25
   ```

3. **Calculate standard deviation:**
   ```
   std = √variance = √1.25 ≈ 1.118
   ```

4. **Normalize each value:**
   ```
   normalized[0] = (1 - 2.5) / 1.118 ≈ -1.341
   normalized[1] = (2 - 2.5) / 1.118 ≈ -0.447
   normalized[2] = (3 - 2.5) / 1.118 ≈ 0.447
   normalized[3] = (4 - 2.5) / 1.118 ≈ 1.341
   ```

5. **Verify (the result should have mean=0, std=1):**
   ```
   mean of normalized = (-1.341 + -0.447 + 0.447 + 1.341) / 4 ≈ 0 ✓
   std of normalized ≈ 1 ✓
   ```

**Great! Now we understand what we need to compute.**

---

## 🔍 Step 2: Identify the Components

Let's break down what we need to do:

1. **Input validation** - Check if the list is empty
2. **Calculate mean** - Sum divided by length
3. **Calculate variance** - Average of squared differences from mean
4. **Calculate standard deviation** - Square root of variance
5. **Handle edge cases** - What if std = 0?
6. **Apply formula** - For each value: (value - mean) / std

---

## 📝 Step 3: Write the Skeleton

Let's start with the function structure:

```python
def normalize(xs):
    """
    Z-score normalization: (x - mean) / std
    """
    # TODO: Implement this
    pass
```

---

## 🛠️ Step 4: Implement Piece by Piece

### Piece 1: Input Validation

**Think:** What if someone passes an empty list? We can't calculate mean of nothing!

```python
def normalize(xs):
    if not xs:  # Empty list is "falsy" in Python
        raise ValueError("Input list cannot be empty")
```

**Test it:**
```python
try:
    normalize([])
    print("ERROR: Should have raised ValueError!")
except ValueError:
    print("✓ Empty list handled correctly")
```

### Piece 2: Edge Case - Single Value

**Think:** What if there's only one value? Can we normalize it?
- Mean = that value
- Variance = 0 (no variation)
- Std = 0
- Formula: (value - value) / 0 = 0/0 = undefined!

**Solution:** Return [0.0] for single values (convention)

```python
def normalize(xs):
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    if len(xs) == 1:
        return [0.0]  # Single value normalized to 0
```

### Piece 3: Calculate Mean

**Think:** Mean is just average - sum divided by count.

```python
def normalize(xs):
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    if len(xs) == 1:
        return [0.0]
    
    # Calculate mean
    mean = sum(xs) / len(xs)
```

**Test it:**
```python
xs = [1, 2, 3, 4]
mean = sum(xs) / len(xs)
print(f"Mean: {mean}")  # Should be 2.5
```

### Piece 4: Calculate Variance

**Think:** 
- Variance = average of (value - mean)²
- We need: sum of squared differences, divided by length
- Use a list comprehension or generator expression

```python
def normalize(xs):
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    if len(xs) == 1:
        return [0.0]
    
    mean = sum(xs) / len(xs)
    
    # Calculate variance
    # Option 1: Using generator (memory efficient)
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    
    # Option 2: Using list comprehension (easier to read)
    # squared_diffs = [(x - mean) ** 2 for x in xs]
    # variance = sum(squared_diffs) / len(xs)
```

**Test it:**
```python
xs = [1, 2, 3, 4]
mean = 2.5
variance = sum((x - mean) ** 2 for x in xs) / len(xs)
print(f"Variance: {variance}")  # Should be 1.25
```

### Piece 5: Calculate Standard Deviation

**Think:** Standard deviation is just square root of variance.

```python
def normalize(xs):
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    if len(xs) == 1:
        return [0.0]
    
    mean = sum(xs) / len(xs)
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    
    # Calculate standard deviation
    std = variance ** 0.5  # or: math.sqrt(variance)
```

**Test it:**
```python
variance = 1.25
std = variance ** 0.5
print(f"Std: {std}")  # Should be ≈ 1.118
```

### Piece 6: Handle Division by Zero

**Think:** What if all values are the same?
- Variance = 0 (no variation)
- Std = 0
- We can't divide by 0!

**Solution:** If std = 0, return list of zeros

```python
def normalize(xs):
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    if len(xs) == 1:
        return [0.0]
    
    mean = sum(xs) / len(xs)
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = variance ** 0.5
    
    # Handle division by zero
    if std == 0:
        return [0.0] * len(xs)  # All values same, return zeros
```

**Test it:**
```python
result = normalize([5, 5, 5, 5])
print(f"Result: {result}")  # Should be [0.0, 0.0, 0.0, 0.0]
```

### Piece 7: Apply the Formula

**Think:** For each value, compute (value - mean) / std
- Use list comprehension: `[(x - mean) / std for x in xs]`

```python
def normalize(xs):
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    if len(xs) == 1:
        return [0.0]
    
    mean = sum(xs) / len(xs)
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = variance ** 0.5
    
    if std == 0:
        return [0.0] * len(xs)
    
    # Apply normalization formula
    return [(x - mean) / std for x in xs]
```

**Test it:**
```python
result = normalize([1, 2, 3, 4])
print(f"Result: {result}")

# Verify mean ≈ 0, std ≈ 1
mean_result = sum(result) / len(result)
variance_result = sum((x - mean_result) ** 2 for x in result) / len(result)
std_result = variance_result ** 0.5

print(f"Mean of result: {mean_result} (should be ~0)")
print(f"Std of result: {std_result} (should be ~1)")
```

---

## ✅ Step 5: Complete Implementation

Here's the complete function:

```python
def normalize(xs):
    """
    Z-score normalization: (x - mean) / std
    
    Args:
        xs: List of numbers to normalize
        
    Returns:
        List of normalized values
        
    Raises:
        ValueError: If input is empty or invalid
    """
    # Input validation
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    # Edge case: single value
    if len(xs) == 1:
        return [0.0]
    
    # Calculate mean
    mean = sum(xs) / len(xs)
    
    # Calculate variance
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    
    # Calculate standard deviation
    std = variance ** 0.5
    
    # Handle division by zero (all values are the same)
    if std == 0:
        return [0.0] * len(xs)
    
    # Apply normalization formula
    return [(x - mean) / std for x in xs]
```

---

## 🧪 Step 6: Test All Cases

Run through all test cases:

```python
# Test 1: Basic case
result = normalize([1, 2, 3, 4])
print(f"Test 1: {result}")
# Should give mean ≈ 0, std ≈ 1

# Test 2: Single value
result = normalize([5])
print(f"Test 2: {result}")  # Should be [0.0]

# Test 3: Empty list
try:
    normalize([])
    print("ERROR!")
except ValueError:
    print("Test 3: ✓ Empty list handled")

# Test 4: Negative numbers
result = normalize([-2, -1, 0, 1, 2])
print(f"Test 4: {result}")  # Should still work

# Test 5: All same values
result = normalize([5, 5, 5, 5])
print(f"Test 5: {result}")  # Should be [0.0, 0.0, 0.0, 0.0]
```

---

## 🎓 Key Takeaways

1. **Start simple** - Get basic case working first
2. **Add edge cases** - Handle empty, single, same values
3. **Test incrementally** - Check each piece as you build
4. **Use list comprehensions** - Clean and Pythonic
5. **Handle errors** - Think about what could go wrong
6. **Verify mathematically** - Check that mean=0, std=1

---

## 💡 Common Mistakes to Avoid

1. **Forgetting to handle empty list** - Will crash on `sum([])`
2. **Not handling std=0** - Division by zero error
3. **Using loops instead of comprehensions** - Less Pythonic
4. **Not testing edge cases** - Function might fail on special inputs
5. **Wrong variance formula** - Should divide by n, not n-1 (for population std)

---

**Now try it yourself with the template! You've got this! 🚀**

