# Simple Guide: How to Complete the Normalize Function

**This guide walks you through the SIMPLE version step-by-step.**

---

## 🎯 What You're Building

A function that takes numbers like `[1, 2, 3, 4]` and transforms them so:
- The mean becomes 0
- The standard deviation becomes 1

Result: `[-1.34, -0.45, 0.45, 1.34]` (approximately)

---

## 📝 Step-by-Step Instructions

### Step 1: Handle Empty List

**What to do:**
```python
if not xs:
    raise ValueError("Input list cannot be empty")
```

**Why:** If the list is empty, we can't calculate mean or std.

**Test it:**
```python
normalize([])  # Should raise ValueError
```

---

### Step 2: Handle Single Value

**What to do:**
```python
if len(xs) == 1:
    return [0.0]
```

**Why:** With only one number, there's no variation, so we return [0.0] by convention.

**Test it:**
```python
normalize([5])  # Should return [0.0]
```

---

### Step 3: Calculate Mean

**What to do:**
```python
mean = sum(xs) / len(xs)
```

**Example:**
- Input: `[1, 2, 3, 4]`
- Sum: `1 + 2 + 3 + 4 = 10`
- Count: `4`
- Mean: `10 / 4 = 2.5`

**Test it:**
```python
xs = [1, 2, 3, 4]
mean = sum(xs) / len(xs)
print(mean)  # Should print 2.5
```

---

### Step 4: Calculate Variance

**What to do:**
```python
variance = sum((x - mean) ** 2 for x in xs) / len(xs)
```

**What this does:**
1. For each number `x`, calculate `(x - mean) ** 2`
2. Sum all those squared differences
3. Divide by the count

**Example with [1, 2, 3, 4] and mean=2.5:**
```
(1-2.5)² = (-1.5)² = 2.25
(2-2.5)² = (-0.5)² = 0.25
(3-2.5)² = (0.5)² = 0.25
(4-2.5)² = (1.5)² = 2.25
Sum = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
Variance = 5.0 / 4 = 1.25
```

**Test it:**
```python
xs = [1, 2, 3, 4]
mean = 2.5
variance = sum((x - mean) ** 2 for x in xs) / len(xs)
print(variance)  # Should print 1.25
```

---

### Step 5: Calculate Standard Deviation

**What to do:**
```python
std = variance ** 0.5
```

**Why:** Standard deviation is just the square root of variance.

**Example:**
- Variance: `1.25`
- Std: `√1.25 ≈ 1.118`

**Test it:**
```python
variance = 1.25
std = variance ** 0.5
print(std)  # Should print approximately 1.118
```

---

### Step 6: Handle Division by Zero

**What to do:**
```python
if std == 0:
    return [0.0] * len(xs)
```

**Why:** If all numbers are the same (like [5, 5, 5, 5]), variance = 0, so std = 0. We can't divide by zero!

**Example:**
- Input: `[5, 5, 5, 5]`
- Mean: `5`
- Variance: `0` (no variation)
- Std: `0`
- Can't divide by 0, so return `[0.0, 0.0, 0.0, 0.0]`

**Test it:**
```python
normalize([5, 5, 5, 5])  # Should return [0.0, 0.0, 0.0, 0.0]
```

---

### Step 7: Apply the Formula

**What to do:**
```python
return [(x - mean) / std for x in xs]
```

**What this does:**
- For each number `x` in the list:
  - Calculate `(x - mean) / std`
  - This is the normalized value
- Return all normalized values as a list

**Example with [1, 2, 3, 4], mean=2.5, std≈1.118:**
```
(1 - 2.5) / 1.118 ≈ -1.34
(2 - 2.5) / 1.118 ≈ -0.45
(3 - 2.5) / 1.118 ≈ 0.45
(4 - 2.5) / 1.118 ≈ 1.34
Result: [-1.34, -0.45, 0.45, 1.34]
```

**Test it:**
```python
result = normalize([1, 2, 3, 4])
print(result)  # Should print approximately [-1.34, -0.45, 0.45, 1.34]
```

---

## ✅ Complete Solution

Here's what your complete function should look like:

```python
def normalize(xs):
    # Step 1: Check if empty
    if not xs:
        raise ValueError("Input list cannot be empty")
    
    # Step 2: Handle single value
    if len(xs) == 1:
        return [0.0]
    
    # Step 3: Calculate mean
    mean = sum(xs) / len(xs)
    
    # Step 4: Calculate variance
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    
    # Step 5: Calculate standard deviation
    std = variance ** 0.5
    
    # Step 6: Handle division by zero
    if std == 0:
        return [0.0] * len(xs)
    
    # Step 7: Apply formula
    return [(x - mean) / std for x in xs]
```

---

## 🧪 Testing Checklist

After implementing, test these cases:

- [ ] `normalize([1, 2, 3, 4])` → Mean ≈ 0, Std ≈ 1
- [ ] `normalize([5])` → Returns `[0.0]`
- [ ] `normalize([])` → Raises `ValueError`
- [ ] `normalize([5, 5, 5, 5])` → Returns `[0.0, 0.0, 0.0, 0.0]`
- [ ] `normalize([-2, -1, 0, 1, 2])` → Works with negative numbers

---

## 💡 Tips

1. **Work step by step** - Don't try to do everything at once
2. **Test each step** - Verify each piece works before moving on
3. **Use print statements** - Print intermediate values to see what's happening
4. **Read error messages** - They tell you what's wrong!
5. **Take breaks** - If stuck, step away and come back fresh

---

## 🆘 If You're Stuck

1. **Read the error message** - It tells you what's wrong
2. **Check each step** - Make sure you completed all 7 steps
3. **Test with simple input** - Try `[1, 2, 3]` first
4. **Print intermediate values** - See what mean, variance, std are
5. **Compare with the example** - Work through the math by hand

---

**You've got this! Take it one step at a time! 🚀**

