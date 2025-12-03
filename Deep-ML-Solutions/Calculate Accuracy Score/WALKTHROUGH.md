# Accuracy Score - Step-by-Step Walkthrough

## Understanding the Problem

**Goal**: Calculate how accurate a model's predictions are by comparing predicted labels with true labels.

**Formula**: Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

## Step-by-Step Breakdown

### Step 1: Convert Inputs to NumPy Arrays
```python
y_true = np.array(y_true)
y_pred = np.array(y_pred)
```
**Why?** 
- Ensures we're working with NumPy arrays (even if lists are passed in)
- Enables vectorized operations (faster and cleaner code)
- Makes element-wise comparison easier

### Step 2: Validate Inputs
```python
assert len(y_true) == len(y_pred), "Length of true labels and predicted labels must be the same."
```
**Why?**
- We can't compare predictions if the arrays have different lengths
- Catches errors early before computation
- `assert` raises an error if condition is False

### Step 3: Count Correct Predictions
```python
correct_predictions = np.sum(y_true == y_pred)
```
**How it works:**
- `y_true == y_pred` creates a boolean array: `[True, False, True, ...]`
  - `True` where predictions match
  - `False` where they don't
- `np.sum()` counts `True` values (True = 1, False = 0)

**Example:**
```python
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 1, 1, 0, 0]
# y_true == y_pred = [True, False, True, False, True]
# np.sum([True, False, True, False, True]) = 3
```

### Step 4: Calculate Accuracy
```python
accuracy = correct_predictions / len(y_true)
```
**Why?**
- Divide correct predictions by total predictions
- Result is a float between 0.0 and 1.0
- 1.0 = 100% accurate, 0.0 = 0% accurate

## Complete Example

```python
import numpy as np

# Example: Binary classification
y_true = [1, 0, 1, 1, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1]

# Step 1: Convert to arrays
y_true = np.array(y_true)  # [1, 0, 1, 1, 0, 1, 0]
y_pred = np.array(y_pred)  # [1, 0, 1, 0, 0, 1, 1]

# Step 2: Validate (both length 7, so passes)

# Step 3: Count matches
# [1==1, 0==0, 1==1, 1==0, 0==0, 1==1, 0==1]
# [True, True, True, False, True, True, False]
correct_predictions = np.sum([True, True, True, False, True, True, False])  # = 5

# Step 4: Calculate
accuracy = 5 / 7 = 0.714... ≈ 71.4%
```

## Alternative Approaches (Simpler)

### Approach 1: Using a Loop (Easier to Understand)
```python
def accuracy_score_simple(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)
```

### Approach 2: Using List Comprehension
```python
def accuracy_score_list(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)
```

## Key Concepts

1. **Vectorization**: NumPy operations work on entire arrays at once (faster)
2. **Boolean Arrays**: Comparisons create arrays of True/False values
3. **Type Conversion**: NumPy automatically handles different input types
4. **Assertions**: Use to validate inputs and catch errors early

## Practice Exercise

Try modifying the function to:
1. Handle empty arrays gracefully
2. Return accuracy as a percentage (0-100) instead of decimal (0-1)
3. Add support for multi-class classification (already works, but verify!)

