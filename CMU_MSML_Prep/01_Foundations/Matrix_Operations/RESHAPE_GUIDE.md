# Reshape Matrix - Guided Walkthrough

## What is Reshaping?

**Reshaping** = Changing matrix dimensions while keeping all the same elements.

Think of it like rearranging items in a box - same items, different arrangement!

---

## Key Rule

**original_rows × original_cols = new_rows × new_cols**

**Must have same total number of elements!**

### ✅ Valid:
- 2×6 = 12 elements → 3×4 = 12 elements ✓
- 3×4 = 12 elements → 2×6 = 12 elements ✓

### ❌ Invalid:
- 2×6 = 12 elements → 3×5 = 15 elements ✗
- 2×3 = 6 elements → 4×2 = 8 elements ✗

---

## How It Works

### Reading Order: Row-Major
Elements are read **row-by-row**, left to right, top to bottom.

**Example:**
```
Original (2×3):
  [1, 2, 3]
  [4, 5, 6]

Reading order: 1 → 2 → 3 → 4 → 5 → 6

Reshaped to (3×2):
  [1, 2]
  [3, 4]
  [5, 6]

Same elements, different arrangement!
```

---

## Common Use Cases

### 1. **CNN to Fully Connected Layer**

**Problem:**
- CNN outputs: (batch, height, width) = (100, 7, 7)
- Fully connected layer needs: (batch, features) = (100, 49)

**Solution:**
```
Reshape (100, 7, 7) → (100, 49)
Each 7×7 image → 49-element vector
Ready for fully connected layer!
```

### 2. **Image Flattening**

**Problem:**
- Image: 28×28 pixels = 784 elements
- Neural network needs: 1D input

**Solution:**
```
Reshape (28, 28) → (1, 784) or (784, 1)
Flatten image for neural network
```

### 3. **Batch Processing**

**Problem:**
- Have: 1000 samples, each with 10 features = (1000, 10)
- Need: Process in batches of 100

**Solution:**
```
Reshape to organize into batches
Or use batch processing functions
```

---

## Step-by-Step Example

### Example: 2×6 → 3×4

**Original (2×6):**
```
[1,  2,  3,  4,  5,  6]
[7,  8,  9, 10, 11, 12]
```

**Step 1:** Read elements row-by-row
```
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12
```

**Step 2:** Rearrange into 3 rows, 4 columns
```
Row 1: 1, 2, 3, 4
Row 2: 5, 6, 7, 8
Row 3: 9, 10, 11, 12
```

**Result (3×4):**
```
[1,  2,  3,  4]
[5,  6,  7,  8]
[9, 10, 11, 12]
```

---

## Why It Matters in Neural Networks

### CNN Architecture:
```
Input Image (28×28)
  ↓
Convolutional Layers
  ↓
Feature Maps (7×7)
  ↓
RESHAPE (7×7 → 49)
  ↓
Fully Connected Layer
  ↓
Output
```

**Without reshaping:** Can't connect CNN to fully connected layer!
**With reshaping:** Seamless connection!

---

## CMU Course Connection

### 10-617/707 (Deep Learning):
- Tensor operations
- CNN architectures
- Data reshaping for neural networks
- Batch processing

### 10-725 (Optimization):
- Matrix operations
- Computational efficiency
- Data organization

---

## Common Mistakes

### Mistake 1: Wrong Total Elements
```
Original: 2×3 = 6 elements
Try: Reshape to 3×3 = 9 elements ✗
Problem: Can't create elements from nothing!
```

### Mistake 2: Wrong Reading Order
```
Thinking column-major instead of row-major
Problem: Elements in wrong order!
```

### Mistake 3: Forgetting Batch Dimension
```
Reshaping (batch, height, width) incorrectly
Problem: Batch dimension matters!
```

---

## Key Takeaways

1. **Reshaping changes dimensions, keeps elements**
2. **Key rule: Total elements must match**
3. **Elements read row-by-row (row-major)**
4. **Essential for CNN → FC layer connection**
5. **Used in batch processing and data prep**

---

## Practice

Try these:
1. Reshape (2×4) to (4×2)
2. Reshape (1×8) to (2×4)
3. Reshape (3×3) to (9×1) - flatten!

**Remember:** Count total elements first!

