# Foundations - Core ML Concepts

**Purpose:** Master the fundamental building blocks of machine learning.

**Order of Study:**
1. Matrix Operations (foundation for everything)
2. Gradient Descent (optimization foundation)
3. Activation Functions (ReLU, Softmax)
4. Preprocessing (Feature Scaling, One-Hot Encoding)
5. Evaluation (Accuracy, Metrics)

---

## 1. Matrix Operations

**Why:** Foundation for all linear algebra in ML
**CMU Courses:** 10-725, 10-617/707, All courses
**Difficulty:** ⭐

**Files:**
- `Matrix_times_Vector/` - Complete walkthrough with examples

**Key Concepts:**
- Matrix-vector multiplication
- Dimensions must match
- Used in neural networks, linear regression

---

## 2. Gradient Descent

**Why:** Foundation for optimization, used everywhere
**CMU Courses:** 10-725, 10-701/715, 10-617/707
**Difficulty:** ⭐⭐

**Files:**
- `Gradient_Descent/` - Conceptual deep dive + implementation

**Key Concepts:**
- Hill climbing analogy
- Learning rate (alpha)
- Update rule: theta = theta - alpha * gradient
- Convergence

---

## 3. Activation Functions

### ReLU
**Why:** Most common activation function
**CMU Courses:** 10-617/707 (Deep Learning)
**Difficulty:** ⭐

**Files:**
- `ReLU/` - Complete walkthrough
- Why sparsity matters
- Why zeroing negatives is usually OK

**Key Concepts:**
- Adds non-linearity
- Creates sparsity
- Prevents vanishing gradients

### Softmax
**Why:** Used in classification, transformers
**CMU Courses:** 10-617/707, 10-701/715
**Difficulty:** ⭐⭐

**Files:**
- `Softmax/` - Complete walkthrough

**Key Concepts:**
- Converts scores to probabilities
- Sums to 1.0
- Used in final layer of classifiers

---

## 4. Preprocessing

### Feature Scaling
**Why:** Essential for gradient descent, distance algorithms
**CMU Courses:** 10-718, 10-701/715
**Difficulty:** ⭐

**Files:**
- `Feature_Scaling/` - High-level overview

**Key Concepts:**
- Min-Max scaling: [0, 1] range
- Z-Score: Mean=0, Std=1
- When to use each

### One-Hot Encoding
**Why:** Convert categories to numbers
**CMU Courses:** 10-718, 10-701/715
**Difficulty:** ⭐

**Files:**
- `One_Hot_Encoding/` - Complete walkthrough

**Key Concepts:**
- Categorical to numerical
- No ordering implied
- Used in preprocessing

---

## 5. Evaluation Metrics

**Why:** How we measure ML success
**CMU Courses:** 10-701/715, 10-718
**Difficulty:** ⭐

**Files:**
- `Evaluation_Metrics/` - CMU-focused overview

**Key Concepts:**
- Accuracy (simple but limited)
- Precision & Recall
- Confusion Matrix
- Evaluation pitfalls

---

## Study Plan

### Week 1-2: Matrix Operations & Gradient Descent
- Master matrix-vector multiplication
- Understand gradient descent conceptually
- Implement from scratch

### Week 3-4: Activation Functions
- ReLU (simple, understand sparsity)
- Softmax (classification)
- When to use each

### Week 5-6: Preprocessing
- Feature scaling (when and why)
- One-hot encoding (categorical data)
- Complete preprocessing pipeline

### Week 7-8: Evaluation
- Accuracy and its limitations
- Precision, recall, F1
- Evaluation best practices

---

## Practice Tips

1. **Implement from scratch** - Don't just read, code it!
2. **Understand the math** - Know why it works
3. **Connect to CMU courses** - See how it fits
4. **Think about research** - How does this relate to evaluation science?

---

## Mastery Checklist

- [ ] Can implement matrix-vector multiplication
- [ ] Understand gradient descent conceptually
- [ ] Can explain why ReLU creates sparsity
- [ ] Understand when to use Softmax
- [ ] Know when feature scaling is needed
- [ ] Can implement one-hot encoding
- [ ] Understand evaluation metrics and their trade-offs

