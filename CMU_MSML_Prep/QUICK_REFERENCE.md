# CMU MSML Prep - Quick Reference

**One-page cheat sheet for all problems and their CMU connections.**

---

## Problem Index

### Foundations

| Problem | Difficulty | CMU Courses | Key Concept |
|---------|-----------|-------------|-------------|
| Matrix × Vector | ⭐ | All | Linear algebra foundation |
| Gradient Descent | ⭐⭐ | 10-725, 10-701, 10-617 | Optimization |
| ReLU | ⭐ | 10-617/707 | Activation, sparsity |
| Softmax | ⭐⭐ | 10-617/707, 10-701 | Classification |
| Feature Scaling | ⭐ | 10-718, 10-701 | Preprocessing |
| One-Hot Encoding | ⭐ | 10-718, 10-701 | Categorical data |
| Accuracy Score | ⭐ | 10-701, 10-718 | Evaluation |

---

## CMU Course → Problems

### 10-701/715: Introduction to ML
- Linear Regression (GD & Normal)
- Evaluation Metrics
- Feature Scaling
- One-Hot Encoding

### 10-617/707: Deep Learning
- ReLU
- Softmax
- Matrix Operations
- Gradient Descent

### 10-718: ML in Practice
- Feature Scaling
- One-Hot Encoding
- Evaluation Metrics
- Complete Pipeline

### 10-725: Optimization
- Gradient Descent
- Matrix Operations
- Linear Regression

---

## Research Connections

### Dr. Shah's Research:
- **Evaluation Science** → Accuracy Score, Evaluation Metrics
- **Annotation Bias** → One-Hot Encoding, Feature Scaling
- **Reproducibility** → Gradient Descent, Evaluation

---

## Key Formulas

### Gradient Descent:
```
theta = theta - alpha * gradient
```

### ReLU:
```
f(x) = max(0, x)
```

### Softmax:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

### Min-Max Scaling:
```
scaled = (x - min) / (max - min)
```

### Z-Score:
```
standardized = (x - mean) / std
```

### Accuracy:
```
accuracy = correct / total
```

---

## Common Mistakes

1. **Feature Scaling**: Using test statistics (use train stats!)
2. **Gradient Descent**: Learning rate too high/low
3. **Softmax**: Forgetting exponential
4. **Evaluation**: Using accuracy for imbalanced data
5. **One-Hot**: Applying to already-encoded data

---

## Study Order

1. Matrix Operations (foundation)
2. Gradient Descent (optimization)
3. Activation Functions (ReLU, Softmax)
4. Preprocessing (Scaling, Encoding)
5. Evaluation (Metrics)

---

## File Locations

- **Foundations**: `01_Foundations/`
- **Course-Specific**: `02_Course_Specific/`
- **Practice**: `03_Practice_Problems/`
- **Research**: `04_Research_Alignment/`

---

**Print this and keep it handy while studying!**

