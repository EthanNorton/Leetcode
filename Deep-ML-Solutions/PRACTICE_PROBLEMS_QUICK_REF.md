# CMU Practice Problems - Quick Reference

## 🎯 Problem Categories by Course

### **10-701/715: Introduction to ML**
- ✅ Matrix operations (multiplication, reshaping)
- ✅ Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- ✅ Linear regression (normal equation, gradient descent)
- ✅ Bias-variance tradeoff
- ✅ Overfitting and regularization

### **10-617/707: Deep Learning**
- ✅ Activation functions (ReLU, sigmoid, softmax)
- ✅ Neural network forward/backward pass
- ✅ Convolutional operations
- ✅ Backpropagation calculations

### **10-725: Optimization**
- ✅ Gradient descent variants (batch, mini-batch, stochastic)
- ✅ Momentum
- ✅ Learning rate scheduling
- ✅ Convergence analysis

### **10-718: ML in Practice**
- ✅ Feature scaling (min-max, standardization)
- ✅ Missing data handling
- ✅ Categorical encoding (one-hot, label)
- ✅ Data preprocessing pipelines

### **36-700/705: Probability & Statistics**
- ✅ Bayes' theorem
- ✅ Maximum likelihood estimation
- ✅ Probability distributions
- ✅ Statistical inference

---

## 📊 Difficulty Distribution

| Difficulty | Count | Focus Areas |
|-----------|-------|-------------|
| ⭐ Beginner | 8 | Fundamentals, basic implementations |
| ⭐⭐ Intermediate | 12 | Standard algorithms, applications |
| ⭐⭐⭐ Advanced | 5 | Theory, derivations, optimizations |

---

## 🎓 Exam Preparation Strategy

### **Week 1-2: Fundamentals**
- Master matrix operations
- Understand evaluation metrics
- Practice basic implementations

### **Week 3-4: Core Algorithms**
- Linear regression (both methods)
- Neural network forward pass
- Activation functions

### **Week 5-6: Advanced Topics**
- Backpropagation
- Optimization variants
- Regularization

### **Week 7: Review & Practice**
- Work through all problems
- Time yourself on exam-style questions
- Review theory explanations

---

## 🔑 Key Formulas to Memorize

### **Linear Regression**
- Normal Equation: `θ = (X^T X)^(-1) X^T y`
- Gradient: `∇J = (1/m) X^T (Xθ - y)`
- Update: `θ = θ - α∇J`

### **Activation Functions**
- ReLU: `f(x) = max(0, x)`
- Sigmoid: `f(x) = 1/(1 + e^(-x))`
- Softmax: `f(x_i) = e^(x_i) / Σe^(x_j)`

### **Evaluation Metrics**
- Accuracy: `(TP + TN) / (TP + TN + FP + FN)`
- Precision: `TP / (TP + FP)`
- Recall: `TP / (TP + FN)`
- F1: `2 × (Precision × Recall) / (Precision + Recall)`

### **Probability**
- Bayes' Theorem: `P(A|B) = P(B|A) × P(A) / P(B)`
- MLE for Normal Mean: `μ = (1/n) Σx_i`

---

## 💡 Common Exam Question Types

1. **Implementation:** Code an algorithm from scratch
2. **Theory:** Explain why/how something works
3. **Calculation:** Manual computation (e.g., gradient, probability)
4. **Comparison:** Compare two methods/algorithms
5. **Debugging:** Find and fix bugs in code
6. **Application:** Choose appropriate method for a scenario

---

## 📚 Study Resources

- **Full Problems:** See `CMU_PRACTICE_PROBLEMS.md`
- **CMU Course Sites:** Check for past exams
- **Textbooks:** Bishop, Goodfellow, Boyd
- **Practice:** Implement all problems yourself

---

**Start with ⭐ problems, then move to ⭐⭐ and ⭐⭐⭐!**

