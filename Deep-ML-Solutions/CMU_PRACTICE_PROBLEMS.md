# CMU MSML Practice Problems: Homework & Exam Style Questions

**Purpose:** Practice problems similar to what you'll encounter in CMU MSML homework assignments and exams.

**Organization:** Problems are grouped by course and difficulty level (⭐ = Beginner, ⭐⭐ = Intermediate, ⭐⭐⭐ = Advanced)

---

## 📚 10-701/715: Introduction to Machine Learning

### Problem Set 1: Linear Algebra & Matrix Operations

#### Problem 1.1: Matrix Multiplication Dimensions ⭐
**Question:** Given matrix A (3×4) and matrix B (4×5), what are the dimensions of A × B? What about B × A?

**Answer:**
- A × B: (3×4) × (4×5) = (3×5) ✓
- B × A: (4×5) × (3×4) = **Invalid!** (5 ≠ 3) ✗

**CMU Connection:** Foundation for understanding neural network forward pass.

---

#### Problem 1.2: Implement Matrix-Vector Multiplication ⭐⭐
**Question:** Implement matrix-vector multiplication from scratch (no NumPy). Given:
```python
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]  # 3×3 matrix

b = [1, 0, 1]    # 3×1 vector
```
Calculate A × b manually and verify with code.

**Solution:**
```python
def matrix_vector_mult(A, b):
    if len(A[0]) != len(b):
        return None  # Dimension mismatch
    
    result = []
    for row in A:
        dot_product = sum(row[i] * b[i] for i in range(len(b)))
        result.append(dot_product)
    return result

# Manual calculation:
# Row 1: 1×1 + 2×0 + 3×1 = 4
# Row 2: 4×1 + 5×0 + 6×1 = 10
# Row 3: 7×1 + 8×0 + 9×1 = 16
# Result: [4, 10, 16]
```

**CMU Connection:** Core operation in linear regression and neural networks.

---

#### Problem 1.3: Reshape for Neural Network Input ⭐⭐
**Question:** You have 100 images, each 28×28 pixels. You need to feed them to a fully connected layer that expects input of shape (batch_size, 784). Write code to reshape the data.

**Solution:**
```python
import numpy as np

# Original shape: (100, 28, 28)
images = np.random.rand(100, 28, 28)

# Reshape to: (100, 784)
# 28 × 28 = 784 pixels per image
reshaped = images.reshape(100, 28 * 28)
# Or: images.reshape(100, -1)  # -1 auto-calculates

print(f"Original shape: {images.shape}")
print(f"Reshaped: {reshaped.shape}")  # (100, 784)
```

**CMU Connection:** Essential for preparing data for neural networks (10-617/707).

---

### Problem Set 2: Evaluation Metrics

#### Problem 2.1: Calculate Precision, Recall, F1 ⭐⭐
**Question:** Given the following confusion matrix:
```
                Predicted
              Positive  Negative
Actual Pos       80       20
      Neg        10       90
```
Calculate: Accuracy, Precision, Recall, F1-Score.

**Solution:**
```python
def calculate_metrics(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# From confusion matrix:
# TP = 80, FP = 10, FN = 20, TN = 90
metrics = calculate_metrics(80, 10, 20, 90)
# Accuracy = 170/200 = 0.85
# Precision = 80/90 = 0.889
# Recall = 80/100 = 0.80
# F1 = 2 × (0.889 × 0.80) / (0.889 + 0.80) = 0.842
```

**CMU Connection:** Standard evaluation in 10-701/715 and 10-718.

---

#### Problem 2.2: ROC Curve and AUC ⭐⭐⭐
**Question:** Given predictions and true labels, implement code to calculate ROC curve points and AUC.

**Solution:**
```python
import numpy as np
from sklearn.metrics import roc_curve, auc

def calculate_roc_auc(y_true, y_scores):
    """
    Calculate ROC curve and AUC.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities or scores
    
    Returns:
        fpr, tpr, thresholds, auc_score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_score

# Example usage
y_true = [0, 0, 1, 1, 1, 0, 1, 0]
y_scores = [0.1, 0.3, 0.4, 0.6, 0.7, 0.2, 0.8, 0.5]

fpr, tpr, thresholds, auc_score = calculate_roc_auc(y_true, y_scores)
print(f"AUC: {auc_score}")
```

**CMU Connection:** Advanced evaluation metric in 10-701/715.

---

### Problem Set 3: Linear Regression

#### Problem 3.1: Normal Equation Derivation ⭐⭐⭐
**Question:** Derive the normal equation for linear regression: θ = (X^T X)^(-1) X^T y

**Solution Steps:**
1. Start with cost function: J(θ) = (1/2m) Σ(h_θ(x^(i)) - y^(i))²
2. In matrix form: J(θ) = (1/2m)(Xθ - y)^T (Xθ - y)
3. Take derivative: ∇_θ J(θ) = (1/m) X^T (Xθ - y)
4. Set to zero: X^T Xθ = X^T y
5. Solve: θ = (X^T X)^(-1) X^T y

**CMU Connection:** Analytical solution in 10-701/715 and 10-725.

---

#### Problem 3.2: Gradient Descent Implementation ⭐⭐
**Question:** Implement gradient descent for linear regression with:
- Learning rate α = 0.01
- 1000 iterations
- Feature scaling

**Solution:**
```python
import numpy as np

def gradient_descent(X, y, alpha=0.01, iterations=1000):
    """
    Implement gradient descent for linear regression.
    
    Args:
        X: Feature matrix (m×n)
        y: Target vector (m×1)
        alpha: Learning rate
        iterations: Number of iterations
    
    Returns:
        theta: Learned parameters
        cost_history: Cost at each iteration
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(m, 1)
    cost_history = []
    
    for i in range(iterations):
        # Hypothesis
        h = X @ theta
        
        # Cost
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Gradient
        gradient = (1/m) * X.T @ (h - y)
        
        # Update
        theta = theta - alpha * gradient
    
    return theta, cost_history

# Feature scaling (important!)
def feature_scaling(X):
    """Normalize features to [0, 1]"""
    X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_scaled
```

**CMU Connection:** Core optimization algorithm in 10-725 and 10-617/707.

---

#### Problem 3.3: Bias-Variance Tradeoff ⭐⭐⭐
**Question:** Explain the bias-variance tradeoff. Given a dataset, how would you diagnose if your model has high bias or high variance? What solutions would you propose?

**Answer:**
- **High Bias (Underfitting):**
  - Symptoms: High training error, high test error
  - Solutions: Add features, increase model complexity, reduce regularization
  
- **High Variance (Overfitting):**
  - Symptoms: Low training error, high test error
  - Solutions: More training data, regularization, reduce model complexity, dropout

**CMU Connection:** Fundamental concept in 10-701/715.

---

## 🧠 10-617/707: Deep Learning

### Problem Set 4: Activation Functions

#### Problem 4.1: ReLU vs Sigmoid ⭐⭐
**Question:** Compare ReLU and Sigmoid activation functions. When would you use each? Implement both and show their outputs for the same input.

**Solution:**
```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Test on range
x = np.linspace(-5, 5, 100)
y_relu = relu(x)
y_sigmoid = sigmoid(x)

# ReLU: 
# - Pros: Fast, no vanishing gradient for positive values, sparse
# - Cons: Dying ReLU problem (outputs 0 for negative inputs)
# - Use: Hidden layers in most modern networks

# Sigmoid:
# - Pros: Smooth, bounded (0,1), good for probabilities
# - Cons: Vanishing gradient, computationally expensive
# - Use: Output layer for binary classification
```

**CMU Connection:** Core topic in 10-617/707.

---

#### Problem 4.2: Softmax for Multi-Class Classification ⭐⭐
**Question:** Implement softmax function. Given logits [2.0, 1.0, 0.1], calculate the probability distribution.

**Solution:**
```python
import numpy as np

def softmax(logits):
    """
    Compute softmax probabilities.
    
    Args:
        logits: Array of raw scores
    
    Returns:
        Probability distribution (sums to 1)
    """
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

# Example
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Probabilities: {probs}")  # Should sum to 1.0
print(f"Sum: {np.sum(probs)}")    # 1.0

# Manual calculation:
# exp(2.0) = 7.39, exp(1.0) = 2.72, exp(0.1) = 1.11
# Sum = 11.22
# Probs: [0.659, 0.242, 0.099]
```

**CMU Connection:** Essential for multi-class classification in 10-617/707.

---

### Problem Set 5: Neural Network Forward Pass

#### Problem 5.1: Implement 2-Layer Neural Network ⭐⭐⭐
**Question:** Implement a 2-layer neural network from scratch:
- Input: 3 features
- Hidden layer: 4 neurons with ReLU
- Output: 1 neuron with sigmoid (binary classification)

**Solution:**
```python
import numpy as np

class TwoLayerNN:
    def __init__(self, input_size=3, hidden_size=4, output_size=1):
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip for stability
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input features (m×3)
        
        Returns:
            output: Predictions (m×1)
        """
        # Layer 1: Input -> Hidden
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)  # ReLU activation
        
        # Layer 2: Hidden -> Output
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)  # Sigmoid for binary classification
        
        return a2

# Test
nn = TwoLayerNN()
X = np.random.randn(10, 3)  # 10 examples, 3 features
output = nn.forward(X)
print(f"Output shape: {output.shape}")  # (10, 1)
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")  # [0, 1]
```

**CMU Connection:** Fundamental neural network implementation in 10-617/707.

---

#### Problem 5.2: Backpropagation Calculation ⭐⭐⭐
**Question:** For a simple 2-layer network, manually calculate the gradients for one training example using backpropagation.

**Given:**
- Input: x = [1, 2]
- True label: y = 1
- Weights: W1 = [[0.5, 0.3], [0.2, 0.4]], W2 = [0.1, 0.2]
- Use sigmoid activation and MSE loss

**Solution Steps:**
1. Forward pass: Calculate activations
2. Calculate loss: L = (y_pred - y)²
3. Backward pass: Calculate gradients using chain rule
4. Update weights: w = w - α × gradient

**CMU Connection:** Core algorithm in 10-617/707.

---

### Problem Set 6: Convolutional Neural Networks

#### Problem 6.1: Convolution Operation ⭐⭐
**Question:** Manually compute the convolution of a 3×3 image with a 2×2 filter (stride=1, padding=0).

**Given:**
```
Image:          Filter:
[1  2  3]       [1  0]
[4  5  6]       [0  1]
[7  8  9]
```

**Solution:**
```python
import numpy as np

def convolve_2d(image, filter_kernel, stride=1, padding=0):
    """
    Perform 2D convolution.
    
    Args:
        image: 2D array
        filter_kernel: 2D filter
        stride: Step size
        padding: Padding size
    
    Returns:
        Convolved output
    """
    # Add padding if needed
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    
    h, w = image.shape
    fh, fw = filter_kernel.shape
    
    # Calculate output dimensions
    out_h = (h - fh) // stride + 1
    out_w = (w - fw) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(0, out_h):
        for j in range(0, out_w):
            # Extract region
            region = image[i*stride:i*stride+fh, j*stride:j*stride+fw]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * filter_kernel)
    
    return output

# Manual calculation:
# Top-left: [1,2; 4,5] * [1,0; 0,1] = 1×1 + 2×0 + 4×0 + 5×1 = 6
# Top-right: [2,3; 5,6] * [1,0; 0,1] = 2×1 + 3×0 + 5×0 + 6×1 = 8
# Bottom-left: [4,5; 7,8] * [1,0; 0,1] = 4×1 + 5×0 + 7×0 + 8×1 = 12
# Bottom-right: [5,6; 8,9] * [1,0; 0,1] = 5×1 + 6×0 + 8×0 + 9×1 = 14
# Result: [[6, 8], [12, 14]]
```

**CMU Connection:** Core CNN operation in 10-617/707.

---

## ⚙️ 10-725: Optimization for Machine Learning

### Problem Set 7: Gradient Descent Variants

#### Problem 7.1: Stochastic vs Batch Gradient Descent ⭐⭐
**Question:** Compare batch, mini-batch, and stochastic gradient descent. Implement mini-batch gradient descent.

**Solution:**
```python
import numpy as np

def mini_batch_gradient_descent(X, y, alpha=0.01, batch_size=32, epochs=100):
    """
    Mini-batch gradient descent.
    
    Args:
        X: Features (m×n)
        y: Targets (m×1)
        alpha: Learning rate
        batch_size: Size of each mini-batch
        epochs: Number of passes through data
    
    Returns:
        theta: Learned parameters
        cost_history: Cost at each iteration
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(m, 1)
    cost_history = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_m = X_batch.shape[0]
            
            # Forward pass
            h = X_batch @ theta
            
            # Cost
            cost = (1/(2*batch_m)) * np.sum((h - y_batch)**2)
            
            # Gradient
            gradient = (1/batch_m) * X_batch.T @ (h - y_batch)
            
            # Update
            theta = theta - alpha * gradient
            
            cost_history.append(cost)
    
    return theta, cost_history

# Comparison:
# Batch GD: Uses all data, slow but stable
# Stochastic GD: Uses 1 example, fast but noisy
# Mini-batch GD: Uses small batches, balance of speed and stability
```

**CMU Connection:** Core optimization topic in 10-725.

---

#### Problem 7.2: Momentum Implementation ⭐⭐⭐
**Question:** Implement gradient descent with momentum. Explain how momentum helps convergence.

**Solution:**
```python
def gradient_descent_momentum(X, y, alpha=0.01, beta=0.9, iterations=1000):
    """
    Gradient descent with momentum.
    
    Args:
        beta: Momentum coefficient (typically 0.9)
    
    Returns:
        theta: Learned parameters
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(m, 1)
    v = np.zeros((n, 1))  # Velocity (momentum term)
    
    for i in range(iterations):
        # Gradient
        h = X @ theta
        gradient = (1/m) * X.T @ (h - y)
        
        # Update velocity (exponentially weighted average)
        v = beta * v + (1 - beta) * gradient
        
        # Update parameters
        theta = theta - alpha * v
    
    return theta

# How momentum helps:
# 1. Accelerates convergence in consistent directions
# 2. Reduces oscillations in narrow valleys
# 3. Helps escape local minima
# 4. Smoothes gradient updates
```

**CMU Connection:** Advanced optimization in 10-725.

---

#### Problem 7.3: Learning Rate Scheduling ⭐⭐
**Question:** Implement learning rate decay. Why is it important?

**Solution:**
```python
def learning_rate_schedule(initial_lr, epoch, decay_rate=0.1, decay_epochs=30):
    """
    Exponential learning rate decay.
    
    Args:
        initial_lr: Starting learning rate
        epoch: Current epoch
        decay_rate: Decay factor
        decay_epochs: Decay every N epochs
    
    Returns:
        Current learning rate
    """
    lr = initial_lr * (decay_rate ** (epoch // decay_epochs))
    return lr

# Why important:
# 1. Large LR initially: Fast learning
# 2. Small LR later: Fine-tuning, avoid overshooting
# 3. Better convergence to optimal solution
```

**CMU Connection:** Practical optimization technique in 10-725.

---

## 🛠️ 10-718: Machine Learning in Practice

### Problem Set 8: Data Preprocessing

#### Problem 8.1: Feature Scaling Implementation ⭐⭐
**Question:** Implement min-max scaling and standardization. When would you use each?

**Solution:**
```python
import numpy as np

def min_max_scale(X):
    """
    Scale features to [0, 1] range.
    Formula: (x - min) / (max - min)
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_scaled = (X - X_min) / (X_max - X_min + 1e-8)  # Add small epsilon
    return X_scaled, X_min, X_max

def standardize(X):
    """
    Standardize features (mean=0, std=1).
    Formula: (x - mean) / std
    """
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / (X_std + 1e-8)
    return X_scaled, X_mean, X_std

# When to use:
# Min-Max: When you know the data range, neural networks (0-1)
# Standardization: When data follows normal distribution, many ML algorithms
```

**CMU Connection:** Essential preprocessing in 10-718.

---

#### Problem 8.2: Handling Missing Data ⭐⭐
**Question:** Given a dataset with missing values, implement strategies to handle them.

**Solution:**
```python
import numpy as np
import pandas as pd

def handle_missing_data(X, strategy='mean'):
    """
    Handle missing values.
    
    Strategies:
    - 'mean': Replace with mean
    - 'median': Replace with median
    - 'mode': Replace with mode
    - 'drop': Drop rows with missing values
    - 'forward_fill': Forward fill
    """
    X_handled = X.copy()
    
    if strategy == 'mean':
        X_handled = X_handled.fillna(X_handled.mean())
    elif strategy == 'median':
        X_handled = X_handled.fillna(X_handled.median())
    elif strategy == 'mode':
        X_handled = X_handled.fillna(X_handled.mode().iloc[0])
    elif strategy == 'drop':
        X_handled = X_handled.dropna()
    elif strategy == 'forward_fill':
        X_handled = X_handled.fillna(method='ffill')
    
    return X_handled

# Considerations:
# - Amount of missing data
# - Type of feature (categorical vs numerical)
# - Relationship with target variable
```

**CMU Connection:** Real-world data handling in 10-718.

---

#### Problem 8.3: One-Hot Encoding vs Label Encoding ⭐⭐
**Question:** When should you use one-hot encoding vs label encoding? Implement both.

**Solution:**
```python
import numpy as np

def one_hot_encode(categories):
    """
    One-hot encode categorical variables.
    """
    unique_cats = np.unique(categories)
    n_cats = len(unique_cats)
    n_samples = len(categories)
    
    encoded = np.zeros((n_samples, n_cats))
    
    for i, cat in enumerate(categories):
        idx = np.where(unique_cats == cat)[0][0]
        encoded[i, idx] = 1
    
    return encoded

def label_encode(categories):
    """
    Label encode categorical variables.
    """
    unique_cats = np.unique(categories)
    mapping = {cat: idx for idx, cat in enumerate(unique_cats)}
    
    encoded = np.array([mapping[cat] for cat in categories])
    return encoded, mapping

# When to use:
# One-Hot: Nominal categories (no order), e.g., colors, countries
# Label: Ordinal categories (has order), e.g., ratings, sizes
```

**CMU Connection:** Feature engineering in 10-718.

---

## 📊 36-700/705: Probability & Statistics

### Problem Set 9: Probability Fundamentals

#### Problem 9.1: Bayes' Theorem Application ⭐⭐
**Question:** In a medical test, 1% of population has a disease. The test is 99% accurate (99% true positive, 99% true negative). If someone tests positive, what's the probability they actually have the disease?

**Solution:**
```python
# Given:
# P(Disease) = 0.01
# P(Test+|Disease) = 0.99
# P(Test-|No Disease) = 0.99

# Calculate P(Disease|Test+)
# Using Bayes' Theorem:
# P(Disease|Test+) = P(Test+|Disease) × P(Disease) / P(Test+)

p_disease = 0.01
p_test_pos_given_disease = 0.99
p_test_neg_given_no_disease = 0.99

# P(Test+) = P(Test+|Disease)×P(Disease) + P(Test+|No Disease)×P(No Disease)
p_test_pos = (p_test_pos_given_disease * p_disease + 
              (1 - p_test_neg_given_no_disease) * (1 - p_disease))

# P(Disease|Test+)
p_disease_given_test_pos = (p_test_pos_given_disease * p_disease) / p_test_pos

print(f"P(Disease|Test+): {p_disease_given_test_pos:.3f}")
# Result: ~0.50 (50%) - Counterintuitive but correct!
```

**CMU Connection:** Fundamental probability in 36-700/705.

---

#### Problem 9.2: Maximum Likelihood Estimation ⭐⭐⭐
**Question:** Derive MLE for the mean of a normal distribution given data points.

**Solution:**
```python
import numpy as np

def mle_normal_mean(data):
    """
    Maximum Likelihood Estimation for normal distribution mean.
    
    For normal distribution: μ_MLE = (1/n) Σ x_i
    """
    return np.mean(data)

# Derivation:
# Likelihood: L(μ, σ²) = Π (1/√(2πσ²)) exp(-(x_i - μ)²/(2σ²))
# Log-likelihood: l(μ, σ²) = -n/2 log(2πσ²) - (1/(2σ²)) Σ(x_i - μ)²
# Take derivative w.r.t. μ and set to 0:
# ∂l/∂μ = (1/σ²) Σ(x_i - μ) = 0
# Therefore: μ_MLE = (1/n) Σ x_i
```

**CMU Connection:** Statistical estimation in 36-700/705.

---

## 🎯 Exam-Style Questions

### Problem 10.1: Short Answer - Activation Functions ⭐⭐
**Question:** Explain why ReLU is preferred over sigmoid in hidden layers of deep neural networks. Give at least 3 reasons.

**Answer:**
1. **Vanishing Gradient Problem:** Sigmoid saturates (gradient → 0) for large inputs, making backpropagation difficult. ReLU has constant gradient (1) for positive inputs.
2. **Computational Efficiency:** ReLU is just max(0, x) - very fast. Sigmoid requires exponential computation.
3. **Sparsity:** ReLU outputs 0 for negative inputs, creating sparse representations which can be more efficient.
4. **Faster Convergence:** ReLU networks typically train faster than sigmoid networks.

---

### Problem 10.2: Code Debugging ⭐⭐
**Question:** The following gradient descent code has a bug. Find and fix it.

```python
def gradient_descent(X, y, alpha=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)  # Bug: Should be (n, 1) or reshape
    y = y.reshape(m, 1)
    
    for i in range(iterations):
        h = X @ theta  # Bug: Dimension mismatch
        error = h - y
        gradient = (X.T @ error) / m
        theta = theta - alpha * gradient  # Bug: Should be alpha * gradient
    
    return theta
```

**Fixed Code:**
```python
def gradient_descent(X, y, alpha=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))  # Fixed: Column vector
    y = y.reshape(m, 1)
    
    for i in range(iterations):
        h = X @ theta  # Now dimensions match
        error = h - y
        gradient = (X.T @ error) / m
        theta = theta - alpha * gradient  # Fixed: Correct update
    
    return theta
```

---

### Problem 10.3: Theoretical - Overfitting ⭐⭐⭐
**Question:** Explain the bias-variance decomposition of generalization error. How does regularization address this?

**Answer:**
**Bias-Variance Decomposition:**
```
E[(y - f̂(x))²] = Bias² + Variance + Irreducible Error
```

- **Bias:** Error from overly simplistic assumptions (underfitting)
- **Variance:** Error from sensitivity to small fluctuations (overfitting)
- **Irreducible Error:** Noise in data

**Regularization:**
- **L2 Regularization (Ridge):** Adds λ||θ||² to cost function
  - Reduces variance by constraining weights
  - Prevents overfitting
  - Trades some bias for lower variance

- **L1 Regularization (Lasso):** Adds λ||θ||₁
  - Also reduces variance
  - Can set some weights to exactly 0 (feature selection)

---

## 📝 Study Tips for CMU Exams

1. **Practice Implementations:** Be able to code algorithms from scratch
2. **Understand Theory:** Know why algorithms work, not just how
3. **Know Tradeoffs:** Understand when to use which method
4. **Time Management:** Practice problems under time constraints
5. **Review Fundamentals:** Linear algebra, calculus, probability basics

---

## 🔗 Additional Resources

- **CMU Course Websites:** Check for past homework/exam problems
- **Textbooks:** Bishop (PRML), Goodfellow (Deep Learning), Boyd (Optimization)
- **Practice Platforms:** Kaggle, LeetCode (ML problems), Papers With Code

---

**Good luck with your CMU MSML preparation! 🎓**

