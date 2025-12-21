# Evaluation Metrics - CMU MSML Prep Overview

## Why This Matters for CMU MSML

**Relevant Courses:**
- **10-701/715 (Introduction to ML)**: Evaluation metrics, model assessment
- **10-718 (ML in Practice)**: Evaluation science, reproducibility
- **Research Alignment**: Dr. Shah's work on evaluation pitfalls and annotation bias

**Key Insight:** Understanding evaluation is crucial for research and practice!

---

## The Big Picture: Why Evaluation Matters

### The Fundamental Question
**"How do we know if our model is actually good?"**

This is THE central question in machine learning, and it's harder than it seems!

### Why It's Hard
1. **Training accuracy ≠ Real-world performance**
2. **Different metrics tell different stories**
3. **Evaluation can be biased** (Dr. Shah's research area!)
4. **Metrics can be gamed/manipulated**

---

## Accuracy Score: The Simplest Metric

### What It Is
**Accuracy = (Number of correct predictions) / (Total predictions)**

**Formula:** `accuracy = sum(y_true == y_pred) / len(y_true)`

### Example
```
True labels:    [cat, dog, cat, bird, cat]
Predictions:    [cat, dog, bird, bird, cat]
Correct:        [✓,   ✓,   ✗,   ✓,   ✓]  (4 out of 5)
Accuracy: 4/5 = 0.80 = 80%
```

### When Accuracy Works Well
- **Balanced classes**: Equal number of each class
- **All errors equally important**
- **Simple, interpretable metric**

### When Accuracy Fails
- **Imbalanced classes**: 99% class A, 1% class B
  - Model predicts A always → 99% accuracy!
  - But completely useless!
- **Different error costs**: 
  - Medical diagnosis: False negative (miss cancer) worse than false positive
  - Accuracy treats them equally!

---

## Beyond Accuracy: The Full Picture

### For CMU Courses, You'll Learn:

#### 1. **Confusion Matrix** (10-701/715)
```
                Predicted
              Cat  Dog  Bird
Actual Cat    [50   5    2]
       Dog    [3   45    7]
       Bird   [1    4   48]
```

**What it shows:**
- True Positives (diagonal)
- False Positives (off-diagonal)
- False Negatives (off-diagonal)

**Why it matters:** More informative than accuracy alone!

#### 2. **Precision & Recall** (10-701/715, 10-718)
- **Precision**: Of what we predicted, how many were correct?
- **Recall**: Of what's actually there, how many did we find?

**Example (Medical):**
- High precision: When we say "cancer", we're usually right
- High recall: We catch most cancers (don't miss many)

**Trade-off:** Often can't have both high!

#### 3. **F1-Score** (10-701/715)
- Harmonic mean of precision and recall
- Balances both concerns
- Useful when you need both precision and recall

#### 4. **ROC-AUC** (10-701/715)
- Receiver Operating Characteristic curve
- Area Under Curve
- Good for binary classification
- Shows performance across all thresholds

---

## CMU Course Connections

### 10-701/715: Introduction to Machine Learning
**Topics you'll cover:**
- Evaluation metrics (accuracy, precision, recall, F1)
- Cross-validation
- Train/validation/test splits
- Overfitting detection
- Bias-variance trade-off

**Why it matters:** Foundation for all ML work!

### 10-718: Machine Learning in Practice
**Topics you'll cover:**
- Evaluation pitfalls (Dr. Shah's research!)
- Reproducibility in evaluation
- Annotation bias
- Benchmark evaluation
- Real-world deployment evaluation

**Why it matters:** Research alignment! This connects to your interests.

### Research Connection: Dr. Shah's Work
**Key papers/themes:**
- "The More You Automate, The Less You See"
- Evaluation science
- Annotation bias
- Reviewer assignment

**Your understanding of evaluation metrics directly connects to this research!**

---

## Real-World Evaluation Challenges

### Challenge 1: Data Leakage
**Problem:** Test data information leaks into training
**Result:** Artificially high accuracy
**Solution:** Proper train/test splits, cross-validation

### Challenge 2: Distribution Shift
**Problem:** Training data ≠ Real-world data
**Example:** Train on ImageNet, deploy on phone photos
**Result:** High training accuracy, poor real-world performance

### Challenge 3: Evaluation Bias
**Problem:** Evaluation set doesn't represent real distribution
**Example:** Test set has 90% class A, real world has 50/50
**Result:** Misleading metrics

### Challenge 4: Metric Gaming
**Problem:** Optimizing for metric that doesn't matter
**Example:** High accuracy on easy examples, fails on hard ones
**Solution:** Use multiple metrics, understand what matters

---

## Evaluation Best Practices (CMU-Level)

### 1. **Use Multiple Metrics**
- Don't rely on accuracy alone
- Use precision, recall, F1, ROC-AUC
- Understand what each tells you

### 2. **Proper Data Splits**
- Train: Learn from this
- Validation: Tune hyperparameters
- Test: Final evaluation (only once!)

### 3. **Cross-Validation**
- K-fold cross-validation
- More robust evaluation
- Reduces variance in estimates

### 4. **Understand Your Domain**
- Medical: Recall matters (don't miss cancers)
- Spam detection: Precision matters (don't block real emails)
- Research: Understand what actually matters!

---

## CMU MSML Preparation Focus

### What to Master Now:
1. **Basic metrics**: Accuracy, precision, recall, F1
2. **Confusion matrix**: How to read and interpret
3. **Train/test splits**: Why and how
4. **Evaluation pitfalls**: Common mistakes

### What You'll Learn at CMU:
1. **Advanced metrics**: ROC-AUC, PR curves, calibration
2. **Evaluation theory**: Statistical significance, confidence intervals
3. **Research evaluation**: Benchmark design, reproducibility
4. **Domain-specific evaluation**: What matters in different fields

### Research Alignment:
- Understanding evaluation connects to Dr. Shah's work
- Evaluation science is a research area
- Can contribute to evaluation methodology research

---

## The Evaluation Pipeline

### Standard Workflow:
```
1. Train model
2. Make predictions
3. Compare predictions to true labels
4. Calculate metrics
5. Interpret results
6. Iterate and improve
```

### CMU Research Level:
```
1. Design evaluation framework
2. Consider evaluation bias
3. Multiple metrics and perspectives
4. Statistical significance testing
5. Reproducibility checks
6. Publication and sharing
```

---

## Key Takeaways for CMU Prep

### 1. **Evaluation is Fundamental**
- Every ML project needs evaluation
- Understanding metrics is essential
- Connects to research (evaluation science)

### 2. **Accuracy is Just the Start**
- Many metrics exist for good reasons
- Different metrics for different problems
- Need to understand trade-offs

### 3. **Evaluation is Hard**
- Many pitfalls and biases
- Requires careful design
- Active research area (Dr. Shah!)

### 4. **Connects to Your Research Interests**
- Evaluation science
- Annotation bias
- Reproducibility
- All relate to evaluation metrics!

---

## Next Steps for CMU Prep

### Immediate:
- Master basic metrics (accuracy, precision, recall)
- Understand confusion matrix
- Practice on real datasets

### Before CMU:
- Read evaluation papers
- Understand evaluation pitfalls
- Learn about Dr. Shah's work on evaluation

### At CMU:
- Deep dive into evaluation theory
- Research evaluation methodology
- Contribute to evaluation science!

---

## Bottom Line

**Evaluation metrics are how we measure ML success.**

For CMU MSML:
- **Foundation**: 10-701/715 covers metrics
- **Practice**: 10-718 covers evaluation science
- **Research**: Connects to Dr. Shah's work
- **Career**: Essential for any ML role

**Mastering evaluation = Essential ML skill!**

