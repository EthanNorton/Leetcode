# CMU MSML Preparation - Practice Problems & Overviews

**Purpose:** Organized collection of ML fundamentals with CMU MSML course alignment and research connections.

**Structure:**
- `00_Skill_Exercises/` - **⭐ START HERE!** Completed solutions for skill-building exercises
- `Templates/` - **⭐ PRACTICE HERE!** Blank templates to practice before looking at solutions
- `01_Foundations/` - Core concepts (ReLU, Gradient Descent, Matrix Ops)
- `02_Course_Specific/` - Problems organized by CMU courses
- `03_Practice_Problems/` - Additional practice with solutions
- `04_Research_Alignment/` - Connections to CMU research (Dr. Shah, etc.)

---

## 🚀 Quick Start Guide

### For Complete Beginners:
1. **Start with `Templates/`** - Practice with blank templates first
   - Begin with `Templates/01_Core_Python/exercise_1_normalize_SIMPLE.py` (easiest!)
   - Read `Templates/01_Core_Python/SIMPLE_GUIDE.md` for step-by-step help
   - Use `Templates/HOW_TO_SOLVE.md` for problem-solving strategies
2. **Compare with solutions** in `00_Skill_Exercises/` after attempting
3. **Use side-by-side guides** like `Templates/02_NumPy/SIDE_BY_SIDE_COMPARISON.md`

### For CMU Prep:
1. **Master fundamentals** in `00_Skill_Exercises/` and `Templates/`
2. Review `01_Foundations/` - Core concepts aligned with CMU courses
3. Review `02_Course_Specific/` by course number
4. Focus on courses you'll take first semester
5. Connect problems to research interests in `04_Research_Alignment/`

### For Practice:
1. **Use `Templates/`** for blank practice exercises
2. Try problems without looking at solutions first
3. Compare your approach with solutions in `00_Skill_Exercises/`
4. Use `03_Practice_Problems/` for additional exercises

---

## CMU Course Mapping

### 10-701/715: Introduction to Machine Learning
- Evaluation Metrics (Accuracy, Precision, Recall)
- Linear Regression (Gradient Descent, Normal Equation)
- Feature Scaling
- One-Hot Encoding

### 10-617/707: Deep Learning
- ReLU Activation Function
- Softmax Activation Function
- Matrix Operations (foundation)
- Neural Network Fundamentals

### 10-718: Machine Learning in Practice
- Feature Scaling
- Evaluation Metrics
- One-Hot Encoding
- Preprocessing Pipeline

### 10-725: Optimization for Machine Learning
- Gradient Descent
- Matrix Operations
- Linear Regression
- Optimization Fundamentals

### 36-700/705: Probability & Statistics
- Matrix Operations (foundation)
- Evaluation Metrics (statistical perspective)

---

## Research Alignment

### Dr. Shah's Research Areas:
- Evaluation Science
- Annotation Bias
- Reviewer Assignment
- "The More You Automate, The Less You See"

**Relevant Problems:**
- Evaluation Metrics (Calculate Accuracy Score)
- Feature Engineering (One-Hot Encoding)
- Preprocessing (Feature Scaling)

---

## Problem Difficulty Levels

- ⭐ **Beginner**: Start here! Builds fundamentals
- ⭐⭐ **Intermediate**: Requires understanding of basics
- ⭐⭐⭐ **Advanced**: For deeper understanding

---

## 📁 Folder Structure Details

### `00_Skill_Exercises/` - Completed Solutions
**Purpose:** Reference solutions for skill-building exercises

**Available Exercises:**
- `01_Core_Python/exercise_1_normalize.py` - Z-score normalization
- `02_NumPy/exercise_2_matrix_ops.py` - Matrix operations & broadcasting
- `03_Loss_Gradients/exercise_3_mse_loss.py` - Loss functions & gradients

**How to Use:**
- ⚠️ **Don't peek until you've tried!** Use `Templates/` first
- Compare your implementation after attempting
- Learn from differences in approach

---

### `Templates/` - Practice Templates
**Purpose:** Blank templates for practicing before looking at solutions

**Available Templates:**
- `01_Core_Python/`
  - `exercise_1_normalize_SIMPLE.py` - ⭐ **EASIEST VERSION!** Simplified with step-by-step hints
  - `exercise_1_normalize_template.py` - Standard template
  - `SIMPLE_GUIDE.md` - ⭐ **START HERE!** Step-by-step problem-solving guide
- `02_NumPy/`
  - `exercise_2_matrix_ops_template.py` - Matrix operations practice
  - `SIDE_BY_SIDE_COMPARISON.md` - ⭐ **NEW!** Template vs solution comparison
  - `NEXT_STEPS.md` - Guidance for next exercises
- `03_Loss_Gradients/`
  - `exercise_3_mse_loss_template.py` - Loss function implementation

**Helpful Guides:**
- `HOW_TO_SOLVE.md` - General problem-solving strategies
- `WALKTHROUGH_EXAMPLE.md` - Complete detailed example walkthrough
- `README.md` - Template usage instructions

**How to Use:**
1. Open a template file
2. Fill in `TODO` sections
3. Test your implementation
4. Compare with `00_Skill_Exercises/` solutions

---

### `01_Foundations/` - Core ML Concepts
**Purpose:** Fundamental ML concepts aligned with CMU courses

**Topics:**
- **Activation_Functions/** - ReLU, Softmax, Sigmoid
- **Gradient_Descent/** - Linear regression optimization
- **Matrix_Operations/** - Matrix-vector multiplication, reshaping
- **Preprocessing/** - Feature scaling, one-hot encoding
- **Evaluation/** - Accuracy, precision, recall metrics

---

### `02_Course_Specific/` - CMU Course Alignment
**Purpose:** Problems organized by specific CMU courses

**Courses Covered:**
- 10-701/715: Introduction to Machine Learning
- 10-617/707: Deep Learning
- 10-718: Machine Learning in Practice
- 10-725: Optimization for Machine Learning
- 36-700/705: Probability & Statistics

---

### `03_Practice_Problems/` - Additional Practice
**Purpose:** Extra exercises for reinforcement

---

### `04_Research_Alignment/` - Research Connections
**Purpose:** Connect problems to CMU research areas (Dr. Shah, etc.)

---

## 🎯 Recommended Learning Path

### Week 1-2: Core Python Fundamentals
1. Start with `Templates/01_Core_Python/exercise_1_normalize_SIMPLE.py`
2. Read `Templates/01_Core_Python/SIMPLE_GUIDE.md`
3. Complete the normalization exercise
4. Compare with `00_Skill_Exercises/01_Core_Python/exercise_1_normalize.py`

### Week 3-4: NumPy & Matrix Operations
1. Work through `Templates/02_NumPy/exercise_2_matrix_ops_template.py`
2. Use `Templates/02_NumPy/SIDE_BY_SIDE_COMPARISON.md` for reference
3. Master matrix multiplication, broadcasting, vectorized operations
4. Compare with `00_Skill_Exercises/02_NumPy/exercise_2_matrix_ops.py`

### Week 5-6: Loss Functions & Gradients
1. Practice with `Templates/03_Loss_Gradients/exercise_3_mse_loss_template.py`
2. Understand MSE, MAE, and gradient computation
3. Compare with `00_Skill_Exercises/03_Loss_Gradients/exercise_3_mse_loss.py`

### Week 7+: Foundations & Course-Specific
1. Review `01_Foundations/` concepts
2. Work through `02_Course_Specific/` problems
3. Explore `04_Research_Alignment/` connections

---

## How to Use This Folder

1. **Practice Mode**: Use `Templates/` - Implement solutions yourself
2. **Study Mode**: Read `01_Foundations/` - Understand concepts
3. **Review Mode**: Compare with `00_Skill_Exercises/` - Learn from solutions
4. **Research Mode**: Explore `04_Research_Alignment/` - Connect to CMU research

---

## 📚 Additional Resources

- **`QUICK_START.md`** - Quick reference guide
- **`ENHANCED_WORKFLOW.md`** - Complete 12-16 week learning path
- **`SKILLS_CHECKLIST.md`** - Track your progress
- **`STUDY_PLAN.md`** - Structured study schedule
- **`MASTER_INDEX.md`** - Complete index of all problems

---

## Next Steps

1. ✅ **Start with Templates** - Begin with `Templates/01_Core_Python/exercise_1_normalize_SIMPLE.py`
2. ✅ **Master fundamentals** - Complete all exercises in `00_Skill_Exercises/`
3. ✅ **Review foundations** - Work through `01_Foundations/`
4. ✅ **Course prep** - Review `02_Course_Specific/` problems
5. ✅ **Practice** - Use `03_Practice_Problems/` for reinforcement
6. ✅ **Research** - Explore `04_Research_Alignment/` connections

**Good luck with your CMU MSML preparation! 🎓**

