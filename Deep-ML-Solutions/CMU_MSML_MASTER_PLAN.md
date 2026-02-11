# CMU MSML-AS Master Plan: Deep ML Problems → Core Courses Mapping

**Objective:** Map all Deep ML problems to CMU MSML core courses, create a 6-month preparation plan with weekly breakdown, YouTube resources, and project alignment.

---

## 📚 CMU MSML Core Courses Overview

| Course Code | Course Name | Key Topics |
|------------|-------------|------------|
| **10-701/715** | Introduction to Machine Learning | Supervised/unsupervised learning, regression, classification, evaluation metrics, bias-variance |
| **10-617/707** | Intermediate/Advanced Deep Learning | Neural networks, backpropagation, CNNs, RNNs, activation functions, optimization |
| **10-708** | Probabilistic Graphical Models | Bayesian networks, Markov models, inference algorithms, learning from data |
| **10-718** | Machine Learning in Practice | End-to-end ML lifecycle, preprocessing, feature engineering, deployment |
| **10-725** | Optimization for Machine Learning | Convex optimization, gradient methods, stochastic optimization, duality |
| **36-700/705** | Probability & Mathematical Statistics | Probability theory, statistical inference, hypothesis testing, estimation |

---

## 🗺️ Deep ML Problems → CMU Course Mapping

### **1. Calculate Accuracy Score**
- **Maps to:** 10-701/715 (Introduction to ML), 10-718 (ML in Practice)
- **Prepares for:** Evaluation metrics, model assessment, classification performance
- **Key Concepts:** Precision/recall, confusion matrices, evaluation methodology
- **Difficulty:** ⭐ (Beginner)

### **2. Feature Scaling Implementation**
- **Maps to:** 10-718 (ML in Practice), 10-701/715 (Introduction to ML)
- **Prepares for:** Data preprocessing, normalization, standardization, feature engineering
- **Key Concepts:** Min-max scaling, z-score normalization, why scaling matters
- **Difficulty:** ⭐ (Beginner)

### **3. Matrix times Vector**
- **Maps to:** 10-725 (Optimization), 36-700/705 (Probability & Stats), All courses (foundation)
- **Prepares for:** Linear algebra fundamentals, matrix operations, vector spaces
- **Key Concepts:** Matrix multiplication, linear transformations, computational efficiency
- **Difficulty:** ⭐ (Beginner)

### **4. Reshape Matrix**
- **Maps to:** 10-617/707 (Deep Learning), 10-725 (Optimization)
- **Prepares for:** Tensor operations, CNN architectures, data reshaping for neural networks
- **Key Concepts:** Array manipulation, tensor dimensions, batch processing
- **Difficulty:** ⭐ (Beginner)

### **5. Linear Regression Using Normal Equation**
- **Maps to:** 10-701/715 (Introduction to ML), 10-725 (Optimization)
- **Prepares for:** Closed-form solutions, linear algebra in ML, analytical optimization
- **Key Concepts:** Pseudoinverse, least squares, matrix calculus
- **Difficulty:** ⭐⭐ (Intermediate)

### **6. Linear Regression Using Gradient Descent**
- **Maps to:** 10-725 (Optimization), 10-701/715 (Introduction to ML), 10-617/707 (Deep Learning)
- **Prepares for:** Optimization algorithms, iterative methods, learning rates, convergence
- **Key Concepts:** Gradient computation, update rules, batch vs stochastic
- **Difficulty:** ⭐⭐ (Intermediate)

### **7. Implement ReLU Activation Function**
- **Maps to:** 10-617/707 (Deep Learning)
- **Prepares for:** Neural network architectures, activation functions, non-linearity
- **Key Concepts:** Non-linear transformations, sparsity, gradient flow
- **Difficulty:** ⭐ (Beginner)

### **8. Leaky ReLU Activation Function**
- **Maps to:** 10-617/707 (Deep Learning)
- **Prepares for:** Advanced activation functions, addressing dying ReLU problem
- **Key Concepts:** Activation function variants, hyperparameter tuning
- **Difficulty:** ⭐ (Beginner)

### **9. Sigmoid Activation Function Understanding**
- **Maps to:** 10-617/707 (Deep Learning), 10-701/715 (Introduction to ML)
- **Prepares for:** Logistic regression, binary classification, probability outputs
- **Key Concepts:** S-shaped curves, probability interpretation, vanishing gradients
- **Difficulty:** ⭐ (Beginner)

### **10. Softmax Activation Function Implementation**
- **Maps to:** 10-617/707 (Deep Learning), 10-701/715 (Introduction to ML)
- **Prepares for:** Multi-class classification, probability distributions, cross-entropy loss
- **Key Concepts:** Normalization, exponential functions, output layers
- **Difficulty:** ⭐⭐ (Intermediate)

### **11. One-Hot Encoding of Nominal Values**
- **Maps to:** 10-718 (ML in Practice), 10-701/715 (Introduction to ML)
- **Prepares for:** Feature engineering, categorical data handling, preprocessing pipelines
- **Key Concepts:** Categorical encoding, sparse representations, dummy variables
- **Difficulty:** ⭐ (Beginner)

---

## 📅 6-Month Learning Plan: Deep ML Problems → CMU MSML Prep

**Timeline:** 26 weeks (6 months)  
**Weekly Commitment:** ~10 hours/week  
**Breakdown:** 3 hrs math/theory, 2 hrs systems/algorithms, 2 hrs research reading, 3 hrs project/dev work

---

### **Month 1 (Weeks 1-4): Foundations & Linear Algebra**

#### **Week 1: Mathematical Foundations**
- **Deep ML Problems:**
  - ✅ Matrix times Vector
  - ✅ Reshape Matrix
- **CMU Course Prep:** 10-725 (Optimization), Foundation for all courses
- **YouTube Resources:**
  - 3Blue1Brown: "Essence of Linear Algebra" (Playlist, ~2 hours)
    - Vectors, Linear combinations, Matrix multiplication
  - StatQuest: "Linear Algebra" (30 min)
- **Learning Goals:**
  - Understand matrix-vector multiplication
  - Master array reshaping operations
  - Build intuition for linear transformations
- **Practice:** Implement matrix operations from scratch (no NumPy)

#### **Week 2: Probability & Statistics Basics**
- **Deep ML Problems:**
  - ✅ Calculate Accuracy Score
- **CMU Course Prep:** 36-700/705 (Probability & Stats), 10-701/715 (Introduction to ML)
- **YouTube Resources:**
  - StatQuest: "Statistics Fundamentals" (1 hour)
  - 3Blue1Brown: "Probability" series (2 hours)
- **Learning Goals:**
  - Understand evaluation metrics
  - Learn precision, recall, F1-score
  - Practice with confusion matrices
- **Practice:** Build evaluation metrics calculator for different classification tasks

#### **Week 3: Data Preprocessing Fundamentals**
- **Deep ML Problems:**
  - ✅ Feature Scaling Implementation
  - ✅ One-Hot Encoding of Nominal Values
- **CMU Course Prep:** 10-718 (ML in Practice)
- **YouTube Resources:**
  - StatQuest: "Feature Scaling" (15 min)
  - "Data Preprocessing for Machine Learning" - Krish Naik (1 hour)
- **Learning Goals:**
  - Implement min-max scaling, standardization
  - Understand when to use each scaling method
  - Master categorical encoding techniques
- **Practice:** Preprocess a real dataset (e.g., UCI ML Repository)

#### **Week 4: Project Skeleton & Review**
- **Review:** All Week 1-3 problems
- **Project:** Set up GitHub repo for "Seed-Stability Module"
- **Deliverable:** Data preprocessing pipeline with scaling + encoding
- **CMU Alignment:** Document how preprocessing prepares for 10-718

---

### **Month 2 (Weeks 5-8): Linear Models & Optimization**

#### **Week 5: Linear Regression - Analytical Approach**
- **Deep ML Problems:**
  - ✅ Linear Regression Using Normal Equation
- **CMU Course Prep:** 10-701/715 (Introduction to ML), 10-725 (Optimization)
- **YouTube Resources:**
  - StatQuest: "Linear Regression" (20 min)
  - 3Blue1Brown: "Linear Transformations" (30 min)
  - "Normal Equation Derivation" - Andrew Ng (Coursera clip, 15 min)
- **Learning Goals:**
  - Derive normal equation mathematically
  - Understand pseudoinverse
  - Compare analytical vs iterative solutions
- **Practice:** Implement normal equation, compare with sklearn

#### **Week 6: Linear Regression - Iterative Optimization**
- **Deep ML Problems:**
  - ✅ Linear Regression Using Gradient Descent
- **CMU Course Prep:** 10-725 (Optimization), 10-617/707 (Deep Learning)
- **YouTube Resources:**
  - StatQuest: "Gradient Descent" (20 min)
  - 3Blue1Brown: "Gradient Descent" (20 min)
  - "Understanding Gradient Descent" - Andrew Ng (Coursera, 30 min)
- **Learning Goals:**
  - Implement gradient descent from scratch
  - Understand learning rate impact
  - Visualize convergence
- **Practice:** Animate gradient descent on 2D loss surface

#### **Week 7: Optimization Deep Dive**
- **Review:** Gradient descent variants
- **CMU Course Prep:** 10-725 (Optimization)
- **YouTube Resources:**
  - "Stochastic Gradient Descent" - Andrew Ng (30 min)
  - "Momentum, RMSprop, Adam" - deeplearning.ai (45 min)
- **Learning Goals:**
  - Compare batch, mini-batch, stochastic GD
  - Implement momentum
  - Understand adaptive learning rates
- **Practice:** Implement SGD, mini-batch GD, compare convergence

#### **Week 8: Project 1 Development**
- **Project 1:** Seed-Stability Module
  - Implement forecasting baseline with gradient descent
  - Run 10 seed experiments
  - Document reproducibility
- **CMU Alignment:** Connect to 10-718 (reproducibility) and 10-725 (optimization)

---

### **Month 3 (Weeks 9-12): Neural Networks & Activation Functions**

#### **Week 9: Activation Functions - ReLU Family**
- **Deep ML Problems:**
  - ✅ Implement ReLU Activation Function
  - ✅ Leaky ReLU Activation Function
- **CMU Course Prep:** 10-617/707 (Deep Learning)
- **YouTube Resources:**
  - StatQuest: "ReLU In Action" (15 min)
  - "Activation Functions Explained" - deeplizard (30 min)
  - 3Blue1Brown: "Neural Networks" series, Part 1-2 (1 hour)
- **Learning Goals:**
  - Understand why ReLU is popular
  - Compare ReLU vs Leaky ReLU
  - Visualize activation function outputs
- **Practice:** Build simple neural network with ReLU, compare with Leaky ReLU

#### **Week 10: Activation Functions - Sigmoid & Softmax**
- **Deep ML Problems:**
  - ✅ Sigmoid Activation Function Understanding
  - ✅ Softmax Activation Function Implementation
- **CMU Course Prep:** 10-617/707 (Deep Learning), 10-701/715 (Introduction to ML)
- **YouTube Resources:**
  - "Sigmoid vs ReLU" - deeplizard (20 min)
  - "Softmax Explained" - StatQuest (15 min)
  - "Activation Functions Comparison" - Sentdex (30 min)
- **Learning Goals:**
  - Understand sigmoid for binary classification
  - Master softmax for multi-class
  - Compare activation functions
- **Practice:** Implement logistic regression with sigmoid, multi-class with softmax

#### **Week 11: Neural Network Fundamentals**
- **Review:** All activation functions
- **CMU Course Prep:** 10-617/707 (Deep Learning)
- **YouTube Resources:**
  - 3Blue1Brown: "Neural Networks" series, Part 3-4 (1.5 hours)
  - "Backpropagation Explained" - StatQuest (20 min)
  - "Building Neural Networks from Scratch" - sentdex (2 hours)
- **Learning Goals:**
  - Understand forward propagation
  - Learn backpropagation intuition
  - Build 2-layer neural network from scratch
- **Practice:** Implement forward/backward pass manually

#### **Week 12: Project 1 Expansion**
- **Project 1:** Add neural network forecasting model
  - Compare linear regression vs neural network
  - Analyze seed stability across both models
  - Generate reproducibility report
- **CMU Alignment:** Document connection to 10-617/707 and 10-718

---

### **Month 4 (Weeks 13-16): Advanced Topics & Integration**

#### **Week 13: Review & Integration Week**
- **Review:** All Deep ML problems completed so far
- **CMU Course Prep:** All core courses
- **YouTube Resources:**
  - "ML Fundamentals Review" - Andrew Ng (1 hour)
  - "Deep Learning Review" - fast.ai (1 hour)
- **Learning Goals:**
  - Create summary document of all problems
  - Map each problem to CMU courses
  - Identify knowledge gaps
- **Practice:** Build comprehensive cheat sheet

#### **Week 14: Advanced Optimization**
- **CMU Course Prep:** 10-725 (Optimization)
- **YouTube Resources:**
  - "Convex Optimization" - Boyd lectures (2 hours)
  - "KKT Conditions" - Khan Academy (30 min)
- **Learning Goals:**
  - Understand convexity
  - Learn constrained optimization
  - Study duality theory
- **Practice:** Solve optimization problems from Boyd exercises

#### **Week 15: Probabilistic Models Introduction**
- **CMU Course Prep:** 10-708 (Probabilistic Graphical Models)
- **YouTube Resources:**
  - "Bayesian Networks" - StatQuest (20 min)
  - "PGMs Introduction" - Daphne Koller (Coursera, 1 hour)
- **Learning Goals:**
  - Understand Bayesian inference
  - Learn Markov models basics
  - Study probabilistic reasoning
- **Practice:** Implement simple Bayesian network

#### **Week 16: Project 2 Skeleton**
- **Project 2:** Annotation Bias Audit
  - Design simulation framework
  - Plan reviewer assignment model
  - Connect to Dr. Shah's work
- **CMU Alignment:** Link to 10-708 (PGMs for modeling bias) and research interests

---

### **Month 5 (Weeks 17-20): Research Alignment & Advanced Practice**

#### **Week 17: Research Literacy - Evaluation Science**
- **CMU Course Prep:** 10-718 (ML in Practice), Research alignment
- **Reading:**
  - "The More You Automate, The Less You See" (Dr. Shah)
  - "Benchmarking in NLP" papers
- **YouTube Resources:**
  - "ML Evaluation Best Practices" - various (1 hour)
  - "Reproducibility in ML" - talks (1 hour)
- **Learning Goals:**
  - Understand evaluation pitfalls
  - Learn about annotation bias
  - Study reproducibility frameworks
- **Practice:** Audit evaluation metrics in Project 1

#### **Week 18: Project 2 Development**
- **Project 2:** Annotation Bias Audit
  - Implement reviewer assignment simulation
  - Model annotation biases
  - Generate bias analysis report
- **CMU Alignment:** Connect to Dr. Shah's reviewer-assignment research
- **Deliverable:** Working simulation with documentation

#### **Week 19: Systems & Deployment**
- **CMU Course Prep:** 10-718 (ML in Practice)
- **YouTube Resources:**
  - "MLOps Fundamentals" - various (1.5 hours)
  - "Docker for ML" - tutorials (1 hour)
- **Learning Goals:**
  - Learn containerization
  - Understand ML pipelines
  - Study model deployment
- **Practice:** Containerize Project 1, set up MLflow tracking

#### **Week 20: Advanced Deep Learning Topics**
- **CMU Course Prep:** 10-617/707 (Deep Learning)
- **YouTube Resources:**
  - "CNNs Explained" - 3Blue1Brown (30 min)
  - "RNNs and LSTMs" - StatQuest (20 min)
  - "Transformers" - various (1 hour)
- **Learning Goals:**
  - Understand CNN architectures
  - Learn RNN/LSTM basics
  - Study attention mechanisms
- **Practice:** Implement simple CNN for image classification

---

### **Month 6 (Weeks 21-26): Integration, Portfolio & Presentation**

#### **Week 21: Project 1 Finalization**
- **Project 1:** Seed-Stability Module
  - Clean code, add documentation
  - Create comprehensive README
  - Generate final reproducibility report
  - Visualize seed stability results
- **Deliverable:** Production-ready Project 1

#### **Week 22: Project 2 Finalization**
- **Project 2:** Annotation Bias Audit
  - Polish simulation code
  - Write analysis report
  - Connect findings to research literature
  - Document CMU alignment
- **Deliverable:** Production-ready Project 2

#### **Week 23: Portfolio Preparation**
- **Tasks:**
  - Update GitHub with all projects
  - Create master README linking to CMU prep
  - Write project summaries
  - Update CV/Resume
- **Deliverable:** Professional GitHub portfolio

#### **Week 24: Research Agenda Draft**
- **Tasks:**
  - Draft "Next Steps for CMU MSML" document
  - Map courses to research interests
  - Identify lab alignment (Shah, Cohen, others)
  - Create 1-year plan at CMU
- **Deliverable:** Research agenda document

#### **Week 25: Presentation Preparation**
- **Tasks:**
  - Create 5-slide summary for each project
  - Practice presentation narrative
  - Prepare for potential interviews
  - Mock presentations with peers
- **Deliverable:** Presentation materials

#### **Week 26: Final Integration & Networking**
- **Tasks:**
  - Publish blog/LinkedIn update: "6-Month Prep for CMU MSML"
  - Network with CMU faculty (Dr. Shah, etc.)
  - Share progress updates
  - Celebrate milestone!
- **Deliverable:** Published reflection, network connections

---

## 📊 Problem Difficulty & Time Estimates

| Problem | Difficulty | Time Estimate | CMU Courses |
|---------|-----------|---------------|-------------|
| Matrix times Vector | ⭐ | 2 hours | 10-725, Foundation |
| Reshape Matrix | ⭐ | 1 hour | 10-617/707 |
| Calculate Accuracy Score | ⭐ | 2 hours | 10-701/715, 10-718 |
| Feature Scaling | ⭐ | 3 hours | 10-718 |
| One-Hot Encoding | ⭐ | 2 hours | 10-718 |
| ReLU | ⭐ | 2 hours | 10-617/707 |
| Leaky ReLU | ⭐ | 1 hour | 10-617/707 |
| Sigmoid | ⭐ | 2 hours | 10-617/707, 10-701/715 |
| Linear Regression (Normal) | ⭐⭐ | 4 hours | 10-701/715, 10-725 |
| Linear Regression (GD) | ⭐⭐ | 5 hours | 10-725, 10-617/707 |
| Softmax | ⭐⭐ | 3 hours | 10-617/707, 10-701/715 |

**Total Estimated Time:** ~27 hours of focused problem-solving

---

## 🎯 Key YouTube Playlists by Topic

### **Linear Algebra & Math Foundations**
1. **3Blue1Brown - Essence of Linear Algebra** (Playlist, ~3 hours)
2. **Khan Academy - Linear Algebra** (Playlist, ~10 hours)
3. **StatQuest - Statistics Fundamentals** (Playlist, ~5 hours)

### **Machine Learning Fundamentals**
1. **StatQuest - Machine Learning** (Playlist, ~8 hours)
2. **Andrew Ng - Machine Learning (Coursera)** (Free audit, ~60 hours)
3. **fast.ai - Practical Deep Learning** (Playlist, ~20 hours)

### **Deep Learning**
1. **3Blue1Brown - Neural Networks** (Series, ~2 hours)
2. **StatQuest - Deep Learning** (Playlist, ~4 hours)
3. **deeplizard - Deep Learning** (Playlist, ~15 hours)

### **Optimization**
1. **Boyd - Convex Optimization Lectures** (Playlist, ~20 hours)
2. **StatQuest - Gradient Descent** (Videos, ~1 hour)
3. **Andrew Ng - Optimization** (Coursera clips, ~3 hours)

### **ML in Practice**
1. **MLOps Playlist** (Various creators, ~10 hours)
2. **Data Preprocessing** (Krish Naik, ~3 hours)
3. **Feature Engineering** (Various, ~5 hours)

---

## 🚀 Mini-Project Themes (Aligned with Deep ML Problems)

### **Project 1: Seed-Stability Module**
- **Deep ML Problems Used:**
  - Linear Regression (Gradient Descent)
  - Calculate Accuracy Score
  - Feature Scaling
- **CMU Course Alignment:**
  - 10-725 (Optimization) - gradient descent
  - 10-701/715 (Introduction to ML) - evaluation metrics
  - 10-718 (ML in Practice) - reproducibility
- **Deliverable:** Reproducibility report with seed stability analysis

### **Project 2: Annotation Bias Audit**
- **Deep ML Problems Used:**
  - One-Hot Encoding (for categorical features)
  - Calculate Accuracy Score (for evaluation)
  - Softmax (for classification probabilities)
- **CMU Course Alignment:**
  - 10-708 (PGMs) - modeling bias
  - 10-718 (ML in Practice) - evaluation science
  - Research alignment with Dr. Shah
- **Deliverable:** Bias simulation and analysis report

### **Project 3 (Stretch): Edge RL Sandbox**
- **Deep ML Problems Used:**
  - All activation functions (ReLU, Leaky ReLU, Sigmoid)
  - Linear Regression (for value functions)
  - Matrix operations (for state representations)
- **CMU Course Alignment:**
  - 10-617/707 (Deep Learning) - neural networks
  - 10-725 (Optimization) - policy optimization
- **Deliverable:** Small RL environment with safe-agent constraints

---

## 📝 Deliverables Checklist

### **By End of Month 3:**
- [ ] All 11 Deep ML problems completed and documented
- [ ] Project 1 skeleton with seed-stability framework
- [ ] Summary document mapping problems to CMU courses

### **By End of Month 6:**
- [ ] **GitHub Portfolio:** `/prep/6-month/projects/` with:
  - [ ] Project 1: Seed-Stability Module (complete)
  - [ ] Project 2: Annotation Bias Audit (complete)
  - [ ] All Deep ML problem solutions
  - [ ] Documentation and README files
- [ ] **Summary Document:** `prep/6-month/summary.md` containing:
  - [ ] Research agenda
  - [ ] Next steps for CMU MSML labs
  - [ ] Faculty alignment (Shah, Cohen, others)
  - [ ] Course mapping and preparation notes
- [ ] **Updated CV/Resume:** Reflecting completed modules and projects
- [ ] **Blog Post/LinkedIn Update:** "6-Month Prep: My Path to CMU MSML"

---

## 🎓 CMU Course Preparation Matrix

| Deep ML Problem | 10-701/715 | 10-617/707 | 10-708 | 10-718 | 10-725 | 36-700/705 |
|----------------|-----------|-----------|--------|--------|---------|-----------|
| Calculate Accuracy Score | ✅✅ | ✅ | | ✅✅ | | ✅ |
| Feature Scaling | ✅ | | | ✅✅ | | |
| Matrix times Vector | ✅ | ✅ | | | ✅✅ | ✅ |
| Reshape Matrix | | ✅✅ | | ✅ | ✅ | |
| Linear Regression (Normal) | ✅✅ | | | | ✅✅ | |
| Linear Regression (GD) | ✅✅ | ✅ | | | ✅✅ | |
| ReLU | | ✅✅ | | | ✅ | |
| Leaky ReLU | | ✅✅ | | | | |
| Sigmoid | ✅✅ | ✅✅ | | | | |
| Softmax | ✅✅ | ✅✅ | | | | |
| One-Hot Encoding | ✅ | | | ✅✅ | | |

**Legend:** ✅ = Relevant, ✅✅ = Highly Relevant

---

## 💡 Pro Tips for Success

1. **Start with Foundations:** Don't skip Weeks 1-2 (linear algebra & probability)
2. **Code from Scratch:** Implement everything without libraries first
3. **Visualize Everything:** Use plots to understand concepts
4. **Connect to Research:** Always link problems to CMU research interests
5. **Document Everything:** Your GitHub is your portfolio
6. **Network Early:** Reach out to CMU faculty during Month 5-6
7. **Practice Explaining:** Teach concepts to others (rubber duck debugging)

---

## 📚 Additional Resources

### **Textbooks (Reference)**
- "Pattern Recognition and Machine Learning" - Bishop (for 10-701/715)
- "Deep Learning" - Goodfellow et al. (for 10-617/707)
- "Convex Optimization" - Boyd & Vandenberghe (for 10-725)
- "Probabilistic Graphical Models" - Koller & Friedman (for 10-708)

### **Online Courses (Supplement)**
- Coursera: Machine Learning (Andrew Ng) - Free audit
- fast.ai: Practical Deep Learning - Free
- edX: MIT 6.034 Artificial Intelligence - Free audit

### **Practice Platforms**
- LeetCode: Algorithm practice
- Kaggle: ML competitions and datasets
- Papers With Code: Latest research implementations

---

**Good luck with your CMU MSML preparation! 🎓🚀**

