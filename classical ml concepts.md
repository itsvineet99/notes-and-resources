### **Fundamental ML Concepts**

- **Types of Machine Learning**
    - Supervised Learning
    - Unsupervised Learning
    - Reinforcement Learning
- **Types of Supervised Learning**
    - Classification
    - Regression
- **Model Categories**
    - Parametric vs. Non-parametric Models
- **Gradient Descent**

###  **Data Preparation & Preprocessing**

- **Types of Data**
    - Categorical vs. Numerical
    - Ordinal vs. Nominal
- **Handling Missing Values**
    - Imputing (adding fake data instead of dropping)
        - Mean Imputation
- **Categorical Data Encoding**
    - One-Hot Encoding
    - Binary Encoding
    - Frequency Encoding
- **Data Exploration**
    - Correlation Matrix

### **Feature Engineering & Dimensionality Reduction**

- **Feature Scaling**
    - Normalization (Min-Max Scaling)
    - Standardization (Z-score Scaling)
- **Feature Selection** (Selecting a subset of features)
    - Sequential Backward Selection
- **Feature Extraction** (Creating new features from existing ones)
    - **PCA (Principal Component Analysis)**
        - Covariance Matrix
        - Projection Matrix
    - **LDA (Linear Discriminant Analysis)**
        - Mean Matrix
        - Within-Class Scatter Matrix
        - Between-Class Scatter Matrix
    - **t-SNE (t-Distributed Stochastic Neighbor Embedding)**

### **Supervised Learning Algorithms**

**A. Linear Models for Classification**
- Perceptron
- Adaline (Adaptive Linear Neuron)
- **Logistic Regression**
    - Maximum Likelihood Estimation (MLE)
    - Log Loss (Binary Cross-Entropy Loss)
    - sigmoid function 
- **Multiclass Classification Strategy**
    - One-vs-All (One-vs-Rest)

**B. Linear Models for Regression**
- **Linear Regression**
    - Simple Linear Regression
    - Multiple Linear Regression
- **Robust Linear Regression**
    - Linear Regression using RANSAC Algorithm
- **Polynomial Regression**
- **Regularized Regression Models**
    - Ridge Regression
    - Lasso Regression
    - Elastic Net

**C. Tree-Based Models**
- **Decision Trees (for Classification)**
    - Impurity Measures: Entropy, Gini Impurity, Classification Error
    - Information Gain
    - Out-of-Bag Dataset
    - Out-of-Bag Error
- **Decision Tree Regression (Regression Trees)**
- **Random Forest** (ensembled version of decision trees)

**D. Other Supervised Models**
- **SVM (Support Vector Machine)**
- **k-NN (k-Nearest Neighbors)**
    - Lazy Learning Algorithm

### **Unsupervised Learning Algorithms (Clustering)**

- **Clustering Concepts**
    - Hard Clustering vs. Soft Clustering
- **Centroid-Based Clustering**
    - **k-Means Algorithm**
    - **k-Means++** (improved version of k-means)
    - **FCM (Fuzzy C-Means)**
        - Fuzziness Coefficient
- **Hierarchical Clustering**
    - Divisive
    - Agglomerative
- **Density-Based Clustering**
    - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **Clustering Evaluation**
    - Elbow Method
    - Silhouette Analysis

### **Ensemble Learning**

- **General Concepts**
    - Majority Voting, Weighted Majority Voting, and Majority Voting using Probabilities
- **Bagging (Bootstrap Aggregating)**
- **Boosting**
    - AdaBoost
    - Gradient Boosting
    - XGBoost

### **Model Evaluation & Metrics**

**A. Validation Techniques**
- Holdout Method (simple cross-validation)
- k-Fold Cross-Validation
- Nested Cross-Validation

**B. Classification Metrics**
- Confusion Matrix
- Precision
- Recall (Sensitivity)
- F1-Score
- MCC (Matthews Correlation Coefficient)
- ROC (Receiver Operating Characteristic) Graphs
- Multiclass versions of all the above metrics

**C. Regression Metrics**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Coefficient of Determination $(R^2)$
- Residual Plots

### **Model Optimization & Advanced Workflows**

- **Core Challenges & Trade-offs**
    - Overfitting vs. Underfitting
    - Bias-Variance Tradeoff
- **Regularization** (to combat overfitting)
    - L2 Regularization (Ridge)
    - L1 Regularization (Lasso)
- **Hyperparameter Optimization**
    - Grid Search
    - Randomized Search
    - Successive Halving
- **Handling Data Challenges**
    - Class Imbalance (and tricks to deal with it)
- **Workflow & Scalability**
    - Building Pipelines (to combine estimators and transformers)
    - **Out-of-Core Learning** (for large datasets)
        - Online Learning
        - Mini-Batch Learning
        - Stochastic Gradient Descent (SGD)

### **Natural Language Processing (NLP)**

- **Text Preprocessing**
    - Word Stemming
    - Stop Words
- **Text Vectorization (Feature Extraction for Text)**
    - **Bag-of-Words Model**
        - Raw Term Frequencies (TF)
        - Term Frequency-Inverse Document Frequency (TF-IDF)
    - Hashing Vectorizer
- **Topic Modeling**
    - Latent Dirichlet Allocation (LDA)