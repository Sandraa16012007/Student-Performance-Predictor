# Student Performance Predictor

An end-to-end machine learning pipeline for predicting student academic performance using structured educational data.

This project explores both **classical machine learning** and **deep learning** approaches on the UCI Student Performance dataset, with a strong focus on preprocessing, feature engineering, model comparison, and evaluation.

---

## Project Overview

The objective is to predict whether a student will **pass or fail** based on demographic, academic, and behavioral attributes.

Two modeling scenarios were explored:

1. **With prior grades** (`G1`, `G2`)  
   - Easier prediction setting with strong academic history signals.

2. **Without prior grades** (`G1`, `G2` removed)  
   - Harder and more realistic prediction setting.

This comparison helps evaluate model robustness under different feature availability conditions.

---

## Dataset

**Dataset:** UCI Student Performance Dataset  
- Source: Kaggle / UCI Repository  
- Samples: **395**
- Features: student demographics, family background, study habits, academic history, lifestyle attributes

Target variable:
- Binary classification:
  - **1 = Pass**
  - **0 = Fail**

---

## Pipeline Architecture

## 1. Data Loading & EDA
Performed exploratory data analysis to understand feature distributions and dataset characteristics.

Tasks completed:
- Dataset inspection using:
  - `head()`
  - `shape`
  - `info()`
  - `describe()`
- Missing value analysis
- Duplicate check
- Class imbalance analysis
- Histograms
- Boxplots
- Correlation heatmap
- Outlier inspection

Key observations:
- No missing values
- No duplicates
- Moderate class imbalance
- Strong correlation of `G1`, `G2` with final grade

---

## 2. Data Preprocessing

Preprocessing steps included:

- One-hot encoding for categorical features
- Standardization using `StandardScaler`
- Train-test split (80/20)
- Stratified sampling for label balance preservation

Scaling applied only where required:
- Logistic Regression
- SVM
- MLP

No scaling used for:
- Random Forest

---

## 3. Feature Engineering

Custom engineered features were created to improve predictive performance.

Features engineered:
- `study_efficiency`
- `risk_score`
- `study_discipline`

Purpose:
- Capture behavioral and academic patterns beyond raw features.

Feature engineering was validated through Random Forest feature importance analysis.

---

## Models Implemented

### 1. Logistic Regression
Used as a linear baseline classifier.

Concepts explored:
- linear decision boundaries
- classification probabilities
- regularization

---

### 2. Support Vector Machine (SVM)
Kernel-based nonlinear classifier.

Hyperparameter tuning:
- `C`

Concepts explored:
- margin maximization
- nonlinear decision boundaries
- kernel methods

---

### 3. Random Forest
Ensemble tree-based classifier.

Hyperparameter tuning:
- `n_estimators`

Concepts explored:
- bagging
- ensemble learning
- feature importance
- variance reduction

---

### 4. Multi-Layer Perceptron (MLP)
Deep learning benchmark built using TensorFlow/Keras.

Architecture:
- 2 hidden layers
- ReLU activation
- Dropout regularization
- Adam optimizer

Training techniques:
- Early stopping
- Validation split

Concepts explored:
- backpropagation
- gradient descent
- neural network regularization

---

## Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Additional validation:
- 5-Fold Cross Validation
- Learning Curves
- Bias-Variance Analysis

---

## Results

| Model | Accuracy (With Grades) | F1 (With Grades) | Accuracy (Without Grades) | F1 (Without Grades) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8734 | 0.9020 | 0.6456 | 0.7544 |
| SVM | 0.8228 | 0.8679 | 0.6835 | 0.7934 |
| Random Forest | **0.8861** | **0.9109** | 0.6709 | 0.7797 |
| MLP | 0.8481 | 0.8846 | 0.6709 | **0.7969** |

---

## Cross Validation Results

| Model | CV F1 (With Grades) | CV F1 (Without Grades) |
|---|---:|---:|
| Logistic Regression | 0.9333 | 0.7818 |
| SVM | 0.8979 | **0.7974** |
| Random Forest | **0.9354** | 0.7890 |

---

## Key Findings

### With prior grades
**Best model:** Random Forest

Why:
- highest F1 score
- highest CV F1
- lowest variance across folds

Insight:
- prior academic grades are highly predictive.

---

### Without prior grades
**Most reliable model:** SVM

Why:
- best CV F1 under reduced feature setting

Insight:
- nonlinear models perform better when strong grade predictors are removed.

---

### Deep learning insight
MLP did not outperform Random Forest on this small tabular dataset.

Conclusion:
- neural networks are not automatically superior for structured data.

---

## Learning Curve Analysis

Learning curves were generated for:
- Logistic Regression
- Random Forest

Insights:
- Logistic Regression showed low bias and mild variance
- Random Forest showed near-perfect training performance with controlled overfitting

MLP was analyzed using:
- training loss curves
- validation loss curves
- early stopping behavior

---

## Limitations

- Small dataset size (395 samples)
- Moderate class imbalance
- Strong grade-based predictors simplify task
- Limited hyperparameter tuning

---

## Future Improvements

Possible extensions:
- GridSearchCV / RandomizedSearchCV
- SMOTE / class balancing
- Feature selection methods
- XGBoost / LightGBM benchmarking
---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Google Colab

---

## Repository Structure

```bash
student-performance-predictor/
│
├── data/
├── notebooks/
├── results/
├── README.md
└── requirements.txt
```

---

## Application Demo

<img width="1088" height="751" alt="image" src="https://github.com/user-attachments/assets/0f26a9db-4cf3-45c3-8b9d-fe15fb28e1c8" />
<img width="997" height="782" alt="image" src="https://github.com/user-attachments/assets/bb7ec94e-81db-413f-a67a-81a8319a68c5" />

---
## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

Built as a hands-on machine learning project to strengthen understanding of:

- supervised learning
- preprocessing pipelines
- model training
- evaluation metrics
- cross-validation
- deep learning fundamentals
- bias-variance tradeoff
