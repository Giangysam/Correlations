# Predictive Quality Control for Industrial Manufacturing
## Overview
This project focuses on the development of a Machine Learning model for early defect detection in an industrial manufacturing process.
The dataset contains production parameters collected throughout the manufacturing cycle, including process measurements and operational variables such as:
- Linear measurements (microns)
- Drawing force (tons)
- Temperature
- Material thickness
- Production speed (m/s)
- Operating time (hours)
- Additional process-specific parameters

The main objective is to predict the occurrence of product defects before the end of the manufacturing process, allowing operators to take corrective actions in advance and reduce scrap, rework, and production costs.

## Business Objective
Quality defects are typically identified only at the end of the production process, when corrective actions are no longer possible for the current product.
This project aims to:
- Detect defects as early as possible during production.
- Support proactive process control.
- Identify the most influential process variables.
- Reduce waste and improve product quality.
- Increase overall manufacturing efficiency.

## Dataset
The dataset consists of historical production records collected from an industrial environment.

### Features
Examples of available process variables include:
- Process temperatures
- Material thickness measurements
- Line speed
- Drawing force (tons)
- Production duration
- Dimensional measurements in microns
- Other machine and process parameters

### Target Variable
The target is a binary classification label indicating:
- **0** → Non-defective product
- **1** → Defective product

## Machine Learning Models Evaluated
Several supervised learning algorithms have been evaluated to identify the most suitable model for defect prediction.

### 1. Random Forest
A tree-based ensemble model capable of handling nonlinear relationships and feature interactions.
**Advantages**
- Robust to noisy data
- Good interpretability through feature importance
- Limited preprocessing requirements

### 2. XGBoost
A gradient boosting algorithm designed to achieve high predictive performance through sequential optimization of decision trees.
**Advantages**
- Excellent predictive accuracy
- Handles complex nonlinear patterns
- Effective with heterogeneous industrial datasets

### 3. Logistic Regression
A baseline linear classification model used for benchmarking purposes.
**Advantages**
- Fast training
- High interpretability
- Useful baseline for performance comparison

## Evaluation Metrics
Given the importance of correctly identifying defective products, model evaluation focuses on classification performance for the defect class.
The following metrics are monitored:
### Precision
Measures the proportion of predicted defects that are actually defects.
Precision = \frac{TP}{TP + FP}
### Recall
Measures the model's ability to detect actual defects.
Recall = \frac{TP}{TP + FN}
### F1-Score
Harmonic mean of Precision and Recall.
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
The F1-Score is considered the primary optimization metric because it provides a balanced evaluation of false positives and false negatives.

## Current Results
Current best performance is obtained with **XGBoost**, achieving approximately **70% F1-Score** on defect prediction.

# Future Improvements
The current objective is to improve predictive performance and approach an **80% F1-Score**.
Potential improvement areas include:
- Advanced feature engineering
- Feature selection techniques
- Hyperparameter optimization
- Class imbalance handling
- Time-aware validation strategies
- Ensemble methods
- Explainability analysis (SHAP, Feature Importance)
- Additional process knowledge integration

## Technology Stack
- Python

## Expected Impact
An effective predictive quality model can:
- Reduce production defects
- Minimize scrap and rework costs
- Improve process stability
- Support data-driven decision making
- Enable predictive quality control in industrial environments

## Project Status

🚧 Ongoing Development
Current best model:
185
**XGBoost – F1-Score ≈ 70%**
Target:
188
**F1-Score ≥ 80%**

### Correlation Analysis
One of the main objectives of the analysis is to identify which production variables are most strongly associated with defect occurrence.
The correlation study aims to:
- Detect potential relationships between process parameters and defects.
- Identify variables with the highest predictive power.
- Support feature selection and feature engineering activities.
- Provide process insights to manufacturing engineers.
- Highlight critical process conditions that may increase defect probability.

The analysis includes:
- Correlation matrices between numerical variables.
- Statistical analysis of process parameters.
- Distribution comparison between defective and non-defective products.
- Feature importance evaluation from tree-based models.
- Investigation of multicollinearity among variables.

The results of this phase serve as a foundation for model development and help identify the most influential manufacturing parameters affecting product quality.

## Methodology
The project follows the following workflow:
1. Data collection from industrial production systems.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Correlation analysis between process variables and defects.
5. Feature engineering and feature selection.
6. Model training and validation.
7. Performance evaluation using Precision, Recall and F1-Score.
8. Model optimization and continuous improvement.

Please see the Code Review and Research Feedback issue to share comments, suggestions, or proposed improvements.
https://github.com/Giangysam/Correlations/issues/5
