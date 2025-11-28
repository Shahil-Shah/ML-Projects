# Student Score Prediction

**Author:** Shahil Shah  
**Date:** November 9, 2025  
**Status:** Complete ✓

## Project Overview
A machine learning project that predicts student math scores based on their reading and writing scores using Linear and Polynomial Regression models.

## Problem Statement
Can we predict a student's performance in mathematics based on their performance in reading and writing? This project explores the relationship between these academic scores and builds predictive models.

## Dataset
- **Source:** Kaggle - Students Performance in Exams
- **Size:** 1000 student records
- **Features Used:**
  - Reading Score (0-100)
  - Writing Score (0-100)
- **Target Variable:** Math Score (0-100)

## Methodology

### 1. Exploratory Data Analysis
- Analyzed correlation between math, reading, and writing scores
- Created correlation heatmap showing strong positive relationships
- Interactive model selection allowing users to predict any subject

### 2. Data Preparation
- User selects target subject (math, reading, or writing)
- Other two subjects become features
- Split data: 80% training, 20% testing

### 3. Model Development

#### Linear Regression
- Simple linear relationship: `Target = a + b*(Feature1) + c*(Feature2)`
- Assumes straight-line relationship between variables

#### Polynomial Regression (Degree 2)
- Captures non-linear patterns and interactions
- Models curved relationships with polynomial terms

### 4. Model Evaluation
Used multiple metrics:
- **R² Score:** Proportion of variance explained (0-1, higher is better)
- **RMSE:** Root Mean Squared Error (average prediction error)
- **MAE:** Mean Absolute Error (average absolute prediction error)

## Features
- **Interactive Subject Selection:** Choose which subject to predict
- **Model Comparison:** Side-by-side Linear vs Polynomial regression
- **Custom Predictions:** Input your own scores for real-time predictions
- **Visualizations:** Correlation heatmap and prediction scatter plots

## Visualizations Created

1. **score_correlation.png** - Heatmap showing correlation between subjects
2. **subject_predictions.png** - Actual vs Predicted scores for both models

## Technologies Used
- **Python 3.11**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning models
- **matplotlib & seaborn** - Data visualization

## Project Structure

```
student-score-prediction/
├── data/
│   └── StudentsPerformance.csv
├── student_score_prediction.py
├── README.md
├── score_correlation.png
└── subject_predictions.png
```

## How to Run
1. Ensure all dependencies are installed: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
2. Run: `python student_score_prediction.py`
3. Select which subject to predict when prompted
4. View generated visualizations and model results
5. Try custom predictions with your own input values