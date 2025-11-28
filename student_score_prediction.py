"""
Student Score Prediction
Author : Shahil Shah
Date : 2024-06-15

Predicting Student Subject Scores(Math, reading, or writing) based on study hours using Linear Regression, and Polynomial Regression.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression        
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

valid_targets = ['math score', 'reading score', 'writing score']
print("=" * 70)
print("STUDENT SCORE PREDICTION PROJECT (Flexible Target)")
print("=" * 70)
print("\nWhich subject score do you want to predict?")
print("Options: math score, reading score, writing score")
target_col = input("Enter your target subject:").strip().lower()

if target_col not in valid_targets:
    print("Invalid choice! Defaulting to math score")
    target_col = 'math score'

feature_cols = [col for col in valid_targets if col != target_col]
print(f"\nPredicting '{target_col}' using {feature_cols}")

try:
    data = pd.read_csv('data/StudentsPerformance.csv')
    print(f"✓ Loaded {len(data)} student records")
except FileNotFoundError:
    print("ERROR: Dataset not found. Make sure StudentsPerformance.csv is in the data/ directory.")
    print("Please add your dataset file and run again.")
    exit()
    
print("\nFirst 5 rows of relevant data:")
print(data[valid_targets].head())

#Visualize Score Correlations
plt.figure(figsize=(7,6))
corr = data[valid_targets].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
plt.title("Correlation Between Subject Scores", fontsize=14)
plt.tight_layout()
plt.savefig('score_correlation.png')
print("✓ Saved score_correlation.png")
plt.show()

# Choose features and target
X = data[feature_cols]
y = data[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Samples: {len(X_train)}, Testing samples: {len(X_test)}")
       
#Linear Regression
                                                        
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_test_pred = linear_model.predict(X_test)

linear_test_r2 = r2_score(y_test, linear_test_pred)
linear_test_rmse = np.sqrt(mean_squared_error(y_test, linear_test_pred))
linear_test_mae = mean_absolute_error(y_test, linear_test_pred)

print("\nLINEAR REGRESSION RESULTS")
print(f"  Testing R² Score:  {linear_test_r2:.4f}")
print(f"    RMSE: {linear_test_rmse:.2f}")
print(f"    MAE: {linear_test_mae:.2f}")

print(f"\n Model Equation for '{target_col}':")
coef_str = "+".join([f"{c:.2f}*{f}" for c, f in zip(linear_model.coef_, feature_cols)])
print(f"{target_col.title()} = {linear_model.intercept_:.2f} + {coef_str}")

#Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_test_pred = poly_model.predict(X_test_poly)

poly_test_r2 = r2_score(y_test, poly_test_pred)
poly_test_rmse = np.sqrt(mean_squared_error(y_test, poly_test_pred))
poly_test_mae = mean_absolute_error(y_test, poly_test_pred)

print("\nPOLYNOMIAL REGRESSION (degree=2) RESULTS")
print(f"  Testing R² Score:  {poly_test_r2:.4f}")
print(f"  RMSE: {poly_test_rmse:.2f}")
print(f"  MAE:  {poly_test_mae:.2f}")

comparison_data = {
    'Model': ['Linear Regression', 'Polynomial Regression'],
    'R² Score': [linear_test_r2, poly_test_r2],
    'RMSE': [linear_test_rmse, poly_test_rmse],
    'MAE': [linear_test_mae, poly_test_mae]
}
comparison_df = pd.DataFrame(comparison_data)
print("\nMODEL COMPARISON\n", comparison_df.to_string(index=False))

if poly_test_r2 > linear_test_r2:
    print(f"\n✓ Polynomial Regression performs better!")
else:
    print(f"\n✓ Linear Regression performs better!")

# Visualization: Actual vs. Predicted
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, linear_test_pred, alpha=0.6, color='blue', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel(f'Actual {target_col.title()}')
plt.ylabel(f'Predicted {target_col.title()}')
plt.title(f'Linear Regression\nR² = {linear_test_r2:.4f}')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, poly_test_pred, alpha=0.6, color='green', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel(f'Actual {target_col.title()}')
plt.ylabel(f'Predicted {target_col.title()}')
plt.title(f'Polynomial Regression\nR² = {poly_test_r2:.4f}')
plt.grid(True)

plt.tight_layout()
plt.savefig('subject_predictions.png')
print("\n✓ Saved: subject_predictions.png")
plt.show()

# Try predicting with user input
print("\nTry your own prediction!")
try:
    input1 = float(input(f"Enter {feature_cols[0]} (0-100): "))
    input2 = float(input(f"Enter {feature_cols[1]} (0-100): "))
    test_point = [[input1, input2]]
    linear_pred = linear_model.predict(test_point)[0]
    poly_pred = poly_model.predict(poly_features.transform(test_point))[0]
    print(f"\nBased on your input:")
    print(f"  Linear Regression {target_col}: {linear_pred:.2f}")
    print(f"  Polynomial Regression {target_col}: {poly_pred:.2f}")
except Exception as e:
    print("Invalid input, skipping custom prediction.")

print("\nAll results generated. Project complete!")