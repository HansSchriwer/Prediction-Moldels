#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Food ingredients and Dementia


# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets (replace 'file_path' with actual paths to your data)
# Data should include food ingredients over time and dementia cases over time
food_data = pd.read_csv('path to food nutrient.csv')  # e.g., sugar, fat, preservatives per year
dementia_data = pd.read_csv('path to dementia data.csv')  # e.g., dementia rates per year

# Display first few rows of the data to understand structure
print(food_data.head())
print(dementia_data.head())

# Data Preprocessing: Ensure both datasets have a common "Year" column for merging
merged_data = pd.merge(food_data, dementia_data, on='Year')

# Correlation Analysis
# Select food composition features and dementia rates for correlation analysis
features = ['Sugar', 'Fat', 'Preservatives', 'Salt', 'Calories']  # Modify according to your dataset
target = 'Dementia_Prevalence'

# Check correlation matrix
correlation_matrix = merged_data[features + [target]].corr()
print("Correlation Matrix:\n", correlation_matrix)

# Visualize correlation matrix using heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Food Composition and Dementia Prevalence')
plt.show()

# Linear Regression Analysis to determine if a trend exists
X = merged_data[features]
y = merged_data[target]

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plotting true vs predicted dementia rates
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Dementia Prevalence')
plt.ylabel('Predicted Dementia Prevalence')
plt.title('Actual vs Predicted Dementia Prevalence')
plt.show()

# Visualize the coefficients (importance of each food composition variable)
coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print("Coefficients of Food Composition Features:\n", coefficients)

# Visualize the coefficients
plt.figure(figsize=(8, 6))
sns.barplot(x=coefficients.index, y='Coefficient', data=coefficients)
plt.title('Impact of Food Composition on Dementia Prevalence')
plt.xlabel('Food Composition Feature')
plt.ylabel('Coefficient')
plt.show()


# In[ ]:


### Key Points of the Code:
Data Preprocessing: It merges two datasets (food composition and dementia prevalence) on a common "Year" column.
Correlation Analysis: The correlation matrix shows the relationships between different food ingredients and dementia prevalence.
Linear Regression: This model tries to predict dementia prevalence based on the composition of food ingredients. The r2_score indicates how well the model fits the data.
Visualization: It includes several visualizations such as the heatmap for correlation, scatter plot for model predictions, and bar plot for feature importance.###


# In[ ]:




