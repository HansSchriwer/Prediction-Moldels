# To develop a machine learning program to predict US housing price developments, we need to:
# Gather and preprocess historical housing price data and economic indicators.
# Develop a model using machine learning algorithms.
# Train and evaluate the model.
# Make predictions based on current and forecasted economic data.
# We'll start with a basic implementation and then build on it for an advanced solution.

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Load Data
# Load your historical housing price data and economic indicators
housing_data = pd.read_csv('housing_data.csv')
economic_data = pd.read_csv('economic_data.csv')
# Merge datasets on a common column, such as date
data = pd.merge(housing_data, economic_data, on='date')

# Step 3: Preprocess Data
# Handle missing values
data = data.dropna()

# Feature selection
X = data[['economic_indicator_1', 'economic_indicator_2', 'economic_indicator_3']]
y = data['housing_price_index']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R2 Score: {r2}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Housing Prices')
plt.ylabel('Predicted Housing Prices')
plt.title('Actual vs Predicted Housing Prices')
plt.show()

# Simple Housing Predictor Analysis. Files 'housing_data.csv' / 'economic_data.csv' / 'current_economic_data.csv'
