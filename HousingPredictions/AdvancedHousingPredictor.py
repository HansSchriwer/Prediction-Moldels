#!/usr/bin/env python
# coding: utf-8

# In[10]:


### Advanced Implementation
#For the advanced implementation, we'll use a more sophisticated model like XGBoost, incorporate hyperparameter tuning, and add more features and economic indicators.###


# In[11]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[12]:


# Step 2: Load Data
# Load your historical housing price data and economic indicators
housing_data = pd.read_csv('housing_data.csv')
economic_data = pd.read_csv('economic_data.csv')

# Merge datasets on a common column, such as date
data = pd.merge(housing_data, economic_data, on='date')


# In[13]:


# Step 3: Preprocess Data
# Handle missing values
data = data.dropna()

# Feature selection
X = data.drop(['date', 'housing_price_index'], axis=1)
y = data['housing_price_index']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[14]:


# Step 4: Hyperparameter Tuning
# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb = XGBRegressor()
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')


# In[15]:


# Step 5: Train the Model
model = XGBRegressor(**best_params)
model.fit(X_train_scaled, y_train)


# In[16]:


# Step 6: Evaluate the Model
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


# In[17]:


# Step 7: Make Predictions
# Load current and forecasted economic data
current_economic_data = pd.read_csv('current_economic_data.csv')
current_economic_data_scaled = scaler.transform(current_economic_data)



# In[20]:


# Make predictions
housing_price_predictions = model.predict(current_economic_data_scaled)
print(f'Predicted Housing Prices: {housing_price_predictions}')


# In[22]:


# This advanced implementation incorporates hyperparameter tuning for better model performance and uses XGBoost, a powerful machine learning algorithm. 
# You can further enhance the model by adding more relevant features, performing feature engineering, and using advanced techniques like ensembling different models.#


# In[ ]:




