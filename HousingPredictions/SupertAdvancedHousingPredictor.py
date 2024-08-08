#!/usr/bin/env python
# coding: utf-8

# In[6]:


# To enhance the model, we can incorporate more features, perform feature engineering, and use advanced techniques like ensembling different models. 
# Here’s an enhanced version of the previous implementation:


# In[7]:


# Import Libraries


# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[9]:


# Load Data


# In[10]:


# Load your historical housing price data and economic indicators
housing_data = pd.read_csv('housing_data.csv')
economic_data = pd.read_csv('economic_data.csv')

# Merge datasets on a common column, such as date
data = pd.merge(housing_data, economic_data, on='date')


# In[11]:


# Preprocess Data


# In[12]:


# Handle missing values
data = data.dropna()

# Feature selection
X = data.drop(['date', 'housing_price_index'], axis=1)
y = data['housing_price_index']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Feature Engineering


# In[14]:


# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Feature scaling
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)


# In[15]:


# Hyperparameter Tuning


# In[16]:


# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb = XGBRegressor()
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, scoring='r2', n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train_poly_scaled, y_train)

best_params_xgb = grid_search_xgb.best_params_
print(f'Best parameters for XGBoost: {best_params_xgb}')


# In[17]:


# Model Training and Ensembling


# In[18]:


# Train individual models with the best parameters
xgb_model = XGBRegressor(**best_params_xgb)
xgb_model.fit(X_train_poly_scaled, y_train)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
rf_model.fit(X_train_poly_scaled, y_train)

gbr_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gbr_model.fit(X_train_poly_scaled, y_train)

# Ensemble model
ensemble_model = VotingRegressor(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('gbr', gbr_model)
])

ensemble_model.fit(X_train_poly_scaled, y_train)


# In[19]:


# Model Evaluation


# In[20]:


y_pred_ensemble = ensemble_model.predict(X_test_poly_scaled)

mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

print(f'Ensemble Model MAE: {mae_ensemble}')
print(f'Ensemble Model MSE: {mse_ensemble}')
print(f'Ensemble Model R2 Score: {r2_ensemble}')

plt.scatter(y_test, y_pred_ensemble)
plt.xlabel('Actual Housing Prices')
plt.ylabel('Predicted Housing Prices')
plt.title('Actual vs Predicted Housing Prices (Ensemble Model)')
plt.show()


# In[21]:


# Make Predictions


# In[22]:


# Load current and forecasted economic data
current_economic_data = pd.read_csv('current_economic_data.csv')
current_economic_data_poly = poly.transform(current_economic_data)
current_economic_data_scaled = scaler.transform(current_economic_data_poly)

# Make predictions
housing_price_predictions = ensemble_model.predict(current_economic_data_scaled)
print(f'Predicted Housing Prices: {housing_price_predictions}')


# In[ ]:





# In[ ]:





# In[ ]:




