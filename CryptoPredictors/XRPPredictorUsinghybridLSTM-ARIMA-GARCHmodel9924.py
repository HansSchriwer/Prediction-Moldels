#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install Required Libraries
get_ipython().system(' pip install numpy pandas scikit-learn tensorflow keras statsmodels arch matplotlib')


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# 1. Load and Prepare the Data
df = pd.read_csv('ripple_2019-09-11_2024-09-09.csv')

# Ensure 'End' column is used as the date and set it as the index
df['End'] = pd.to_datetime(df['End'])
df.set_index('End', inplace=True)

# Extract close prices for LSTM and other models
data = df[['Close']]

# Plot the historical data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Historical Close Prices')
plt.title('Cryptocurrency Price History')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# 2. Prepare Data for LSTM Model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

def create_lstm_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time step and prepare training data
time_step = 60
X_train, y_train = create_lstm_dataset(scaled_data, time_step)

# Reshape input data for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 3. Build and Train the LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

# Compile the LSTM model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

# Prepare test data for LSTM (last 60 days)
test_data = scaled_data[-time_step:]
X_test = np.array([test_data])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict the next 365 days using LSTM
lstm_predictions = []
for _ in range(365):
    lstm_pred = lstm_model.predict(X_test)
    lstm_predictions.append(lstm_pred[0, 0])

    # Reshape the prediction to match the dimensions of X_test before appending
    lstm_pred_reshaped = np.reshape(lstm_pred[0, 0], (1, 1, 1))
    
    # Append new prediction to the test set and continue forecasting
    X_test = np.append(X_test[:, 1:, :], lstm_pred_reshaped, axis=1)

# Rescale LSTM predictions back to the original scale
lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))

# 4. ARIMA Model for Long-Term Trend Prediction
# Differencing to make the data stationary
diff_data = data['Close'].diff().dropna()

# Fit ARIMA model for trend prediction
arima_model = ARIMA(diff_data, order=(5, 1, 2))
arima_result = arima_model.fit()

# Forecast for 365 days using ARIMA
arima_forecast = arima_result.forecast(steps=365)
arima_forecast = arima_forecast.values  # Convert to array

# 5. GARCH Model for Volatility Prediction
# Rescale differenced data
scaled_diff_data = diff_data * 100  # Scale the data to avoid convergence issues

# Fit GARCH model for volatility prediction
garch_model = arch_model(scaled_diff_data, vol='Garch', p=1, q=1)
garch_result = garch_model.fit(disp="off")

# Forecast volatility for 365 days using GARCH
garch_forecast = garch_result.forecast(horizon=365)
volatility_forecast = garch_forecast.variance[-1:].values.flatten()  # Get the variances

# Rescale volatility forecast back to original scale
volatility_forecast = volatility_forecast / (100 ** 2)

# 6. Combine LSTM, ARIMA, and GARCH Predictions
combined_forecast = []
for i in range(365):
    combined_pred = lstm_predictions[i] + arima_forecast[i] + np.sqrt(volatility_forecast[i])
    combined_forecast.append(combined_pred)

# Convert combined_forecast to a numpy array
combined_forecast = np.array(combined_forecast)

# Create future dates for the forecast
future_dates = pd.date_range(data.index.max(), periods=365)

# 7. Plot the Combined Forecast
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Historical Close Prices')
plt.plot(future_dates, combined_forecast, label='Hybrid Model Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Cryptocurrency Price Forecast (Hybrid LSTM-ARIMA-GARCH)')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




