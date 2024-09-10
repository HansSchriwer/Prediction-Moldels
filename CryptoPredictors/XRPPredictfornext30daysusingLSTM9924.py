#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Install the required libraries: Run this in your terminal:
get_ipython().system(' pip install tensorflow scikit-learn')


# In[4]:


get_ipython().system(' pip install tensorflow scikit-learn matplotlib pandas')


# In[8]:


# LSTM Model Implementation: You can use the following code to train the LSTM model and predict the next 30 days of XRP prices.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load your data (replace with your own data if necessary)
xrp_data = pd.read_csv('ripple_2019-09-11_2024-09-09.csv')

# Use the 'End' column as the date
xrp_data['End'] = pd.to_datetime(xrp_data['End'])
close_price = xrp_data['Close'].values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_price.reshape(-1, 1))

# Prepare the data for the LSTM model
def create_lstm_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:i + time_step, 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # We use 60 days of historical data to predict the next price
X_train, y_train = create_lstm_dataset(scaled_data, time_step)

# Reshape the input for LSTM (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

# Prepare test data (the last 60 days of data)
test_data = scaled_data[-time_step:]
X_test = []
X_test.append(test_data)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict the next 30 days
predicted_prices = []
for _ in range(30):
    prediction = model.predict(X_test)
    predicted_prices.append(prediction[0, 0])

    # Reshape the prediction to match the dimensions of X_test before appending
    prediction_reshaped = np.reshape(prediction[0, 0], (1, 1, 1))
    X_test = np.append(X_test[:, 1:, :], prediction_reshaped, axis=1)

# Rescale the predicted prices back to the original scale
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Create future dates for the predictions
future_dates = pd.date_range(xrp_data['End'].max(), periods=30, freq='D')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(xrp_data['End'], close_price, label='Historical Close Price')
plt.plot(future_dates, predicted_prices, label='Predicted Future Price (LSTM)', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.title('XRP Price Prediction for the Next 30 Days using LSTM')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




