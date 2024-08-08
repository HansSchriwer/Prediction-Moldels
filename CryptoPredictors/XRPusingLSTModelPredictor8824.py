#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[34]:


pip install yfinance pandas


# In[35]:


import yfinance as yf
import pandas as pd

# Define the ticker symbol for XRP
ticker = 'XRP-USD'

# Fetch historical market data for the past 7 years
xrp_data = yf.download(ticker, start='2017-01-01', end='2024-08-01')

# Ensure the Date column is parsed and set as index
xrp_data.reset_index(inplace=True)

# Save to CSV
xrp_data.to_csv('xrp_data.csv', index=False)

# Display the first few rows of the dataframe
print(xrp_data.head())


# In[42]:


# Load your data
data = pd.read_csv('xrp_data.csv', parse_dates=['Date'])

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Select only the 'Close' column
data = data[['Close']]

# Display the first few rows to ensure it's loaded correctly
print(data.head())


# In[43]:


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# In[44]:


# Prepare the training and testing datasets
train_size = int(len(scaled_data) * 0.80)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60  # 60 days
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[45]:


# Reshape the data to be compatible with LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[46]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)


# In[47]:


# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[48]:


# Transform back to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[49]:


# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(data.index, scaler.inverse_transform(scaled_data), label='Original Price')
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, :] = train_predict
plt.plot(data.index, train_plot, label='Train Predict')
test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict
plt.plot(data.index, test_plot, label='Test Predict')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




