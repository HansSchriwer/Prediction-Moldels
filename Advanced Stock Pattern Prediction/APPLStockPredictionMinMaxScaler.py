#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install scikit-learn')


# In[3]:


import requests

def get_apple_stock_data():
    api_key = "50ed4e9c8cca4a7881a9f001e650429d"  # Replace with your actual API key
    symbol = "AAPL"  # Apple stock symbol

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    # Process the data as needed (e.g., extract historical prices)
    # ...

    return data

apple_data = get_apple_stock_data()
print(apple_data)


# In[5]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load historical Apple stock data (replace with your data)
# Example data loading (uncomment and replace with actual data loading code)
# apple_data = pd.read_csv('apple_stock_data.csv')

# For demonstration purposes, let's create a dummy DataFrame
# Uncomment the above line and comment this block when using actual data
dates = pd.date_range(start='2020-01-01', periods=100)
prices = np.random.lognormal(mean=0, sigma=0.02, size=len(dates)) * 150
apple_data = pd.DataFrame(data={'date': dates, 'close': prices})
apple_data.set_index('date', inplace=True)

# Check if the 'close' column exists
if 'close' not in apple_data.columns:
    raise KeyError("The 'close' column is not present in the data.")

# Normalize data
scaler = MinMaxScaler()
try:
    scaled_data = scaler.fit_transform(apple_data["close"].values.reshape(-1, 1))
except KeyError as e:
    print(f"Error: {e}")
    scaled_data = None

# Ensure scaled_data is defined
if scaled_data is None:
    raise ValueError("Scaled data is not defined. Please check the input data.")

# Create sequences for LSTM
sequence_length = 10
sequences = []
for i in range(len(scaled_data) - sequence_length):
    sequences.append(scaled_data[i : i + sequence_length])

X = np.array(sequences)
y = scaled_data[sequence_length:]

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation="relu", input_shape=(sequence_length, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=10, batch_size=32)

# Predict future stock prices
future_data = apple_data["close"].values[-sequence_length:]
future_data = scaler.transform(future_data.reshape(-1, 1))
future_data = future_data.reshape(1, sequence_length, 1)
predicted_price = model.predict(future_data)
predicted_price = scaler.inverse_transform(predicted_price)

print("Predicted Apple stock price:", predicted_price[0][0])

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(apple_data.index, apple_data["close"], label='Actual Prices')
plt.axvline(x=apple_data.index[-1], color='r', linestyle='--', label='Prediction Point')
plt.scatter(apple_data.index[-1], predicted_price[0][0], color='r', label='Predicted Price', zorder=5)
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

