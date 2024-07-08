Model 1: Python code for Forex Prediction - simple model

# Python code using TensorFlow and Keras to build and train an LSTM model for foreign exchange (forex) trading prediction. 
This example assumes you have historical forex data (e.g., EUR/USD prices) in a CSV file.

### Requirements
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

# You can install the necessary packages using pip:

pip install tensorflow pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load the data
data = pd.read_csv('forex_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the data
plt.figure(figsize=(14, 5))
plt.plot(data['Close'], label='Close Price history')
plt.show()

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare the data for LSTM
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)

# Reshape X for LSTM model
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot the predictions
plt.figure(figsize=(14, 5))
plt.plot(data.index[len(data) - len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Forex Price')
plt.plot(data.index[len(data) - len(y_test):], predictions, color='red', label='Predicted Forex Price')
plt.title('Forex Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, scaler.inverse_transform(predictions)))
print(f'Root Mean Squared Error: {rmse}')


### Explanation
1. **Data Loading and Plotting**: The code loads historical forex data from a CSV file and plots the closing prices.
2. **Data Preprocessing**: The data is normalized using `MinMaxScaler` to fit within the range [0, 1].
3. **Dataset Creation**: A helper function `create_dataset` is used to create the input sequences for the LSTM model with a specified look-back period (e.g., 60 days).
4. **Model Building**: An LSTM model is built with two LSTM layers and dropout for regularization.
5. **Model Training**: The model is trained on the training data for a specified number of epochs.
6. **Prediction and Visualization**: The model's predictions are plotted against the actual prices to visualize the performance.
7. **Error Calculation**: The Root Mean Squared Error (RMSE) is calculated to evaluate the model's performance.
Make sure to replace `'forex_data.csv'` with the path to your actual forex data file. Additionally, you might need to adjust the model parameters, such as the number of LSTM units, dropout rates, and epochs, based on your specific dataset and requirements.
