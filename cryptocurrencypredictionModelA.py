# Model A:
Advanced Features for crypto predictions.
A very advanced cryptocurrency price prediction model can leverage machine learning techniques such as Long Short-Term Memory (LSTM) networks or Temporal Fusion Transformers (TFT). 
For this example, we'll use an LSTM model as it's well-suited for time-series forecasting. We'll also use historical price data and relevant features such as trading volume, moving averages, and volatility.

# Step 1: Install Necessary Libraries

pip install pandas numpy scikit-learn tensorflow yfinance

# Step 2: Data Collection and Preparation
We'll use yfinance to fetch historical cryptocurrency data. This example uses Bitcoin (BTC) data, but you can adapt it for other cryptocurrencies.

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Fetch historical data
def fetch_crypto_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)
    return df

# Fetch data for Bitcoin
symbol = 'BTC-USD'
start_date = '2020-01-01'
end_date = '2024-01-01'
df = fetch_crypto_data(symbol, start_date, end_date)

# Feature engineering
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()
df['Volatility'] = df['Close'].rolling(window=50).std()
df.dropna(inplace=True)

# Select features and target
features = ['Close', 'Volume', 'MA_50', 'MA_200', 'Volatility']
target = 'Close'

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Prepare the data for LSTM
def prepare_data(data, target_col, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, target_col])
    return np.array(X), np.array(y)

window_size = 30
X, y = prepare_data(scaled_data, features.index(target), window_size)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, len(features))),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(np.concatenate([predicted_prices, np.zeros((predicted_prices.shape[0], len(features)-1))], axis=1))[:, 0]

# Prepare data for future prediction
def predict_future_prices(model, data, window_size, steps):
    future_predictions = []
    current_window = data[-window_size:]

    for _ in range(steps):
        prediction = model.predict(current_window[np.newaxis, :, :])[0, 0]
        future_predictions.append(prediction)
        
        # Update the current window
        new_row = np.concatenate([[prediction], current_window[-1, 1:]])
        current_window = np.append(current_window[1:], [new_row], axis=0)

    return future_predictions

# Predict future prices for the next 30 days
future_steps = 30
future_scaled_data = np.concatenate([scaled_data, np.zeros((future_steps, scaled_data.shape[1]))], axis=0)
future_predictions = predict_future_prices(model, future_scaled_data, window_size, future_steps)
future_predictions = scaler.inverse_transform(np.concatenate([np.array(future_predictions).reshape(-1, 1), np.zeros((future_steps, len(features)-1))], axis=1))[:, 0]

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(df['Date'][-len(y_test):], scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features)-1))], axis=1))[:, 0], color='blue', label='Actual Prices')
plt.plot(df['Date'][-len(y_test):], predicted_prices, color='red', label='Predicted Prices')
plt.plot(pd.date_range(start=df['Date'].iloc[-1], periods=future_steps+1)[1:], future_predictions, color='green', label='Future Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{symbol} Price Prediction')
plt.legend()
plt.show()

# Explanation
Data Collection:
We use yfinance to fetch historical price data for Bitcoin (BTC-USD).
Feature Engineering:
Moving averages (MA_50, MA_200) and volatility are calculated and added as features.
Data Normalization:
The data is normalized using MinMaxScaler to improve model performance.
Data Preparation:
# We prepare the data for the LSTM model by creating sequences of past data to predict future prices.
Model Building:
A two-layer LSTM model with dropout layers to prevent overfitting is built using TensorFlow/Keras.
# Training and Evaluation:
The model is trained and evaluated on the training and testing datasets.
Future prices are predicted using the trained model.
# Visualization:
The actual, predicted, and future prices are plotted for visualization.
# Comments
Feature Engineering: Additional features such as trading volume, technical indicators (RSI, MACD), and market sentiment can further improve prediction accuracy.
Hyperparameter Tuning: Experiment with different model architectures, hyperparameters, and optimization techniques to improve performance.
Advanced Models: Consider using Temporal Fusion Transformers (TFT) for more sophisticated time-series forecasting.
