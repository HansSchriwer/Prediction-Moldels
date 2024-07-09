Model B:
We'll build on the previous example by adding more advanced features like trading volume, RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), and market sentiment. 
We'll also include hyperparameter tuning and use the Temporal Fusion Transformer (TFT) for the prediction model.

# Step 1: Install Necessary Libraries

pip install pandas numpy scikit-learn tensorflow yfinance ta transformers

# Step 2: Data Collection and Feature Engineering

import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

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
df['RSI'] = RSIIndicator(df['Close']).rsi()
df['MACD'] = MACD(df['Close']).macd_diff()
df.dropna(inplace=True)

# Placeholder function for market sentiment
def fetch_market_sentiment(date):
    # Replace with actual sentiment fetching logic
    return np.random.rand()

df['Sentiment'] = df['Date'].apply(fetch_market_sentiment)

# Select features and target
features = ['Close', 'Volume', 'MA_50', 'MA_200', 'Volatility', 'RSI', 'MACD', 'Sentiment']
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
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = build_lstm_model((window_size, len(features)))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the LSTM model
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the LSTM model
predicted_prices = lstm_model.predict(X_test)
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
future_predictions = predict_future_prices(lstm_model, future_scaled_data, window_size, future_steps)
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
Explanation
Data Collection and Feature Engineering:

# We fetch historical price data for Bitcoin and calculate additional features: MA_50, MA_200, Volatility, RSI, and MACD.
A placeholder function simulates fetching market sentiment.
Data Normalization:

# We normalize the data using MinMaxScaler.
Data Preparation:

# We prepare the data for the LSTM model by creating sequences of past data to predict future prices.
Model Building:

# A two-layer LSTM model with dropout layers to prevent overfitting is built using TensorFlow/Keras.
Training and Evaluation:

# The model is trained and evaluated on the training and testing datasets.
Future prices are predicted using the trained model.

# Step 3: Adding Hyperparameter Tuning and Temporal Fusion Transformer (TFT)
For hyperparameter tuning and using TFT, we need to install additional libraries and implement these advanced techniques.

# Install Necessary Libraries

pip install optuna tft-models

# Hyperparameter Tuning with Optuna

import optuna
from sklearn.model_selection import train_test_split

def objective(trial):
    # Define the hyperparameters to tune
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units = trial.suggest_int('units', 32, 128)
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    model = Sequential()
    for _ in range(n_layers):
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(units))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    return val_loss

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Best hyperparameters: ', study.best_params)
Temporal Fusion Transformer (TFT) Implementation
To implement a TFT model, you can use the tft-models library or the PyTorch-based pytorch-forecasting library. Here's a basic example with tft-models.

# Install Necessary Libraries

pip install pytorch-forecasting pytorch-lightning
Implementing TFT

import torch
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping

# Prepare data for TFT
df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days
df['group'] = 0  # Single time series

max_encoder_length = 30
max_prediction_length = 30

training_cutoff = df['time_idx'].max() - max_prediction_length
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx='time_idx',
    target='Close',
    group_ids
