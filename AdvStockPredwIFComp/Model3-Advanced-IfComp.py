# Model 3 - is the enhanced version of the advanced IF-COMP implementation that includes the calculation of RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence) 
parameters for an even more comprehensive stock price prediction model.

### Step-by-Step Guide with RSI and MACD

#### 1. Data Collection and Preprocessing
Include the computation of technical indicators RSI and MACD.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

# Load historical stock data
data = pd.read_csv('historical_stock_data.csv', index_col='Date', parse_dates=True)

# Feature engineering: Adding technical indicators
def compute_RSI(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def compute_MACD(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

data['Moving_Avg_10'] = data['Close'].rolling(window=10).mean()
data['Moving_Avg_50'] = data['Close'].rolling(window=50).mean()
data['RSI'] = compute_RSI(data['Close'], window=14)
data['MACD'] = compute_MACD(data['Close'])
data = data.dropna()

# Preprocess data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Split into train and test
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

X_train, X_test, y_train, y_test = train_test_split(train_data[:, :-1], train_data[:, -1], test_size=0.2, random_state=42)

#### 2. Model Training
Enhance the models and incorporate more diverse models for a robust prediction.

# ARIMA model
arima_model = ARIMA(y_train, order=(5, 1, 0))  # Adjust order as needed
arima_model_fit = arima_model.fit()

# Random Forest model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = create_lstm_model((X_train.shape[1], 1))
lstm_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=20, batch_size=32, validation_data=(np.expand_dims(X_test, axis=-1), y_test))

#### 3. Composite Pattern Recognition
Utilize more sophisticated pattern recognition techniques, like clustering historical patterns.

from sklearn.cluster import KMeans

# Example pattern recognition with KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(train_data)
cluster_labels = kmeans.predict(test_data)


#### 4. Integration of Predictions
Use a more advanced ensemble method to combine predictions from multiple models.

# Generate predictions
arima_pred = arima_model_fit.forecast(steps=7)
rf_pred = rf_model.predict(X_test[-7:])
gb_pred = gb_model.predict(X_test[-7:])
lstm_pred = lstm_model.predict(np.expand_dims(X_test[-7:], axis=-1))

# Advanced ensemble method: Weighted average based on model performance
weights = [0.3, 0.3, 0.2, 0.2]  # Adjust weights based on cross-validation performance
final_pred = (weights[0] * arima_pred + weights[1] * rf_pred + weights[2] * gb_pred + weights[3] * lstm_pred.squeeze())

# Rescale predictions back to original scale
final_pred_rescaled = scaler.inverse_transform(final_pred.reshape(-1, 1))

print('Advanced 7-day stock price prediction:', final_pred_rescaled)
