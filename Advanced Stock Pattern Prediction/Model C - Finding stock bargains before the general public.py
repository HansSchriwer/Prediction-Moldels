# Model C:
To create a more advanced model using LightGBM and neural networks, we will:
1.	Enhance Data Preparation: Include moving averages, volatility, and sector performance.
2.	Train LightGBM Model: Utilize LightGBM for classification.
3.	Train Neural Network: Build a neural network using TensorFlow or Keras.
4.	Evaluate Models: Compare performance and evaluate predictions.

# Step 1: Install Necessary Libraries

pip install requests pandas lightgbm tensorflow

# Step 2: Enhanced Code with LightGBM and Neural Network

import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define your API keys (replace 'YOUR_API_KEY' with actual API keys)
ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY'
IEX_CLOUD_API_KEY = 'YOUR_API_KEY'

# Function to fetch stock data from Alpha Vantage
def fetch_alpha_vantage_data(symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    return data

# Function to fetch stock data from IEX Cloud
def fetch_iex_cloud_data(symbol):
    url = f'https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_KEY}'
    response = requests.get(url)
    data = response.json()
    return data

# Function to fetch historical stock data for moving averages and volatility
def fetch_historical_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    time_series = data.get('Time Series (Daily)', {})
    df = pd.DataFrame(time_series).T
    df = df.rename(columns=lambda x: x.split(' ')[1])
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df

# Function to calculate moving averages and volatility
def calculate_indicators(df):
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['MA_200'] = df['close'].rolling(window=200).mean()
    df['Volatility'] = df['close'].rolling(window=50).std()
    return df

# List of stock symbols to evaluate
symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']  # Add more symbols as needed

# Function to fetch and prepare data
def prepare_data(symbols):
    data_list = []
    for symbol in symbols:
        try:
            av_data = fetch_alpha_vantage_data(symbol)
            iex_data = fetch_iex_cloud_data(symbol)
            historical_data = fetch_historical_data(symbol)
            historical_data = calculate_indicators(historical_data)
            
            pe_ratio = float(av_data.get('PERatio', '0'))
            market_cap = float(av_data.get('MarketCapitalization', '0'))
            sector = av_data.get('Sector', 'Unknown')
            latest_price = float(iex_data.get('latestPrice', '0'))
            week_high_52 = float(iex_data.get('week52High', '0'))
            week_low_52 = float(iex_data.get('week52Low', '0'))
            ma_50 = historical_data['MA_50'][-1]
            ma_200 = historical_data['MA_200'][-1]
            volatility = historical_data['Volatility'][-1]
            
            bargain = 1 if latest_price < week_low_52 * 1.1 else 0  # Binary classification for bargain
            data_list.append({
                'Symbol': symbol,
                'PE_Ratio': pe_ratio,
                'Market_Cap': market_cap,
                'Sector': sector,
                '52_Week_High': week_high_52,
                '52_Week_Low': week_low_52,
                'Latest_Price': latest_price,
                'MA_50': ma_50,
                'MA_200': ma_200,
                'Volatility': volatility,
                'Bargain': bargain
            })
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    return pd.DataFrame(data_list)

# Fetch and prepare data
df = prepare_data(symbols)

# Encode categorical data
df = pd.get_dummies(df, columns=['Sector'])

# Feature selection
features = ['PE_Ratio', 'Market_Cap', '52_Week_High', '52_Week_Low', 'Latest_Price', 'MA_50', 'MA_200', 'Volatility']
features += [col for col in df.columns if col.startswith('Sector_')]
X = df[features]
y = df['Bargain']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### LightGBM Model ###

# Train the LightGBM classifier
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
lgbm_model = lgb.train(params, lgb_train, valid_sets=lgb_eval, num_boost_round=100, early_stopping_rounds=10)

# Make predictions with LightGBM
lgbm_predictions = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
lgbm_predictions = [1 if x > 0.5 else 0 for x in lgbm_predictions]

# Evaluate LightGBM model
print("LightGBM Model:")
print(classification_report(y_test, lgbm_predictions))
print(f"Accuracy: {accuracy_score(y_test, lgbm_predictions)}")

### Neural Network Model ###

# Build the neural network model
nn_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the neural network model
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions with the neural network model
nn_predictions = nn_model.predict(X_test)
nn_predictions = [1 if x > 0.5 else 0 for x in nn_predictions]

# Evaluate neural network model
print("Neural Network Model:")
print(classification_report(y_test, nn_predictions))
print(f"Accuracy: {accuracy_score(y_test, nn_predictions)}")

# Function to evaluate new stocks using both models
def evaluate_new_stocks(new_symbols):
    new_data = prepare_data(new_symbols)
    new_data = pd.get_dummies(new_data, columns=['Sector'])
    new_features = new_data[features]
    
    lgbm_new_predictions = lgbm_model.predict(new_features, num_iteration=lgbm_model.best_iteration)
    lgbm_new_predictions = [1 if x > 0.5 else 0 for x in lgbm_new_predictions]
    
    nn_new_predictions = nn_model.predict(new_features)
    nn_new_predictions = [1 if x > 0.5 else 0 for x in nn_new_predictions]
    
    new_data['LGBM_Prediction'] = lgbm_new_predictions
    new_data['NN_Prediction'] = nn_new_predictions
    
    return new_data

# Evaluate and display potential bargains for new symbols
new_symbols = ['FB', 'NFLX', 'NVDA']  # Add more symbols as needed
bargain_stocks = evaluate_new_stocks(new_symbols)
print(bargain_stocks)

# Explanation
1.	Data Preparation:
o	Fetches financial and historical stock data.
o	Calculates moving averages and volatility.
o	Encodes sector performance using one-hot encoding.
2.	LightGBM Model:
o	Trains a LightGBM model using the prepared data.
o	Evaluates the model and makes predictions.
3.	Neural Network Model:
o	Builds and trains a neural network using TensorFlow/Keras.
o	Evaluates the model and makes predictions.
4.	Evaluation:
o	Functions to evaluate new stocks using both models.

# Comments
•	Data Sources: Ensure reliable and up-to-date data sources.
•	Model Tuning: Fine-tune hyperparameters for both models to improve performance.
•	Feature Engineering: Explore additional features for better predictions.
•	Advanced Models: Consider integrating ensemble methods or other advanced techniques for further improvements.
This enhanced approach provides a robust framework for identifying potential stock bargains using advanced machine learning models.
