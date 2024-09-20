#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Overview of the Advanced Strategy A
# Data acquisition/Feature Engenieering/Machine Learning model/Backtesting/Risk Management/Visualization & Performance Metrics


# In[ ]:


# Load libraries
get_ipython().system(' pip install pandas numpy yfinance matplotlib scikit-learn ta nltk')


# In[ ]:


get_ipython().system(' pip install pandas numpy yfinance matplotlib scikit-learn ta nltk')


# In[ ]:


# Implementation
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from ta import trend, momentum, volatility
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import datetime
import requests
from bs4 import BeautifulSoup
import time

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Parameters
INITIAL_CAPITAL = 10000
STOCKS = ['SMCI', 'TGTX']  # Add cryptocurrency tickers if desired (e.g., 'BTC-USD', 'ETH-USD')
START_DATE = '2020-01-01'
END_DATE = datetime.datetime.today().strftime('%Y-%m-%d')
SHORT_WINDOW = 50
LONG_WINDOW = 200
BUY_SPACING_DAYS = 30  # Minimum days between buys for the same asset
TEST_SIZE = 0.3  # For train-test split in ML
RANDOM_STATE = 42

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to fetch historical data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data['Adj Close'], data['High'], data['Low'], data['Volume']

# Function to calculate technical indicators
def calculate_indicators(df):
    indicators = pd.DataFrame(index=df.index)
    
    # Moving Averages
    indicators['SMA_50'] = trend.SMAIndicator(close=df, window=50).sma_indicator()
    indicators['SMA_200'] = trend.SMAIndicator(close=df, window=200).sma_indicator()
    
    # RSI
    indicators['RSI'] = momentum.RSIIndicator(close=df, window=14).rsi()
    
    # MACD
    macd = trend.MACD(close=df)
    indicators['MACD'] = macd.macd()
    indicators['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = volatility.BollingerBands(close=df, window=20, window_dev=2)
    indicators['BB_High'] = bollinger.bollinger_hband()
    indicators['BB_Low'] = bollinger.bollinger_lband()
    
    # ATR
    indicators['ATR'] = volatility.AverageTrueRange(high=df, low=df, close=df, window=14).average_true_range()
    
    return indicators

# Function to perform sentiment analysis (simplified for demonstration)
def fetch_sentiment(ticker):
    """
    Fetches sentiment for a given ticker from a news source.
    This is a placeholder function. Implement specific scraping/parsing as needed.
    """
    # Example: Scrape headlines from a financial news website
    # This requires handling specific website structures and may need updates
    try:
        url = f'https://finance.yahoo.com/quote/{ticker}/news?p={ticker}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('h3', {'class': 'Mb(5px)'})
        sentiments = []
        for headline in headlines[:5]:  # Analyze top 5 headlines
            text = headline.get_text()
            sentiment = sentiment_analyzer.polarity_scores(text)
            sentiments.append(sentiment['compound'])
        avg_sentiment = np.mean(sentiments) if sentiments else 0
    except Exception as e:
        print(f"Error fetching sentiment for {ticker}: {e}")
        avg_sentiment = 0
    return avg_sentiment

# Fetch historical data
prices, highs, lows, volumes = fetch_data(STOCKS, START_DATE, END_DATE)

# Calculate technical indicators for each stock
for stock in STOCKS:
    indicators = calculate_indicators(prices[stock])
    prices = prices.join(indicators, rsuffix=f'_{stock}')

# Drop rows with NaN values
prices.dropna(inplace=True)

# Feature Engineering
feature_list = []
for stock in STOCKS:
    feature_list.extend([
        f'SMA_50_{stock}',
        f'SMA_200_{stock}',
        f'RSI_{stock}',
        f'MACD_{stock}',
        f'MACD_Signal_{stock}',
        f'BB_High_{stock}',
        f'BB_Low_{stock}',
        f'ATR_{stock}'
    ])

# Initialize DataFrame for features
features = pd.DataFrame(index=prices.index)

# Populate features
for stock in STOCKS:
    features[f'SMA_50_{stock}'] = prices[f'SMA_50']
    features[f'SMA_200_{stock}'] = prices[f'SMA_200']
    features[f'RSI_{stock}'] = prices[f'RSI']
    features[f'MACD_{stock}'] = prices[f'MACD']
    features[f'MACD_Signal_{stock}'] = prices[f'MACD_Signal']
    features[f'BB_High_{stock}'] = prices[f'BB_High']
    features[f'BB_Low_{stock}'] = prices[f'BB_Low']
    features[f'ATR_{stock}'] = prices[f'ATR']

# Add sentiment features (simplified; in practice, ensure synchronization and data integrity)
sentiment_scores = {}
for stock in STOCKS:
    sentiments = []
    for date in prices.index:
        sentiments.append(fetch_sentiment(stock))
        time.sleep(1)  # To prevent overwhelming the server; adjust as needed
    sentiment_scores[f'Sentiment_{stock}'] = sentiments

sentiment_df = pd.DataFrame(sentiment_scores, index=prices.index)
features = features.join(sentiment_df)

# Define target variable
# For simplicity, we'll define a 'Buy' signal when the next day's return is positive
target = pd.Series(0, index=prices.index)
for stock in STOCKS:
    future_returns = prices[stock].pct_change().shift(-1)
    target = target | (future_returns > 0).astype(int)

# Align features and target
data = features.join(target.rename('Target'))
data.dropna(inplace=True)

# Split into features and target
X = data.drop('Target', axis=1)
y = data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Backtesting Strategy
class Portfolio:
    def __init__(self, initial_capital, stocks):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = {stock: 0 for stock in stocks}
        self.total = initial_capital
        self.history = []

    def buy(self, stock, price, shares):
        cost = price * shares
        if self.cash >= cost:
            self.cash -= cost
            self.holdings[stock] += shares
            print(f"Bought {shares} shares of {stock} at ${price:.2f} on {self.current_date.date()}")
        else:
            print(f"Insufficient cash to buy {shares} shares of {stock} on {self.current_date.date()}")

    def update_total(self, prices):
        total = self.cash
        for stock, shares in self.holdings.items():
            total += shares * prices[stock]
        self.total = total
        self.history.append({'Date': self.current_date, 'Total': self.total})

    def set_date(self, date):
        self.current_date = date

# Initialize portfolio
portfolio = Portfolio(INITIAL_CAPITAL, STOCKS)

# Tracking last buy date for spacing
last_buy_date = {stock: None for stock in STOCKS}

# Iterate through the test set for backtesting
for idx, row in X_test.iterrows():
    portfolio.set_date(idx)
    # Predict whether to buy
    prediction = model.predict(row.values.reshape(1, -1))[0]
    
    if prediction == 1:
        for stock in STOCKS:
            # Check if SMA crossover condition is met as an additional filter
            # You can adjust this logic as needed
            if row[f'SMA_50_{stock}'] > row[f'SMA_200_{stock}']:
                # Check buy spacing
                if last_buy_date[stock] is None or (portfolio.current_date - last_buy_date[stock]).days >= BUY_SPACING_DAYS:
                    # Determine position size based on available cash and ATR
                    atr = row[f'ATR_{stock}']
                    if atr > 0:
                        position_size = (INITIAL_CAPITAL / (len(STOCKS) * 4)) / atr
                        shares_to_buy = int(position_size // row[f'ATR_{stock}'])  # Simplified position sizing
                        price = prices.loc[idx, stock]
                        portfolio.buy(stock, price, shares_to_buy)
                        last_buy_date[stock] = portfolio.current_date
    
    # Update portfolio total
    current_prices = prices.loc[idx]
    portfolio.update_total(current_prices)

# Create a DataFrame for portfolio history
portfolio_history = pd.DataFrame(portfolio.history)
portfolio_history.set_index('Date', inplace=True)

# Plot the portfolio value over time
plt.figure(figsize=(14, 7))
plt.plot(portfolio_history['Total'], label='Portfolio Value')
plt.title('Advanced Genesis Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Print final portfolio value
print(f"Final Portfolio Value: ${portfolio_history['Total'][-1]:.2f}")

# Calculate returns
initial_value = portfolio_history['Total'].iloc[0]
final_value = portfolio_history['Total'].iloc[-1]
total_return = ((final_value - initial_value) / initial_value) * 100
print(f"Total Return: {total_return:.2f}%")

# Additional Performance Metrics
# Calculate daily returns
portfolio_history['Daily_Return'] = portfolio_history['Total'].pct_change()
cumulative_return = (1 + portfolio_history['Daily_Return']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(cumulative_return, label='Cumulative Returns')
plt.title('Cumulative Returns of the Advanced Genesis Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

