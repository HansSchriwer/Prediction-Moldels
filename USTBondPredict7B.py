# Install Python Libraries
! pip install numpy pandas tensorflow requests vaderSentiment
# Coding
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TreasuryBondPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def fetch_historical_data(self, api_key, symbol="US10Y"):
        """Fetch historical bond prices from Alpha Vantage API."""
        api_url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,  # 10-year Treasury bond symbol (can adjust as needed)
            "apikey": api_key
        }

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Convert the time series data into a DataFrame
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                print("No time series data found in response.")
                return pd.DataFrame()

            df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)  # Convert index to datetime
            df = df.sort_index()  # Sort by date
            
            # Rename the '4. close' column to 'price'
            df.rename(columns={"4. close": "price"}, inplace=True)

            # Keep only the 'price' column
            df = df[['price']]
            return df
        except requests.RequestException as e:
            print(f"An error occurred while fetching historical data: {e}")
            return pd.DataFrame()

    def fetch_news_data(self, news_api_url):
        """Fetch recent news articles related to the economy."""
        try:
            response = requests.get(news_api_url)
            response.raise_for_status()
            data = response.json()
            return data['articles']  # Adjust based on your API structure
        except requests.RequestException as e:
            print(f"An error occurred while fetching news data: {e}")
            return []

    def sentiment_analysis(self, news_articles):
        """Perform sentiment analysis on news articles."""
        sentiments = []
        for article in news_articles:
            sentiment_score = self.sentiment_analyzer.polarity_scores(article['title'])
            sentiments.append(sentiment_score['compound'])
        return np.mean(sentiments)  # Return average sentiment

    def preprocess_data(self, df):
        """Preprocess the data by scaling and adding lagged features."""
        df_scaled = self.scaler.fit_transform(df)
        return df_scaled

    def create_dataset(self, df, time_step=7):
        """Create dataset for LSTM with specified time steps."""
        X, y = [], []
        for i in range(len(df) - time_step - 1):
            X.append(df[i:(i + time_step), :])
            y.append(df[i + time_step, 0])  # Assuming price is the first column
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build LSTM model."""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        """Train the LSTM model."""
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        """Make predictions using the trained model."""
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions

    def pipeline(self, api_key, news_api_url):
        """Complete pipeline for fetching data, training model, and making predictions."""
        # Fetch and preprocess historical data
        df = self.fetch_historical_data(api_key)
        if df.empty:
            return []

        news_articles = self.fetch_news_data(news_api_url)
        sentiment_score = self.sentiment_analysis(news_articles)
        df['sentiment'] = sentiment_score  # Add sentiment as a feature

        df_scaled = self.preprocess_data(df[['price']])  # Use only the price column for now
        X, y = self.create_dataset(df_scaled)

        # Split into training and testing data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train model
        self.build_model((X_train.shape[1], X_train.shape[2]))
        self.train_model(X_train, y_train)

        # Predict future prices
        predictions = self.predict(X_test)
        return predictions

# Example usage
predictor = TreasuryBondPredictor()

# Replace with your actual Alpha Vantage API key and News API URL
api_key = "YOUR_ALPHA_VANTAGE_API_KEY"
news_api_url = "https://newsapi.org/v2/everything?q=economy&apiKey=YOUR_NEWS_API_KEY"

predictions = predictor.pipeline(api_key, news_api_url)

# Print predictions for the next 7 days
print(predictions)
