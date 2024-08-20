# Predict US T Bond 7 days out
# Install required libraries, i.e. pandas and requests
! pip install requests pandas
# Coding possibilities
import requests
import pandas as pd

class USTreasuryBond:
    def __init__(self):
        self.columns = {
            'price': 0.0,
            'yield': 0.0,
            'dv01': 0.0,
            'expiration': None
        }

    def set_value(self, column: str, value):
        self.columns[column] = value

    def get_value(self, column: str):
        return self.columns.get(column, None)

    def get_live_price(self, api_key):
        """Fetch live treasury bond price from Alpha Vantage API."""
        api_url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": "US10Y",  # Example for 10-year Treasury bond
            "apikey": api_key
        }

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()

            # Parse the latest available closing price
            time_series = data.get("Time Series (Daily)", {})
            if time_series:
                latest_date = list(time_series.keys())[0]
                latest_close = time_series[latest_date]['4. close']
                live_price = float(latest_close)
                self.set_value('price', live_price)
                print(f"Live price updated to: {live_price}")
            else:
                print("No time series data available.")
        except requests.RequestException as e:
            print(f"An error occurred: {e}")

    def get_economic_data(self, indicator, api_key):
        """Fetch economic indicators like inflation or GDP from Alpha Vantage."""
        api_url = "https://www.alphavantage.co/query"
        params = {
            "function": indicator,
            "apikey": api_key
        }

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()

            # Extract relevant economic data
            return data
        except requests.RequestException as e:
            print(f"An error occurred while fetching economic data: {e}")
            return None

    def predict_future_price(self, economic_data):
        """Use economic data and historical prices to predict future prices."""
        # Implement your machine learning model here
        # This is a placeholder for future integration
        # Example: Use a linear regression model, LSTM, etc.

        # For now, let's mock a future price prediction
        future_price = self.get_value('price') * 1.02  # Simple mock prediction
        print(f"Predicted future price (7 days out): {future_price}")
        return future_price

# Example usage
api_key = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your actual API key

us_bond = USTreasuryBond()
us_bond.get_live_price(api_key)

# Fetch economic data as needed (e.g., inflation, GDP, etc.)
economic_data = us_bond.get_economic_data("REAL_GDP", api_key)

# Predict future price based on economic data
predicted_price = us_bond.predict_future_price(economic_data)
