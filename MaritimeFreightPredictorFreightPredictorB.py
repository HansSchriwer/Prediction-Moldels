import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt

# Load your actual data (replace with your dataset)
data = pd.read_csv('your_data.csv')

# Feature engineering (add more relevant features)
# Handle missing data (e.g., forward fill or interpolate)
data.fillna(method='ffill', inplace=True)

# Create time-series data for modeling
time_series_data = data.set_index('Date')  # Assuming you have a 'Date' column
time_series_data.sort_index(inplace=True)

# Train a linear regression model (as before)
X_train = train_data.drop(columns=['Freight_Price'])
y_train = train_data['Freight_Price']
model = LinearRegression()
model.fit(X_train, y_train)

# Time-series modeling (using Exponential Smoothing)
model_ts = ExponentialSmoothing(time_series_data['Freight_Price'], trend='add', seasonal='add', seasonal_periods=12)
model_ts_fit = model_ts.fit()

# Predictions for the next 3 months
future_dates = pd.date_range(start=time_series_data.index[-1], periods=90, freq='D')
forecast = model_ts_fit.forecast(steps=90)

# Plot historical and forecasted prices
plt.figure(figsize=(10, 6))
plt.plot(time_series_data.index, time_series_data['Freight_Price'], label='Historical Prices')
plt.plot(future_dates, forecast, label='Forecasted Prices', linestyle='--', color='orange')
plt.xlabel('Date')
plt.ylabel('Freight Price')
plt.title('Freight Price Forecast')
plt.legend()
plt.show()

# Example: Predict freight price for a specific scenario (linear regression)
new_scenario = pd.DataFrame({
    'Vessel_Size': [40000],
    'Route_A-B': [1],
    'Container_Type_40ft': [1]
})
predicted_price_lr = model.predict(new_scenario)[0]

# Example: Predict freight price for a specific scenario (time-series)
predicted_price_ts = model_ts_fit.forecast(steps=1).iloc[0]

print(f"Predicted freight price (Linear Regression): ${predicted_price_lr:.2f}")
print(f"Predicted freight price (Time-Series): ${predicted_price_ts:.2f}")
