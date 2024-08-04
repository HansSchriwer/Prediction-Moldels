### Advanced Python Program: Economic Activity Analysis

1. **Data Collection**: Collect data from relevant sources (e.g., Federal Reserve Economic Data - FRED).
2. **Data Processing**: Clean and preprocess the data for analysis.
3. **Statistical Analysis**: Analyze historical data to detect trends and patterns.
4. **Prediction Model**: Use a machine learning model (e.g., Random Forest or LSTM) to predict the next quarter's economic activity.

### Step-by-Step Implementation

#### Step 1: Data Collection

You'll need to collect data from sources like FRED. For this example, let's assume we have the data locally.

#### Step 2: Data Processing

import pandas as pd

# Load data (assumed to be in CSV files)
gdp_data = pd.read_csv('gdp_data.csv')
unemployment_data = pd.read_csv('unemployment_data.csv')
consumer_sentiment_data = pd.read_csv('consumer_sentiment_data.csv')

# Merge data into a single DataFrame
economic_data = pd.merge(gdp_data, unemployment_data, on='date')
economic_data = pd.merge(economic_data, consumer_sentiment_data, on='date')

# Convert date column to datetime type
economic_data['date'] = pd.to_datetime(economic_data['date'])

# Set date as index
economic_data.set_index('date', inplace=True)

print(economic_data.head())

#### Step 3: Statistical Analysis

import matplotlib.pyplot as plt

# Plot historical data
plt.figure(figsize=(14, 7))
plt.plot(economic_data['gdp_growth'], label='GDP Growth')
plt.plot(economic_data['unemployment_rate'], label='Unemployment Rate')
plt.plot(economic_data['consumer_sentiment'], label='Consumer Sentiment Index')
plt.legend()
plt.title('Historical Economic Indicators')
plt.show()

#### Step 4: Prediction Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Feature selection
features = economic_data[['gdp_growth', 'unemployment_rate', 'consumer_sentiment']]
target = economic_data['gdp_growth'].shift(-1)  # Predicting next quarter's GDP growth

# Drop last row due to shift
features = features[:-1]
target = target[:-1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Predict next quarter's GDP growth
next_quarter_prediction = model.predict(features.tail(1))
print(f'Next Quarter GDP Growth Prediction: {next_quarter_prediction[0]}')

### Explanation

1. **Data Collection**: Load GDP growth, unemployment rate, and consumer sentiment index data.
2. **Data Processing**: Merge the data into a single DataFrame and set the date as the index.
3. **Statistical Analysis**: Plot historical data to visualize trends.
4. **Prediction Model**: Use a Random Forest Regressor to predict the next quarter's GDP growth based on the selected economic indicators.

### Customization

- **Additional Indicators**: You can include more economic indicators (e.g., inflation rate, interest rates) for a more comprehensive analysis.
- **Advanced Models**: Consider using more sophisticated models like LSTM for time series forecasting if the data shows significant temporal dependencies.
- **Model Tuning**: Optimize model parameters using techniques like GridSearchCV to improve prediction accuracy.
