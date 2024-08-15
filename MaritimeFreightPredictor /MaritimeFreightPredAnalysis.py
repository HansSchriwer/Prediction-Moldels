import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Simulate the expanded historical pricing data and market indicators
extended_historical_pricing_data = pd.DataFrame({
    'route': ['Port A-Port D', 'Port B-Port E', 'Port C-Port F'] * 12,
    'vessel_type': ['Type 1', 'Type 2', 'Type 3'] * 12,
    'container_size': [20, 40] * 18,  # 20 ft or 40 ft containers
    'date': pd.date_range(start='2022-01-01', periods=36, freq='M'),
    'price_per_container': np.random.randint(1500, 3500, size=36)  # Simulated prices
})

market_indicators_data = pd.DataFrame({
    'date': pd.date_range(start='2022-01-01', periods=36, freq='M'),
    'global_trade_index': np.random.uniform(0.8, 1.2, size=36),  # Simulated index values
    'economic_conditions': np.random.uniform(0.7, 1.3, size=36)   # Simulated economic conditions index
})

# Merge the data
merged_data = pd.merge(extended_historical_pricing_data, market_indicators_data, on='date')

# Normalize the external factors
merged_data['global_trade_index'] = (merged_data['global_trade_index'] - merged_data['global_trade_index'].min()) / \
                                    (merged_data['global_trade_index'].max() - merged_data['global_trade_index'].min())
merged_data['economic_conditions'] = (merged_data['economic_conditions'] - merged_data['economic_conditions'].min()) / \
                                     (merged_data['economic_conditions'].max() - merged_data['economic_conditions'].min())

# Feature engineering
merged_data['month'] = merged_data['date'].dt.month
merged_data['price_lag_1'] = merged_data.groupby(['route', 'vessel_type', 'container_size'])['price_per_container'].shift(1)
merged_data['price_lag_2'] = merged_data.groupby(['route', 'vessel_type', 'container_size'])['price_per_container'].shift(2)

# Drop rows with missing values
merged_data.dropna(inplace=True)

# Define features and target variable
features = ['month', 'global_trade_index', 'economic_conditions', 'price_lag_1', 'price_lag_2']
target = 'price_per_container'

X = merged_data[features]
y = merged_data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Manual Tuning of the Random Forest Model with reasonable parameters
rf_model_manual = RandomForestRegressor(
    n_estimators=200,  # Number of trees
    max_depth=20,      # Maximum depth of the tree
    min_samples_split=5,  # Minimum number of samples required to split an internal node
    min_samples_leaf=2,   # Minimum number of samples required to be at a leaf node
    bootstrap=True,       # Whether bootstrap samples are used when building trees
    random_state=42
)

# Fit the model on the training data
rf_model_manual.fit(X_train, y_train)

# Enhanced scenario analysis tool with route, vessel, and container selection
def scenario_analysis_enhanced(route, vessel_type, container_size, fuel_price_change=0, demand_change=0, economic_change=0):
    # Filter data based on user-selected route, vessel type, and container size
    filtered_data = merged_data[
        (merged_data['route'] == route) &
        (merged_data['vessel_type'] == vessel_type) &
        (merged_data['container_size'] == container_size)
    ]
    
    # Adjust the global trade index and economic conditions based on input changes
    adjusted_data = filtered_data.copy()
    adjusted_data['global_trade_index'] += demand_change
    adjusted_data['economic_conditions'] += economic_change
    
    # Predict prices with the adjusted data
    X_adjusted = adjusted_data[features]
    adjusted_predictions = rf_model_manual.predict(X_adjusted)
    
    # Calculate the average price before and after the scenario change
    original_avg_price = filtered_data[target].mean()
    adjusted_avg_price = adjusted_predictions.mean()
    
    # Impact of the scenario
    impact = adjusted_avg_price - original_avg_price
    
    # Visualization: Plot original vs. adjusted prices
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data.index, filtered_data[target], label='Original Prices', color='blue')
    plt.plot(filtered_data.index, adjusted_predictions, label='Adjusted Prices', color='red', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Price per Container (USD)')
    plt.title(f'Scenario Analysis: {route}, {vessel_type}, {container_size} ft Container')
    plt.legend()
    plt.show()
    
    return {
        "Original Average Price": original_avg_price,
        "Adjusted Average Price": adjusted_avg_price,
        "Price Impact": impact
    }

# Example usage with user-selected inputs
route = 'Port A-Port D'
vessel_type = 'Type 1'
container_size = 20
scenario_result_enhanced = scenario_analysis_enhanced(route, vessel_type, container_size, fuel_price_change=0.1, demand_change=0.05, economic_change=0.02)

print(scenario_result_enhanced)
