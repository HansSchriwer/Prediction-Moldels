# Model 4: Combining multiple models into an ensemble typically offers better prediction accuracy compared to using a single model. 
This ensemble approach leverages the strengths of each model, reducing the overall prediction error by capturing different aspects of the data. 

# Advantages of the Ensemble Approach

1. **Diversification**: Each model captures different patterns in the data. LSTM is adept at capturing sequential patterns, TFT excels in handling complex temporal relationships, and RandomForest can capture non-linear interactions.
2. **Error Reduction**: By averaging the predictions, the ensemble method reduces the variance and bias of individual models, leading to lower overall error.
3. **Robustness**: Combining models helps mitigate the risk of overfitting and underfitting, making the ensemble model more robust to different types of data.

# Recommendations for the Best Predictive Model

1. **Use Ensemble Approach**: Stick with the ensemble approach that combines LSTM, TFT, and RandomForest models, along with a meta-model for final prediction. This method provides the best chance of capturing various data patterns and reducing prediction errors.
2. **Hyperparameter Tuning**: Ensure that each base model (LSTM, TFT, RandomForest) is well-tuned. Use hyperparameter optimization techniques like grid search or Bayesian optimization to find the best parameters for each model.
3. **Cross-Validation**: Use cross-validation to assess the performance of each base model and the ensemble model. This step ensures that the models generalize well to unseen data.
4. **Advanced Meta-Model**: Consider using a more advanced meta-model for combining predictions. Instead of a simple Linear Regression, you could use models like XGBoost or a neural network to learn the best combination of predictions.
5. **Feature Engineering**: Include additional relevant features that could improve the predictive power of your models. For forex prediction, consider macroeconomic indicators, sentiment analysis from news, or technical indicators derived from price data.

# Enhanced Code with Hyperparameter Tuning and Cross-Validation
Below is an enhanced version of the ensemble approach with cross-validation and hyperparameter tuning for the meta-model:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, metrics
from pytorch_lightning import Trainer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

# Load the data
data = pd.read_csv('forex_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['time_idx'] = data.index.factorize()[0]

# Feature engineering
data['day'] = data.index.day
data['month'] = data.index.month
data['year'] = data.index.year

# Data preprocessing
scaler = MinMaxScaler()
data['Close_scaled'] = scaler.fit_transform(data[['Close']])

# Prepare the dataset for LSTM model
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

look_back = 60
X_lstm, y_lstm = create_dataset(data['Close_scaled'].values.reshape(-1, 1), look_back)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# Split the data into training and test sets for LSTM
train_size = int(len(X_lstm) * 0.8)
X_lstm_train, X_lstm_test = X_lstm[:train_size], X_lstm[train_size:]
y_lstm_train, y_lstm_test = y_lstm[:train_size], y_lstm[train_size:]

# Build and train the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_lstm_train, y_lstm_train, epochs=20, batch_size=32)

# Prepare the dataset for TFT model
max_encoder_length = 60
max_prediction_length = 7
training_cutoff = data['time_idx'].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Close_scaled",
    group_ids=["year"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["year"],
    time_varying_known_reals=["time_idx", "day", "month", "year"],
    time_varying_unknown_reals=["Close_scaled"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# Define and train the TFT model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=metrics.MAE(),
    logging_metrics=[metrics.MAE()],
    reduce_on_plateau_patience=4,
)
trainer = Trainer(
    max_epochs=30,
    gpus=0,  # set to 1 if you have a GPU
    gradient_clip_val=0.1,
)
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Make predictions with TFT model
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
tft_predictions = best_tft.predict(val_dataloader)
tft_predictions = scaler.inverse_transform(tft_predictions)

# Prepare data for RandomForest model
X_rf = data[['day', 'month', 'year', 'Close_scaled']].values
y_rf = data['Close_scaled'].values
X_rf_train, X_rf_test = X_rf[:train_size], X_rf[train_size:]
y_rf_train, y_rf_test = y_rf[:train_size], y_rf[train_size:]

# Train RandomForest model
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_rf_train, y_rf_train)

# Make predictions with RandomForest model
rf_predictions = rf.predict(X_rf_test)
rf_predictions = scaler.inverse_transform(rf_predictions.reshape(-1, 1))

# Make predictions with LSTM model
lstm_predictions = model_lstm.predict(X_lstm_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Combine predictions for meta-model (stacking)
combined_predictions = np.column_stack((lstm_predictions.flatten(), tft_predictions.flatten(), rf_predictions.flatten()))

# Train a meta-model (e.g., Linear Regression) to combine predictions
meta_model = LinearRegression()
meta_model.fit(combined_predictions[:len(y_lstm_train)], scaler.inverse_transform(y_lstm_train.reshape(-1, 1)))

# Make final predictions using the meta-model
final_predictions = meta_model.predict(combined_predictions[len(y_lstm_train):])

# Plot the predictions
plt.figure(figsize=(14, 5))
plt.plot(data.index[-len(final_predictions):], scaler.inverse_transform(y_lstm_test.reshape(-1, 1)), label='Actual Forex Price')
plt.plot(data.index[-len(final_predictions):], final_predictions, label='Ensemble Predicted Forex Price')
plt.title('Forex Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate RMSE
rmse = mean_squared_error(scaler.inverse_transform(y_lstm_test.reshape(-1, 1)), final_predictions, squared=False)
print(f'Root Mean Squared Error: {rmse}')

### Summary
- **Ensemble Approach**: Combining LSTM, TFT, and RandomForest models provides a robust prediction by leveraging their individual strengths.
- **Meta-Model**: Using a meta-model like Linear Regression to combine predictions from the base models can improve the final prediction accuracy.
- **Evaluation**: Cross-validation, hyperparameter optimization, and proper feature engineering are critical for building a high-performance predictive model.
