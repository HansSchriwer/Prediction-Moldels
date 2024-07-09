# Model 3: For predicting forex rates with high accuracy, using an ensemble approach combining multiple models can often yield the best results. 
This method leverages the strengths of different models to produce a more robust and accurate prediction. 
Below is an example of how you might combine LSTM, Temporal Fusion Transformer (TFT), and a basic machine learning model like Random Forest to predict forex rates.

### Requirements
- Python 3.x
- TensorFlow
- Keras
- PyTorch
- PyTorch Forecasting
- Pandas
- Scikit-learn
- Matplotlib

# You can install the necessary packages using pip:

pip install tensorflow pandas numpy scikit-learn matplotlib torch pytorch-forecasting

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, metrics
from pytorch_lightning import Trainer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

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

# Ensemble predictions
ensemble_predictions = (lstm_predictions.flatten() + tft_predictions.flatten() + rf_predictions.flatten()) / 3

# Plot the predictions
plt.figure(figsize=(14, 5))
plt.plot(data.index[-len(ensemble_predictions):], scaler.inverse_transform(y_lstm_test.reshape(-1, 1)), label='Actual Forex Price')
plt.plot(data.index[-len(ensemble_predictions):], ensemble_predictions, label='Ensemble Predicted Forex Price')
plt.title('Forex Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate RMSE
rmse = mean_squared_error(scaler.inverse_transform(y_lstm_test.reshape(-1, 1)), ensemble_predictions, squared=False)
print(f'Root Mean Squared Error: {rmse}')

### Explanation
1. **Data Loading and Preprocessing**: The code loads the historical forex data, performs feature engineering, and scales the closing prices.
2. **LSTM Model**:
   - Prepares the data for the LSTM model.
   - Defines, trains, and makes predictions using the LSTM model.
3. **Temporal Fusion Transformer (TFT) Model**:
   - Prepares the data for the TFT model.
   - Defines, trains, and makes predictions using the TFT model.
4. **RandomForest Model**:
   - Prepares the data for the RandomForest model.
   - Trains and makes predictions using the RandomForest model.
5. **Ensemble Predictions**: Combines the predictions from all three models (LSTM, TFT, and RandomForest) by averaging them to get the final prediction.
6. **Visualization and Evaluation**: Plots the actual vs. predicted forex prices and calculates the RMSE to evaluate the performance.
