# Model 2: More advanced models that can potentially provide better predictions for forex trading. 
One such model is the Temporal Fusion Transformer (TFT), which can capture temporal patterns and relationships in time series data more effectively than traditional LSTM models.

# Below is an example of how you can implement a Temporal Fusion Transformer for forex trading prediction using PyTorch and the `pytorch-forecasting` library.

### Requirements
- Python 3.x
- PyTorch
- PyTorch Forecasting
- Pandas
- Scikit-learn

# You can install the necessary packages using pip:

pip install torch pytorch-forecasting pandas scikit-learn

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, metrics
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import Trainer

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

# Prepare the dataset for the TFT model
max_encoder_length = 60
max_prediction_length = 30

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

# Create dataloaders
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# Define the TFT model
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

# Train the model
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

# Make predictions
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)

# Inverse transform predictions
actuals = scaler.inverse_transform(actuals)
predictions = scaler.inverse_transform(predictions)

# Plot the predictions
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(data.index[-len(actuals):], actuals, label='Actual Forex Price')
plt.plot(data.index[-len(predictions):], predictions, label='Predicted Forex Price')
plt.title('Forex Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate RMSE
rmse = mean_squared_error(actuals, predictions, squared=False)
print(f'Root Mean Squared Error: {rmse}')

### Explanation
1. **Data Loading and Preprocessing**: The code loads the historical forex data and scales the closing prices.
2. **Feature Engineering**: Additional time-related features (day, month, year) are added.
3. **Dataset Preparation**: The data is prepared for the Temporal Fusion Transformer using the `TimeSeriesDataSet` class from `pytorch-forecasting`.
4. **Model Definition**: A Temporal Fusion Transformer model is defined with the specified parameters.
5. **Model Training**: The model is trained using PyTorch Lightning's `Trainer`.
6. **Prediction and Visualization**: The model's predictions are plotted against the actual prices, and RMSE is calculated to evaluate performance.

