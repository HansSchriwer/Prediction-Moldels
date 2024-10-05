#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Python-based AI agent to predict AMC stock prices 120 days into the future #


# In[ ]:


# step 1: set up and import libraries #


# In[ ]:


get_ipython().system(' pip install pandas numpy scikit-learn matplotlib yfinance tensorflow')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


# step 2: data collection #


# In[ ]:


# Define the ticker symbol and the period
ticker = 'AMC'
start_date = '2010-01-01'
end_date = '2024-10-05'

# Fetch the data
data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows
print(data.head())


# In[ ]:


# step 3: data preprocessing #


# In[ ]:


# handling missing values #
# Check for missing values
print(data.isnull().sum())

# If missing values exist, you can fill them using forward fill
data.fillna(method='ffill', inplace=True)


# In[ ]:


# feature seelction #
# Use the 'Close' price
close_prices = data[['Close']]


# In[ ]:


# normalization #


# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)


# In[ ]:


# creating training and testing sets #
# Define the training data length
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

# Split the data
train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len - 120:]  # Include 120 days for sequences


# In[ ]:


# creating the sequences #
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 120

X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)


# In[ ]:


# reshaping the data #
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[ ]:


# step 4: building the LSTM model #
# Initialize the model
model = Sequential()

# Add LSTM layers and Dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[ ]:


# step 5: training the model #
# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stop]
)


# In[ ]:


# plotting, training and validation loss #
plt.figure(figsize=(14,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# step 6: making predictions #
# Get the model's predicted price values
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Get the actual prices
actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[ ]:


# step 7: evaluating the model #
from sklearn.metrics import mean_squared_error

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual, predictions))
print(f'RMSE: {rmse}')

# Plot the results
plt.figure(figsize=(14,5))
plt.plot(actual, label='Actual AMC Price')
plt.plot(predictions, label='Predicted AMC Price')
plt.title('AMC Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AMC Stock Price')
plt.legend()
plt.show()


# In[ ]:


# step 8: forecasting future prices #
# Get the last 120 days from the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Assuming the previous steps have been executed and 'model' is trained

# Get the last 120 days from the scaled data
last_120_days = scaled_data[-sequence_length:]
last_120_days = last_120_days.reshape((1, sequence_length, 1))

# Initialize the list to store future predictions
future_predictions = []

# Set the current input to the last 120 days
current_input = last_120_days

# Predict the next 120 days
for i in range(120):
    # Make a prediction
    pred = model.predict(current_input)
    
    # Append the predicted value to the future_predictions list
    future_predictions.append(pred[0, 0])
    
    # Reshape pred to (1, 1, 1) to match the dimensions for concatenation
    pred_reshaped = pred.reshape((1, 1, 1))
    
    # Update the current_input by appending the new prediction and removing the first value
    current_input = np.concatenate((current_input[:, 1:, :], pred_reshaped), axis=1)

# Inverse transform the predictions to get actual price values
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create a dataframe for future predictions with appropriate dates
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=120, freq='B')  # 'B' for business days
future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Predicted_Close'])

# Display the first few predictions
print(future_df.head())

# Plot the future predictions along with historical data
plt.figure(figsize=(14, 5))
plt.plot(data['Close'], label='Historical AMC Price')
plt.plot(future_df['Predicted_Close'], label='Predicted AMC Price', color='red')
plt.title('AMC Stock Price Forecast for Next 120 Days')
plt.xlabel('Date')
plt.ylabel('AMC Stock Price')
plt.legend()
plt.show()


# In[ ]:


# step 9: saving the model #
get_ipython().system(' pip install h5py')
model.save('amc_stock_prediction_model.h5')
print("Model saved to disk.")


# In[ ]:


# if you are using anaconda, you can install as follows #
get_ipython().system(' conda install -c anaconda h5p')


# In[ ]:


# verify the instlalation #
import h5py
print(h5py.__version__)


# In[ ]:


# save the model #
# Ensure h5py is installed and imported
import h5py

# Save the model in HDF5 format
model.save('amc_stock_prediction_model.h5')
print("Model saved to 'amc_stock_prediction_model.h5'.


# In[ ]:


# step 10: loading and using the model for predictions #
from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model('amc_stock_prediction_model.h5')

# Use the loaded model to make predictions
# Example: Predicting the next 120 days as done previously


# In[ ]:




