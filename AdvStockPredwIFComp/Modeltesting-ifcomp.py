# Model-Testing: To evaluate the prediction accuracy of the model, we need to measure its performance on a validation or test dataset. 
Common metrics used for regression tasks include Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Below is the code to calculate these metrics for the combined prediction model:

### Evaluate Prediction Accuracy

1. **Split the data into training and testing sets.**
2. **Train the models and generate predictions.**
3. **Evaluate the predictions using performance metrics.**

# Here’s how you can calculate these metrics:

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assuming the code to train the models and generate predictions (arima_pred, rf_pred, gb_pred, lstm_pred) is already done

# Generate predictions for the entire test set to evaluate accuracy
arima_pred_full = arima_model_fit.forecast(steps=len(y_test))
rf_pred_full = rf_model.predict(X_test)
gb_pred_full = gb_model.predict(X_test)
lstm_pred_full = lstm_model.predict(np.expand_dims(X_test, axis=-1))

# Combine predictions using the same weights
combined_pred_full = (weights[0] * arima_pred_full + weights[1] * rf_pred_full + weights[2] * gb_pred_full + weights[3] * lstm_pred_full.squeeze())

# Rescale predictions back to original scale
combined_pred_full_rescaled = scaler.inverse_transform(combined_pred_full.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate performance metrics
mae = mean_absolute_error(y_test_rescaled, combined_pred_full_rescaled)
mse = mean_squared_error(y_test_rescaled, combined_pred_full_rescaled)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

### Explanation of Metrics
- **MAE (Mean Absolute Error)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.
  
- **MSE (Mean Squared Error)**: Measures the average of the squares of the errors. It’s the average squared difference between the estimated values and what is estimated. MSE gives a higher weight to large errors, making it useful when large errors are particularly undesirable.
  
- **RMSE (Root Mean Squared Error)**: The square root of the MSE. It is the standard deviation of the prediction errors (residuals). RMSE serves to aggregate the magnitudes of the errors in predictions into a single measure of predictive power.

### Notes:
- The weights `[0.3, 0.3, 0.2, 0.2]` should be determined through cross-validation or a similar performance evaluation process. Adjust these weights based on the specific performance of each model.
- Ensure that the `scaler.inverse_transform` step correctly matches the scaling transformation applied during preprocessing.
