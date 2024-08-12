# Enhancing the earthquake prediction model involves incorporating more sophisticated modeling techniques, 
# such as Temporal Fusion Transformers (TFT) for better handling of temporal data, advanced feature engineering methods like Principal Component Analysis (PCA) 
# or autoencoders for dimensionality reduction, and hybrid models that combine the strengths of different approaches. Below is an enhanced version of the model:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shap
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# Load and preprocess data
earthquake_data = pd.read_csv('earthquake_data.csv')
weather_data = pd.read_csv('weather_data.csv')
fracking_data = pd.read_csv('fracking_data.csv')

# Merge datasets
data = pd.merge(earthquake_data, weather_data, on='date')
data = pd.merge(data, fracking_data, on='location')

# Feature engineering
data['proximity_to_fault'] = calculate_proximity(data['location'], fault_lines)
data['fracking_intensity'] = calculate_fracking_intensity(data['fracking_depth'], data['number_of_wells'])
data['time_idx'] = (data['date'] - data['date'].min()).dt.days

# Prepare training and testing data
X = data[['magnitude', 'depth', 'temperature_anomaly', 'fracking_intensity', 'proximity_to_fault']]
y = data['earthquake_occurred']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Random Forest model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

# 2. LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(np.expand_dims(X_train, axis=2), y_train, batch_size=1, epochs=1)

# 3. XGBoost model
xg_model = xgb.XGBClassifier()
xg_model.fit(X_train, y_train)
xg_preds = xg_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, xg_preds))

# 4. Autoencoder for anomaly detection
input_dim = X_train.shape[1]
encoding_dim = 4

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
anomaly_scores = pd.DataFrame({'mse': mse, 'threshold': y_test})

threshold = np.percentile(mse, 95)
anomalies = anomaly_scores[anomaly_scores.mse > threshold]
print(f"Number of anomalies detected: {len(anomalies)}")

# 5. Temporal Fusion Transformer (TFT)
max_prediction_length = 1
max_encoder_length = 30
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="earthquake_occurred",
    group_ids=["location"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["location"],
    time_varying_known_reals=["time_idx", "temperature_anomaly", "fracking_intensity", "proximity_to_fault"],
    time_varying_unknown_reals=["earthquake_occurred"],
    target_normalizer=GroupNormalizer(groups=["location"]),
)

validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
train_dataloader = training.to_dataloader(train=True, batch_size=64)
val_dataloader = validation.to_dataloader(train=False, batch_size=64)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # Quantile output
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

trainer = torch.optim.Adam(tft.parameters(), lr=0.03)
tft.fit(train_dataloader, val_dataloader, max_epochs=10, gpus=[0])

predictions = tft.predict(val_dataloader, mode="prediction")

# 6. PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 7. Ensemble modeling
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgboost', xgb.XGBClassifier()),
        ('logreg', LogisticRegression())
    ],
    voting='soft'  # Use 'hard' for majority voting, 'soft' for weighted probabilities
)

pipeline_pca = Pipeline([
    ('pca', PCA(n_components=0.95)),
    ('ensemble', ensemble)
])

pipeline_pca.fit(X_train_pca, y_train)
ensemble_pca_preds = pipeline_pca.predict(X_test_pca)
print("PCA + Ensemble Accuracy:", accuracy_score(y_test, ensemble_pca_preds))

# 8. SHAP for interpretability
explainer = shap.TreeExplainer(ensemble)
shap_values = explainer.shap_values(X_test_pca)

shap.summary_plot(shap_values, X_test_pca, feature_names=X.columns)

# 9. Continuous Integration with Real-Time Data
# (Note: This step requires API integration and data pipelines to update the model with real-time data)

### Summary of the Steps:
# Data Preparation and Feature Engineering: Merging and cleaning data, creating features such as proximity to fault lines and fracking intensity.
Random Forest, LSTM, and XGBoost Models: Training and evaluating different models.
Autoencoder for Anomaly Detection: Detecting anomalies that might precede earthquakes.
Temporal Fusion Transformer (TFT): A state-of-the-art model for time-series forecasting, particularly useful for capturing temporal dynamics.
PCA for Dimensionality Reduction: Reducing the complexity of the dataset while retaining most of the information.
Ensemble Modeling: Combining multiple models into a more robust prediction system.
Interpretability with SHAP: Understanding the impact of each feature on the model’s predictions.
Continuous Integration: (Placeholder for real-time data integration).
This complete code framework provides a robust foundation for earthquake prediction, combining traditional machine learning techniques with cutting-edge approaches like TFT and autoencoders. ###
