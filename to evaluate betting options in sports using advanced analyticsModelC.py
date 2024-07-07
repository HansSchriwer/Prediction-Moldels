Model C: Let's refine the code by incorporating more complex features and improving prediction accuracy. 

# Advanced Features:
Recent performance metrics (e.g., average goals scored/conceded in the last 5 matches).
Team ranking or ELO rating.
Weather conditions during the match.
Injuries and suspensions.
Home/Away win rates.
Betting odds movement.
Feature Scaling:

# Normalize or standardize features to improve model performance.
Ensemble Models:

# Combine predictions from multiple models to improve accuracy.
Here's an updated version of the code incorporating these refinements:

# Step 1: Install Necessary Libraries

pip install requests pandas scikit-learn xgboost

# Step 2: Enhanced Code

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Example function to fetch sports data (replace with actual API calls)
def fetch_sports_data():
    # Placeholder for actual data fetching logic
    data = {
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'HomeTeam': ['TeamA', 'TeamB', 'TeamC'],
        'AwayTeam': ['TeamD', 'TeamE', 'TeamF'],
        'HomeGoals': [2, 1, 3],
        'AwayGoals': [1, 2, 0],
        'HomeOdds': [1.8, 2.0, 1.5],
        'AwayOdds': [2.1, 1.9, 2.5],
        'DrawOdds': [3.2, 3.0, 3.4],
        'HomeForm': [3, 2, 4],
        'AwayForm': [2, 3, 1],
        'HeadToHeadHomeWins': [5, 6, 4],
        'HeadToHeadAwayWins': [2, 3, 2],
        'HomePlayerStats': [0.75, 0.85, 0.80],
        'AwayPlayerStats': [0.70, 0.65, 0.60],
        'HomeRanking': [1, 2, 3],
        'AwayRanking': [4, 5, 6],
        'Weather': [0, 1, 0],  # 0: Good, 1: Bad
        'InjuriesHome': [1, 2, 0],
        'InjuriesAway': [0, 1, 1]
    }
    return pd.DataFrame(data)

# Fetch and prepare data
df = fetch_sports_data()

# Feature engineering
df['HomeWin'] = (df['HomeGoals'] > df['AwayGoals']).astype(int)
df['AwayWin'] = (df['AwayGoals'] > df['HomeGoals']).astype(int)
df['Draw'] = (df['HomeGoals'] == df['AwayGoals']).astype(int)

# Example features and target
features = [
    'HomeOdds', 'AwayOdds', 'DrawOdds', 'HomeForm', 'AwayForm', 'HeadToHeadHomeWins', 
    'HeadToHeadAwayWins', 'HomePlayerStats', 'AwayPlayerStats', 'HomeRanking', 
    'AwayRanking', 'Weather', 'InjuriesHome', 'InjuriesAway'
]
X = df[features]
y = df[['HomeWin', 'AwayWin', 'Draw']]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Train the XGBoost model for each outcome using GridSearchCV
models = {}
for target in ['HomeWin', 'AwayWin', 'Draw']:
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train[target])
    models[target] = grid_search.best_estimator_

# Create an ensemble model
ensemble_models = [(f'xgb_{target}', model) for target, model in models.items()]
ensemble_clf = VotingClassifier(estimators=ensemble_models, voting='soft')

# Train the ensemble model
ensemble_clf.fit(X_train, y_train)

# Evaluate the models
for target, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Model Evaluation for {target}:")
    print(classification_report(y_test[target], y_pred))
    print(f"Accuracy: {accuracy_score(y_test[target], y_pred)}\n")

# Evaluate the ensemble model
y_pred_ensemble = ensemble_clf.predict(X_test)
for i, target in enumerate(['HomeWin', 'AwayWin', 'Draw']):
    print(f"Ensemble Model Evaluation for {target}:")
    print(classification_report(y_test[target], y_pred_ensemble[:, i]))
    print(f"Accuracy: {accuracy_score(y_test[target], y_pred_ensemble[:, i])}\n")

# Function to predict and evaluate betting options for new matches
def evaluate_betting_options(new_matches):
    new_df = pd.DataFrame(new_matches)
    new_X = new_df[features]
    new_X_scaled = scaler.transform(new_X)
    
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(new_X_scaled)
    
    new_df['Predicted_HomeWin'] = predictions['HomeWin']
    new_df['Predicted_AwayWin'] = predictions['AwayWin']
    new_df['Predicted_Draw'] = predictions['Draw']
    
    return new_df

# Example new matches to evaluate
new_matches = {
    'Date': ['2023-01-04', '2023-01-05'],
    'HomeTeam': ['TeamG', 'TeamH'],
    'AwayTeam': ['TeamI', 'TeamJ'],
    'HomeOdds': [1.7, 2.2],
    'AwayOdds': [2.4, 1.8],
    'DrawOdds': [3.1, 3.3],
    'HomeForm': [4, 3],
    'AwayForm': [2, 1],
    'HeadToHeadHomeWins': [5, 7],
    'HeadToHeadAwayWins': [2, 3],
    'HomePlayerStats': [0.78, 0.81],
    'AwayPlayerStats': [0.64, 0.70],
    'HomeRanking': [1, 2],
    'AwayRanking': [3, 4],
    'Weather': [0, 1],
    'InjuriesHome': [1, 0],
    'InjuriesAway': [0, 2]
}

# Evaluate betting options
evaluated_bets = evaluate_betting_options(new_matches)
print(evaluated_bets)
Explanation
Advanced Features:

# Added features: HomeRanking, AwayRanking, Weather, InjuriesHome, InjuriesAway.
Feature Scaling:

# Applied standard scaling to features for better model performance.
Ensemble Models:

# Combined predictions from individual XGBoost models using VotingClassifier for an ensemble approach.
Hyperparameter Tuning:

# Used GridSearchCV to find the best hyperparameters for XGBoost models.
Prediction and Evaluation:

# Evaluated both individual models and the ensemble model.
Provided a function to predict and evaluate new matches.
# Comments
Feature Engineering: Continuously refine and add more features based on domain knowledge.
Model Tuning: Explore other machine learning algorithms and ensemble methods for improved accuracy.
Data Quality: Ensure high-quality, up-to-date data for accurate predictions.
