# Model 2: 
To create a more advanced code that includes team form, player statistics, head-to-head records, and other relevant metrics, we'll need to expand our data collection and feature engineering processes. 

# Step 1: Install Necessary Libraries

pip install requests pandas scikit-learn xgboost

# Step 2: Enhanced Code

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
        'AwayPlayerStats': [0.70, 0.65, 0.60]
    }
    return pd.DataFrame(data)

# Fetch and prepare data
df = fetch_sports_data()

# Feature engineering
df['HomeWin'] = (df['HomeGoals'] > df['AwayGoals']).astype(int)
df['AwayWin'] = (df['AwayGoals'] > df['HomeGoals']).astype(int)
df['Draw'] = (df['HomeGoals'] == df['AwayGoals']).astype(int)

# Example features and target
features = ['HomeOdds', 'AwayOdds', 'DrawOdds', 'HomeForm', 'AwayForm', 'HeadToHeadHomeWins', 'HeadToHeadAwayWins', 'HomePlayerStats', 'AwayPlayerStats']
X = df[features]
y = df[['HomeWin', 'AwayWin', 'Draw']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Evaluate the models
for target, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Model Evaluation for {target}:")
    print(classification_report(y_test[target], y_pred))
    print(f"Accuracy: {accuracy_score(y_test[target], y_pred)}\n")

# Function to predict and evaluate betting options for new matches
def evaluate_betting_options(new_matches):
    new_df = pd.DataFrame(new_matches)
    new_X = new_df[features]
    
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(new_X)
    
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
    'AwayPlayerStats': [0.64, 0.70]
}

# Evaluate betting options
evaluated_bets = evaluate_betting_options(new_matches)
print(evaluated_bets)

# Explanation
1.	Data Collection:
o	The fetch_sports_data function simulates fetching sports data, including team form, player statistics, and head-to-head records. Replace this with actual API calls for real data.
2.	Feature Engineering:
o	Additional features are included: HomeForm, AwayForm, HeadToHeadHomeWins, HeadToHeadAwayWins, HomePlayerStats, and AwayPlayerStats.
o	These features are used to predict the outcomes.
3.	Model Training:
o	Hyperparameter tuning is implemented using GridSearchCV to find the best parameters for the XGBoost model.
o	The model is trained separately for each outcome (home win, away win, draw).
4.	Model Evaluation:
o	The models are evaluated using classification metrics.
5.	Predicting New Matches:
o	evaluate_betting_options function predicts outcomes for new matches and evaluates betting options.
# Comments
•	Advanced Features: Continuously add more relevant features for improved predictions.
•	Hyperparameter Tuning: Fine-tuning hyperparameters can significantly improve model performance.
•	Data Sources: Ensure the data is comprehensive and up-to-date.
•	Ethical Considerations: Always gamble responsibly and understand the risks involved.
This enhanced version provides a robust framework for evaluating betting options using advanced analytics and machine learning models. Further refinement and incorporation of more complex features can improve prediction accuracy.
