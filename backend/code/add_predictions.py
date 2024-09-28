import pandas as pd
import joblib

# Load processed data
print("Loading data...")
data = pd.read_csv('../data/processed/processed_data.csv')

# Select the original features that the models were trained on
features = ['open', 'high', 'low', 'volume', 'return', 'rolling_mean', 'rolling_std']
target = 'close'
X = data[features]

# Load the trained models
print("Loading models...")
svr_model = joblib.load('../models/svr_model.joblib')
rf_model = joblib.load('../models/rf_model.joblib')

# Generate predictions for SVR
print("Generating SVR predictions...")
svr_predictions = svr_model.predict(X)

# Generate predictions for Random Forest
print("Generating RF predictions...")
rf_predictions = rf_model.predict(X)

# Add the predictions to the dataframe
data['svr_predicted'] = svr_predictions
data['rf_predicted'] = rf_predictions

# Save the dataframe with the predictions
output_file = '../data/processed/processed_data_with_predictions.csv'
data.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")


