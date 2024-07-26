import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb

def evaluate_models(input_file, svr_model_file, rf_model_file, lstm_model_file, xgb_model_file, lgbm_model_file, catboost_model_file, results_file):
    # Load data
    print("Loading data...")
    data = pd.read_csv(input_file)
    print("Data loaded successfully")

    features = ['open', 'high', 'low', 'volume', 'return', 'rolling_mean', 'rolling_std']
    target = 'close'
    X = data[features]
    y = data[target]

    # Sample a smaller subset of the data for quicker testing
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    # Load models
    print("Loading models...")
    svr_model = joblib.load(svr_model_file)
    rf_model = joblib.load(rf_model_file)
    lstm_model = keras.models.load_model(lstm_model_file, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    xgb_model = joblib.load(xgb_model_file)
    lgbm_model = joblib.load(lgbm_model_file)
    catboost_model = joblib.load(catboost_model_file)
    print("Models loaded successfully")

    # Evaluate models
    results = {}

    print("Evaluating models...")
    models = {
        'SVR': svr_model,
        'RandomForest': rf_model,
        'LSTM': lstm_model,
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model,
        'CatBoost': catboost_model
    }

    for model_name, model in models.items():
        if model_name == 'LSTM':
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
        
        # Save the predictions as numpy files
        np.save(f'results/{model_name.lower()}_predictions.npy', y_pred)
        
        mse = mean_squared_error(y_test, y_pred)
        results[model_name] = mse
        print(f"{model_name} MSE: {mse}")

    # Save true values for plotting
    np.save('results/true_values.npy', y_test)

    # Save results
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'MSE'])
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    input_file = 'data/processed/features_data.csv'
    svr_model_file = 'models/svr_model.joblib'
    rf_model_file = 'models/rf_model.joblib'
    lstm_model_file = 'models/lstm_model.h5'
    xgb_model_file = 'models/xgboost_model.joblib'
    lgbm_model_file = 'models/lightgbm_model.joblib'
    catboost_model_file = 'models/catboost_model.joblib'
    results_file = 'results/model_evaluation_results.csv'
    evaluate_models(input_file, svr_model_file, rf_model_file, lstm_model_file, xgb_model_file, lgbm_model_file, catboost_model_file, results_file)
