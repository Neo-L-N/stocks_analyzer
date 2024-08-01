import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def evaluate_models(input_file, svr_model_file, rf_model_file, results_file):
    # Load data
    print("Loading data...")
    data = pd.read_csv(input_file)
    data['date'] = pd.to_datetime(data['date'])
    print("Data loaded successfully")

    # Select features and target
    features = ['open', 'high', 'low', 'volume', 'return', 'rolling_mean', 'rolling_std']
    target = 'close'
    stocks = data['Name'].unique()
    
    results = []

    # Load SVR model
    print("Loading SVR model...")
    svr_model = joblib.load(svr_model_file)
    print("SVR model loaded successfully")

    # Load RF model
    print("Loading RF model...")
    rf_model = joblib.load(rf_model_file)
    print("RF model loaded successfully")

    # Create directories for SVR and RF plots
    svr_plots_dir = 'results/svr_plots'
    rf_plots_dir = 'results/rf_plots'
    os.makedirs(svr_plots_dir, exist_ok=True)
    os.makedirs(rf_plots_dir, exist_ok=True)

    for stock in stocks:
        stock_data = data[data['Name'] == stock]
        X = stock_data[features]
        y = stock_data[target]
        dates = stock_data['date']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state=42)

        # Sort test data by date
        X_test_sorted = X_test.sort_index()
        y_test_sorted = y_test.sort_index()
        dates_test_sorted = dates_test.sort_index()

        # Predict and evaluate using SVR
        svr_predictions = svr_model.predict(X_test_sorted)

        mse = mean_squared_error(y_test_sorted, svr_predictions)
        mae = mean_absolute_error(y_test_sorted, svr_predictions)
        r2 = r2_score(y_test_sorted, svr_predictions)
        results.append({
            'Stock': stock,
            'Model': 'SVR',
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        })

        # Plotting the results for SVR
        plt.figure(figsize=(10, 6))
        plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values')
        plt.plot(dates_test_sorted, svr_predictions, label='SVR Predictions')
        plt.title(f'{stock} - SVR')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(os.path.join(svr_plots_dir, f'{stock}_SVR.png'))
        plt.close()

        # Predict and evaluate using RF
        rf_predictions = rf_model.predict(X_test_sorted)

        mse = mean_squared_error(y_test_sorted, rf_predictions)
        mae = mean_absolute_error(y_test_sorted, rf_predictions)
        r2 = r2_score(y_test_sorted, rf_predictions)
        results.append({
            'Stock': stock,
            'Model': 'RF',
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        })

        # Plotting the results for RF
        plt.figure(figsize=(10, 6))
        plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values')
        plt.plot(dates_test_sorted, rf_predictions, label='RF Predictions')
        plt.title(f'{stock} - RF')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(os.path.join(rf_plots_dir, f'{stock}_RF.png'))
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    input_file = 'data/processed/features_data.csv'
    svr_model_file = 'models/svr_model.joblib'
    rf_model_file = 'models/rf_model.joblib'
    results_file = 'results/evaluation_results.csv'
    evaluate_models(input_file, svr_model_file, rf_model_file, results_file)

