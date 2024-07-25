import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_results(true_values_file, predictions_files, model_names, output_file):
    y_true = pd.read_csv(true_values_file)['close'].values
    
    plt.figure(figsize=(14, 8))
    
    for pred_file, model_name in zip(predictions_files, model_names):
        y_pred = np.load(pred_file)
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label=model_name)
    
    plt.title('Model Predictions vs Actual Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    true_values_file = 'data/processed/features_data.csv'
    predictions_files = [
        'results/svr_predictions.npy',
        'results/rf_predictions.npy',
        'results/lstm_predictions.npy',
        'results/xgboost_predictions.npy',
        'results/lightgbm_predictions.npy',
        'results/catboost_predictions.npy'
    ]
    model_names = ['SVR', 'Random Forest', 'LSTM', 'XGBoost', 'LightGBM', 'CatBoost']
    output_file = 'results/model_predictions_vs_actual.png'
    
    plot_results(true_values_file, predictions_files, model_names, output_file)
