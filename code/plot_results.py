import numpy as np
import matplotlib.pyplot as plt

def plot_results(true_values_file, predictions_file, model_name, output_file):
    y_true = np.load(true_values_file)
    y_pred = np.load(predictions_file)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label=f'{model_name} Predictions')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title(f'{model_name} Predictions vs True Values')
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    true_values_file = 'results/true_values.npy'
    models = {
        'SVR': 'results/svr_predictions.npy',
        'RandomForest': 'results/rf_predictions.npy',
        'LSTM': 'results/lstm_predictions.npy',
        'XGBoost': 'results/xgboost_predictions.npy',
        'LightGBM': 'results/lightgbm_predictions.npy',
        'CatBoost': 'results/catboost_predictions.npy'
    }

    for model_name, predictions_file in models.items():
        output_file = f'results/{model_name}_prediction_vs_true.png'
        plot_results(true_values_file, predictions_file, model_name, output_file)
