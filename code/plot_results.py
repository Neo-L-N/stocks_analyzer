import numpy as np
import matplotlib.pyplot as plt

def plot_results(true_values_file, predictions_files, model_names, output_file):
    y_true = np.load(true_values_file)
    plt.figure(figsize=(12, 6))

    plt.plot(y_true, label='True Values')

    for pred_file, model_name in zip(predictions_files, model_names):
        y_pred = np.load(pred_file)
        plt.plot(y_pred, label=f'{model_name} Predictions')
    
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Predictions vs True Values')
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    true_values_file = 'results/true_values.npy'
    predictions_files = [
        'results/svr_predictions.npy',
        'results/randomforest_predictions.npy',
        'results/lstm_predictions.npy',
        'results/xgboost_predictions.npy',
        'results/lightgbm_predictions.npy',
        'results/catboost_predictions.npy'
    ]
    model_names = ['SVR', 'Random Forest', 'LSTM', 'XGBoost', 'LightGBM', 'CatBoost']
    output_file = 'results/prediction_vs_true.png'

    plot_results(true_values_file, predictions_files, model_names, output_file)

