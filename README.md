# Stock Price Analyzer

# Stock Price Prediction using Various Machine Learning Models

## Project Overview

This project aims to predict stock prices using historical data of S&P 500 stocks. Multiple machine learning models including Support Vector Regression (SVR), Random Forest (RF), Long Short-Term Memory (LSTM), LightGBM, CatBoost, and XGBoost are trained and evaluated based on their performance.

## Directory Structure

- `data/`: Contains raw and processed data.
  - `raw/`: Contains raw data files.
    - `all_stocks_5yr.csv`
  - `processed/`: Contains processed data files.
    - `features_data.csv`
    - `processed_data.csv`
- `code/`: Contains scripts for data preprocessing, feature engineering, model training, and evaluation.
  - `train_svr.py`
  - `train_rf.py`
  - `train_lstm.py`
  - `train_lightgbm.py`
  - `train_catboost.py`
  - `train_xgboost.py`
  - `evaluation.py`
  - `data_preprocessing.py`
  - `ensemble.py`
  - `feature_engineering.py`
- `models/`: Contains trained models in `.joblib` format.
  - `svr_model.joblib`
  - `rf_model.joblib`
  - `lstm_model.h5`
  - `lightgbm_model.joblib`
  - `catboost_model.joblib`
  - `xgboost_model.joblib`
- `notebooks/`: Contains Jupyter notebooks for exploration and development.
  - `data_exploration.ipynb`
  - `evaluation.ipynb`
  - `model_training.ipynb`
- `scripts/`: Contains shell scripts to run the pipeline.
  - `run_ensemble.sh`
  - `run_evaluation.sh`
  - `run_preprocessing.sh`
  - `run_training_lstm.sh`
  - `run_training_rf.sh`
  - `run_training_svr.sh`
- `results/`: Contains predictions and evaluation metrics (optional).
- `README.md`: Project documentation.
- `requirements.txt`: Required Python packages.

## Setup and Usage

### Step 1: Install Required Packages

Make sure you have Python 3.7 or later. Install the required packages using pip:

```sh
pip install -r requirements.txt
```

# Stock Price Prediction using Various Machine Learning Models

## Project Overview

This project aims to predict stock prices using historical data of S&P 500 stocks. Multiple machine learning models including Support Vector Regression (SVR), Random Forest (RF), Long Short-Term Memory (LSTM), LightGBM, CatBoost, and XGBoost are trained and evaluated based on their performance.

## Directory Structure

- `data/`: Contains raw and processed data.
  - `raw/`: Contains raw data files.
    - `all_stocks_5yr.csv`
  - `processed/`: Contains processed data files.
    - `features_data.csv`
    - `processed_data.csv`
- `code/`: Contains scripts for data preprocessing, feature engineering, model training, and evaluation.
  - `train_svr.py`
  - `train_rf.py`
  - `train_lstm.py`
  - `train_lightgbm.py`
  - `train_catboost.py`
  - `train_xgboost.py`
  - `evaluation.py`
  - `data_preprocessing.py`
  - `ensemble.py`
  - `feature_engineering.py`
- `models/`: Contains trained models in `.joblib` format.
  - `svr_model.joblib`
  - `rf_model.joblib`
  - `lstm_model.h5`
  - `lightgbm_model.joblib`
  - `catboost_model.joblib`
  - `xgboost_model.joblib`
- `notebooks/`: Contains Jupyter notebooks for exploration and development.
  - `data_exploration.ipynb`
  - `evaluation.ipynb`
  - `model_training.ipynb`
- `scripts/`: Contains shell scripts to run the pipeline.
  - `run_ensemble.sh`
  - `run_evaluation.sh`
  - `run_preprocessing.sh`
  - `run_training_lstm.sh`
  - `run_training_rf.sh`
  - `run_training_svr.sh`
- `results/`: Contains predictions and evaluation metrics (optional).
- `README.md`: Project documentation.
- `requirements.txt`: Required Python packages.

## Project Overview

This project aims to predict stock prices using historical data of S&P 500 stocks. Multiple machine learning models including Support Vector Regression (SVR), Random Forest (RF), Long Short-Term Memory (LSTM), LightGBM, CatBoost, and XGBoost are trained and evaluated based on their performance.

## Directory Structure

- `data/`: Contains raw and processed data.
  - `raw/`: Contains raw data files.
    - `all_stocks_5yr.csv`
  - `processed/`: Contains processed data files.
    - `features_data.csv`
    - `processed_data.csv`
- `code/`: Contains scripts for data preprocessing, feature engineering, model training, and evaluation.
  - `train_svr.py`
  - `train_rf.py`
  - `train_lstm.py`
  - `train_lightgbm.py`
  - `train_catboost.py`
  - `train_xgboost.py`
  - `evaluation.py`
  - `data_preprocessing.py`
  - `ensemble.py`
  - `feature_engineering.py`
- `models/`: Contains trained models in `.keras` format.
  - `svr_model.keras`
  - `rf_model.keras`
  - `lstm_model.keras`
  - `lightgbm_model.keras`
  - `catboost_model.keras`
  - `xgboost_model.keras`
- `notebooks/`: Contains Jupyter notebooks for exploration and development.
  - `data_exploration.ipynb`
  - `evaluation.ipynb`
  - `model_training.ipynb`
- `scripts/`: Contains shell scripts to run the pipeline.
  - `run_ensemble.sh`
  - `run_evaluation.sh`
  - `run_preprocessing.sh`
  - `run_training_lstm.sh`
  - `run_training_rf.sh`
  - `run_training_svr.sh`
- `results/`: Contains predictions and evaluation metrics (optional).
- `README.md`: Project documentation.
- `requirements.txt`: Required Python packages.

## Model Explanation

### Support Vector Regression (SVR)

Support Vector Regression (SVR) is a type of Support Vector Machine (SVM) that is used for regression tasks. The main idea of SVR is to find a function that approximates the mapping from the input space to the output space by minimizing the prediction error. SVR uses a linear model in a high-dimensional space created by a kernel function, allowing it to handle non-linear relationships in the data. The key parameters of SVR include the kernel type (e.g., linear, polynomial, RBF), the regularization parameter (C), and the epsilon parameter which defines a margin of tolerance where no penalty is given to errors.

### Random Forest (RF)

Random Forest is an ensemble learning method that builds multiple decision trees and merges them together to get a more accurate and stable prediction. Each tree in the forest is trained on a bootstrap sample from the training data, and during the construction of the tree, a random subset of features is considered for splitting at each node. This randomness helps to reduce the variance of the model and prevent overfitting. The final prediction of the Random Forest is obtained by averaging the predictions of all individual trees (for regression) or by majority voting (for classification).

### Long Short-Term Memory (LSTM)

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) that is designed to model temporal sequences and long-range dependencies. Unlike standard RNNs, LSTMs can effectively capture long-term dependencies in the data without suffering from the vanishing gradient problem. An LSTM network consists of a series of LSTM cells, each containing three gates: the input gate, the forget gate, and the output gate. These gates control the flow of information into, out of, and through the cell, allowing the network to maintain a memory of past inputs over long sequences. LSTMs are particularly well-suited for time series forecasting tasks.

### LightGBM

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be highly efficient and scalable, making it suitable for handling large datasets with high-dimensional features. LightGBM introduces several optimizations, such as histogram-based learning, leaf-wise tree growth, and exclusive feature bundling, which significantly speed up training and reduce memory usage. The key hyperparameters of LightGBM include the number of boosting iterations, learning rate, number of leaves, and the objective function.

### CatBoost

CatBoost (Categorical Boosting) is a gradient boosting library that is specifically designed to handle categorical features without the need for extensive preprocessing. CatBoost uses an ordered boosting approach, which helps to reduce overfitting and improve the generalization ability of the model. It also incorporates techniques such as target-based categorical feature encoding and Bayesian bootstrapping. CatBoost is known for its ease of use, high accuracy, and fast training speed. The key hyperparameters of CatBoost include the number of boosting iterations, learning rate, depth of the trees, and the loss function.

### XGBoost

XGBoost (eXtreme Gradient Boosting) is an optimized implementation of gradient boosting that is designed to be highly efficient, flexible, and portable. XGBoost introduces several enhancements over traditional gradient boosting, such as regularization techniques to prevent overfitting, parallel and distributed computing capabilities, and the use of sparsity-aware algorithms. It supports various objective functions, custom evaluation metrics, and early stopping to improve performance. The key hyperparameters of XGBoost include the number of boosting rounds, learning rate, maximum tree depth, and the objective function.


## Model Performance Evaluation

The performance of each machine learning model is evaluated using the Mean Squared Error (MSE) metric. Below are the explanations for the MSE values of each model:

### SVR MSE: 1.555318016401746

The SVR (Support Vector Regression) model has an MSE of approximately 1.56%. This means that, on average, the SVR model's predictions are off by about 1.56% from the actual stock prices. For instance, if the actual stock price is $100, the SVR model's prediction would typically be within the range of $98.44 to $101.56. This indicates that the SVR model's predictions are the closest to the true values among all the models evaluated.

### RandomForest MSE: 3.806296152736704

The RandomForest model has an MSE of approximately 3.81%. This means that, on average, the RandomForest model's predictions are off by about 3.81% from the actual stock prices. For example, if the actual stock price is $100, the RandomForest model's prediction would typically be within the range of $96.19 to $103.81. This indicates that the RandomForest model's predictions are less accurate compared to the SVR model and have larger deviations from the true values on average.

### LSTM MSE: 1653137682715647.8

The LSTM (Long Short-Term Memory) model has an extremely high MSE, indicating that its predictions are significantly off from the true values. This poor performance suggests that the LSTM model did not capture the underlying patterns in the stock price data effectively.

### XGBoost MSE: 272.8561486297449

The XGBoost model has an MSE of approximately 272.86%. This means that, on average, the XGBoost model's predictions are off by about 272.86% from the actual stock prices. For instance, if the actual stock price is $100, the XGBoost model's prediction would typically be within the range of $-172.86 to $372.86. This high percentage indicates that the XGBoost model's predictions are not very close to the true values. While it performs better than the LSTM model, it is still far from the performance of the SVR and RandomForest models.

### LightGBM MSE: 340.29349361064504

The LightGBM model has an MSE of approximately 340.29%. This means that, on average, the LightGBM model's predictions are off by about 340.29% from the actual stock prices. For example, if the actual stock price is $100, the LightGBM model's prediction would typically be within the range of $-240.29 to $440.29. This high MSE indicates poor performance in predicting stock prices, with large deviations from the true values on average.

### CatBoost MSE: 268.12847603201175

The CatBoost model has an MSE of approximately 268.13%. This means that, on average, the CatBoost model's predictions are off by about 268.13% from the actual stock prices. For instance, if the actual stock price is $100, the CatBoost model's prediction would typically be within the range of $-168.13 to $368.13. This high MSE indicates that the CatBoost model's predictions are also not very close to the true values. However, it performs slightly better than the XGBoost and LightGBM models.
