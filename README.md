
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

## Setup and Usage

### Step 1: Install Required Packages

Make sure you have Python 3.7 or later. Install the required packages using pip:

```sh
pip install -r requirements.txt
```
# stocks_analyzer
