import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor
import joblib

def train_lightgbm(input_file, model_file):
    print("Loading data...")
    data = pd.read_csv(input_file)
    print("Data loaded successfully")

    features = ['open', 'high', 'low', 'volume', 'return', 'rolling_mean', 'rolling_std']
    target = 'close'
    X = data[features]
    y = data[target]

    print("Selecting features and target variable...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets")

    model = LGBMRegressor()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50],
        'boosting_type': ['gbdt'],
        'objective': ['regression'],
    }

    print("Starting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("GridSearchCV completed")

    print(f"Best parameters found: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    print("Saving the best model...")
    joblib.dump(best_model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    input_file = 'data/processed/features_data.csv'
    model_file = 'models/lightgbm_model.joblib'
    train_lightgbm(input_file, model_file)

