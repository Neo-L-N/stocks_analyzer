import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

def train_xgboost(input_file, model_file):
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

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    input_file = 'data/processed/features_data.csv'
    model_file = 'models/xgboost_model.joblib'
    train_xgboost(input_file, model_file)
