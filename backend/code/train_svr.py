import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def train_svr(input_file, model_file):
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

    # Further reduce dataset size for quicker training
    X_train = X_train.sample(500, random_state=42)
    y_train = y_train.loc[X_train.index]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    # Simplify the parameter grid
    param_grid = {
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': [1, 10],
        'svr__gamma': ['scale']
    }

    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("GridSearchCV completed")

    print(f"Best parameters found: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    input_file = 'data/processed/features_data.csv'
    model_file = 'models/svr_model.joblib'
    train_svr(input_file, model_file)

