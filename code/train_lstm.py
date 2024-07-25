import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm(input_file, model_file):
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    print("Building LSTM model...")
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    print("Training LSTM model...")
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

    model.save(model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    input_file = 'data/processed/features_data.csv'
    model_file = 'models/lstm_model.h5'
    train_lstm(input_file, model_file)

