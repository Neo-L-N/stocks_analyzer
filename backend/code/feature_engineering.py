import pandas as pd
import numpy as np
import ta  # Technical Analysis library

def engineer_features(input_file, output_file):
    print("Starting feature engineering...")
    data = pd.read_csv(input_file, sep=',')
    print("Data loaded successfully")

    # Add technical indicators
    data['ma_5'] = data['close'].rolling(window=5).mean()
    data['ma_10'] = data['close'].rolling(window=10).mean()
    data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
    data['macd'] = ta.trend.MACD(close=data['close']).macd()
    
    # Drop rows with NaN values created by rolling window
    data = data.dropna()

    data.to_csv(output_file, index=False)
    print(f"Feature engineering completed. Features data saved to {output_file}")

if __name__ == "__main__":
    input_file = 'data/processed/processed_data.csv'
    output_file = 'data/processed/features_data.csv'
    engineer_features(input_file, output_file)
