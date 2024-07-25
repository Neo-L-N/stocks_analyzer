import pandas as pd
import numpy as np

def preprocess_data(input_file, output_file):
    print("Starting data preprocessing...")
    # Load the data
    data = pd.read_csv(input_file)

    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Sort data by date and stock name
    data = data.sort_values(by=['Name', 'date'])

    # Handle missing values (if any)
    data = data.dropna()

    # Create additional features
    data['return'] = data['close'].pct_change()
    data['rolling_mean'] = data['close'].rolling(window=5).mean()
    data['rolling_std'] = data['close'].rolling(window=5).std()

    # Drop rows with NaN values created by rolling window
    data = data.dropna()

    # Save the processed data
    data.to_csv(output_file, index=False)
    print(f"Data preprocessing completed. Processed data saved to {output_file}")

if __name__ == "__main__":
    input_file = 'data/raw/all_stocks_5yr.csv'
    output_file = 'data/processed/processed_data.csv'
    preprocess_data(input_file, output_file)
