from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the data for stock predictions
data = pd.read_csv("../data/processed/processed_data_with_predictions.csv")

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    # Return the unique stock names
    stocks = data["Name"].unique().tolist()
    return jsonify(stocks)

@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    stock_name = request.args.get("stock")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    # Filter the data based on the stock name and date range
    filtered_data = data.query(
        "Name == @stock_name and date >= @start_date and date <= @end_date"
    )

    # Prepare the response
    response = {
        "date": filtered_data["date"].tolist(),
        "actual_values": filtered_data["close"].tolist(),
        "svr_predicted": filtered_data["svr_predicted"].tolist(),
        "rf_predicted": filtered_data["rf_predicted"].tolist(),
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
