import pandas as pd
from dash import Dash, dcc, html, Input, Output

# Load data with predictions
data = pd.read_csv("../data/processed/processed_data_with_predictions.csv")

# Format date and sort the data
data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")
data = data.sort_values(by="date")

# Get unique stock names
stocks = data["Name"].sort_values().unique()

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

# Initialize the app
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Stock Prediction Dashboard"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(children="Stock Price Prediction", className="header-title"),
                html.P(children="Compare actual stock prices vs. predicted values", className="header-description"),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Stock", className="menu-title"),
                        dcc.Dropdown(
                            id="stock-filter",
                            options=[{"label": stock, "value": stock} for stock in stocks],
                            value=stocks[0],  # Default to the first stock
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Date Range", className="menu-title"),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=data["date"].min().date(),
                            max_date_allowed=data["date"].max().date(),
                            start_date=data["date"].min().date(),
                            end_date=data["date"].max().date(),
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Select Graphs to Display", className="menu-title"),
                        dcc.Checklist(
                            id="graph-options",
                            options=[
                                {"label": "Actual Values", "value": "actual"},
                                {"label": "SVR Predicted", "value": "svr_predicted"},
                                {"label": "RF Predicted", "value": "rf_predicted"},
                            ],
                            value=["actual", "svr_predicted", "rf_predicted"],  # Default to show all graphs
                            inline=True,
                        ),
                    ],
                ),
            ],
            className="menu",
        ),
        html.Div(
            id="graph-container",
            className="wrapper",
        ),
    ]
)

@app.callback(
    Output("graph-container", "children"),
    [Input("stock-filter", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("graph-options", "value")]
)
def update_graph(stock, start_date, end_date, selected_graphs):
    filtered_data = data.query(
        "Name == @stock and date >= @start_date and date <= @end_date"
    )

    # Initialize graph traces
    traces = []

    # Check for actual values
    if "actual" in selected_graphs:
        traces.append({
            "x": filtered_data["date"],
            "y": filtered_data["close"],
            "type": "lines",
            "name": "Actual Values",
        })

    # Check for SVR predicted values
    if "svr_predicted" in selected_graphs:
        traces.append({
            "x": filtered_data["date"],
            "y": filtered_data["svr_predicted"],
            "type": "lines",
            "name": "SVR Predicted",
        })

    # Check for RF predicted values
    if "rf_predicted" in selected_graphs:
        traces.append({
            "x": filtered_data["date"],
            "y": filtered_data["rf_predicted"],
            "type": "lines",
            "name": "RF Predicted",
        })

    # Create the graph
    return [
        dcc.Graph(
            figure={
                "data": traces,
                "layout": {
                    "title": f"Stock Price Predictions for {stock}",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Stock Price"},
                },
            },
            config={"displayModeBar": False},
        )
    ]


if __name__ == "__main__":
    app.run_server(debug=True)
