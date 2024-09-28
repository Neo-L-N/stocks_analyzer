import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import openai

# Initialize OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key

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
    {
        "href": "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css",
        "rel": "stylesheet",
    },
    {
        "href": "/assets/style.css",  # Assuming you have the custom CSS file under 'assets' directory
        "rel": "stylesheet",
    },
]

# Initialize the app
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Stock Prediction Dashboard"

# Layout of the dashboard
app.layout = html.Div(
    children=[
        # Dashboard Header
        html.Div(
            children=[
                html.H1(children="Stock Price Prediction", className="header-title"),
                html.P(children="Compare actual stock prices vs. predicted values", className="header-description"),
            ],
            className="header",
        ),
        # Menu for Stock, Date Range, and Graph Options
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
        # Graph Container
        html.Div(
            id="graph-container",
            className="wrapper",
        ),
        # Chatbot Circle and Container
        html.Div(
            id="chatbot-container",
            className="chatbot-container-hidden",
            children=[
                html.Div(
                    id="chat-header",
                    className="chat-header",
                    children=[
                        html.Button("X", id="close-chat", className="close-chat"),
                    ]
                ),
                dcc.Textarea(
                    id="user-input",
                    placeholder="Ask me about stock predictions or trends...",
                    style={"width": "100%", "height": 50, "margin": "10px 0"},
                ),
                html.Button("Submit", id="submit-button", className="btn btn-primary"),
                html.Div(id="chat-response", className="menu-title"),
            ]
        ),
        # Chatbot Circle Button
        html.Div(
            id="chatbot-button",
            className="chatbot-button",
            children="💬",
        ),
    ]
)

# Callback for updating the graphs
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

# Callback for handling ChatGPT responses
@app.callback(
    Output("chat-response", "children"),
    [Input("submit-button", "n_clicks")],
    [State("user-input", "value")]
)
def update_chat_response(n_clicks, user_input):
    # Check if n_clicks is None and handle it
    if n_clicks is not None and n_clicks > 0 and user_input:
        # Send the user input to OpenAI API and get the response using the updated model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for stock prediction analysis."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=100,
            temperature=0.7,
        )
        chat_response = response["choices"][0]["message"]["content"]
        return f"ChatGPT: {chat_response}"
    return ""

# Callback to show or hide the chatbot
@app.callback(
    [Output("chatbot-container", "className"),
     Output("chatbot-button", "className")],
    [Input("chatbot-button", "n_clicks"),
     Input("close-chat", "n_clicks")],
    [State("chatbot-container", "className"),
     State("chatbot-button", "className")]
)
def toggle_chatbot(open_clicks, close_clicks, container_class, button_class):
    if open_clicks or close_clicks:
        if "hidden" in container_class:
            return "chatbot-container-visible", "chatbot-button-hidden"
        else:
            return "chatbot-container-hidden", "chatbot-button-visible"
    return container_class, button_class


if __name__ == "__main__":
    app.run_server(debug=True)

