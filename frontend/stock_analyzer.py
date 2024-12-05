
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout='wide')
# Inject custom CSS for centering
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        color: #2E86C1; /* Blue for the main title */
        font-size: 2.5em;
        font-weight: bold;
    }
    .header {
        color: #2B3036; /* Dark gray for headers */
        font-size: 1.8em;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .subheader {
        color: #2874A6; /* Medium blue for subheaders */
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 15px;
    }
    .markdown-text {
        color: #566573; /* Light gray for general text */
        font-size: 1.1em;
        line-height: 1.6;
    }
    .highlight {
        color: #148F77; /* Teal for key points or highlights */
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Use the custom CSS class for the title
st.markdown("<h1 class='centered-title'>Stock Price Analyzer with Machine Learning</h1>", unsafe_allow_html=True)

st.image('media/stocks.jpeg', use_column_width=True)

# Headers and General Info
st.markdown("<div class='header'>Machine Learning Models for Stock Price Prediction</div>", unsafe_allow_html=True)
st.markdown("""
<div class='markdown-text'>
This project aims to predict stock prices using historical data of S&P 500 stocks. Multiple machine learning models,
including <span class='highlight'>Support Vector Regression (SVR)</span>, <span class='highlight'>Random Forest (RF)</span>,
<span class='highlight'>Long Short-Term Memory (LSTM)</span>, <span class='highlight'>LightGBM</span>, 
<span class='highlight'>CatBoost</span>, and <span class='highlight'>XGBoost</span> are trained and evaluated based on their performance.
</div>
""", unsafe_allow_html=True)

# Train-Test Section
st.markdown("<div class='subheader'>Train and Test Split</div>", unsafe_allow_html=True)
st.markdown("""
<div class='markdown-text'>
   <span class='highlight'>Training Set</span>: Initially 80% of the data, further reduced to 500 samples for quicker training.<br>
   <span class='highlight'>Testing Set</span>: 20% of the data.<br><br>
This sampling helps speed up the training process but may impact the model's performance depending on the dataset's size and characteristics.
</div>
""", unsafe_allow_html=True)
st.image('media/traintest.png', width=600)

# Dropdown for Model Selection
st.markdown("<div class='subheader'>Choose a Model to See How It Works</div>", unsafe_allow_html=True)
model = st.selectbox('',
                     options=['Choose a Model', 'Support Vector Regression (SVR)', 'Random Forest (RF)',
                              'Long Short-Term Memory (LSTM)', 'Light Gradient-Boosting Machine (LightGBM)',
                              'CatBoost', 'XGBoost'])
if model == 'Support Vector Regression (SVR)':

    st.markdown("""
        <div class='markdown-text'>
        <span class='highlight'>Support Vector Regression (SVR)</span> is a tool used to make predictions, especially when 
        the relationship between variables isn't straightforward. It works by finding a line (or curve) that best fits the 
        data while keeping errors as small as possible.
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.video('media/svm.mp4')


elif model == 'Random Forest (RF)':

    st.markdown("""
    <div class='markdown-text'>
    <span class='highlight'>Random Forest</span> is a prediction tool that works by creating a forest
    of decision trees and combining their results. Each tree is like an independent expert making its
    own prediction, and the forest combines all these opinions for a more reliable result.<br>
    Here’s how Random Forest works:
    \n\nEach tree looks at a different random portion of the data, which helps it focus on unique aspects
    of the problem.
    \n*    At each step of building a tree, it picks a random subset of features to decide how to 
    split the data.
    \n\n By combining all the trees:
    \n*    For tasks like predicting numbers (regression), it averages their results.
    \n*    For tasks like deciding categories (classification), it goes with the majority vote.
    \n\nThis randomness helps Random Forest avoid overfitting (where a model memorizes the data too 
    much and struggles with new data) and ensures more accurate and stable predictions.
    </div >
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.video('media/random_forest.mp4')

elif model == 'Long Short-Term Memory (LSTM)':
    st.markdown("""
    <div class='markdown-text'>
    <span class='highlight'>Long Short-Term Memory (LSTM)</span> is a type of advanced computer model that helps us understand patterns
    over time. Think of it as a really smart memory system that can remember important information from the past while deciding what details to forget. This makes it especially useful for tasks where the 
    order of events matters, like predicting stock prices, where yesterday's prices affect today's.
    Here's how it works:
    \n\nLSTM is made up of cells, which are like tiny decision-makers.
    \n\nEach cell has three gates (like doors) that decide:\n\n
    \n*  What new information to let in.
    \n*  What old information to forget.\n
    \n*    What information to pass on to the next step.
    \n\nThis design allows LSTM to focus on the important parts of a sequence and ignore the unimportant ones. Unlike older 
    models, it doesn’t lose track of details from the distant past, which is a big advantage for understanding 
    long-term trends.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.video('media/lstm.mp4')

elif model == 'Light Gradient-Boosting Machine (LightGBM)':
    st.markdown("""
    <div class='markdown-text'>
    <span class='highlight'>LightGBM </span> is a tool used to make accurate predictions quickly and efficiently. Think of it as a powerful
    problem-solver designed to handle large and complex datasets. It uses a technique called gradient
    boosting, which builds smarter decision trees to improve accuracy with each step.
    That makes LightGBM special is how it works faster and uses less memory:
    \n*  It organizes data in a way that speeds up learning (called histogram-based learning).
    \n*  It grows decision trees in a unique way that focuses on the most important parts of the data.\n
    \n*  It bundles similar features together to save time and resources.
    \n\nLightGBM is great for solving problems where there’s a lot of data and many details to consider.
    It can be fine-tuned by adjusting settings like how many trees to build, how fast it learns, 
    and what kind of problem it’s solving.
    </div>
    """, unsafe_allow_html=True)

    st.image('media/lgbm.png', width=800)

elif model == 'CatBoost':
    st.markdown("""
        <div class='markdown-text'>
        <span class='highlight'>CatBoost </span> is a tool made for predictions, especially when working with data that includes categories like 
        'Yes/No' or 'Red/Blue/Green.' It’s designed to handle this kind of data easily without much preparation.
        \n\nWhat makes CatBoost stand out:
        \n*  It uses a clever method called ordered boosting to avoid mistakes like overfitting 
        (where a model learns too much from past data and struggles to work with new data).
        \n*  It has special ways of working with categories that improve accuracy, such as assigning smart numerical values to categories.\n
        \n\nCatBoost is known for being easy to use, fast to train, and very accurate. You can adjust settings 
        like the number of steps it takes to improve, how quickly it learns, and how deep its decision trees go 
        to tailor it to your needs.
        </div>
    """, unsafe_allow_html=True)

    st.image('media/catboost.png', width=800)

elif model == 'XGBoost':
    st.markdown("""
        <div class='markdown-text'>
        <span class='highlight'>XGBoost </span> is like an upgraded version of traditional tools for making predictions. It’s designed to work 
        efficiently and handle big tasks without slowing down.
        \n\nWhy XGBoost is so powerful:
        \n*  It uses techniques to avoid overfitting, which helps it make better predictions on new data.
        \n*  It can run calculations in parallel, meaning it works much faster.\n
        \n*  It’s smart about handling missing or sparse data, so it doesn’t waste time on irrelevant details.\n
        \n\nXGBoost is flexible and can be adjusted for different types of problems by tweaking settings 
        like how many rounds of improvement it goes through, how fast it learns, how complex the trees are 
        and what kind of goal it’s trying to achieve.
        </div>
    """, unsafe_allow_html=True)

    st.image('media/xgboost.png', width=800)

if st.checkbox("What is an MSE?"):
# Introduction to MSE
    st.markdown("<div class='subheader'>What is Mean Squared Error (MSE)?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='markdown-text'>
    Mean Squared Error (MSE) is a popular metric for evaluating regression models. 
    It measures the average squared difference between the predicted values and the actual values. 
    The formula is:
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
    st.markdown("""
    <div class='markdown-text'>
    - A <span class='highlight'>lower MSE</span> indicates better accuracy.<br>
    - A <span class='highlight'>higher MSE</span> suggests larger errors in predictions.<br><br>
    Note: MSE is sensitive to outliers as it squares the differences, giving more weight to larger errors.
    </div>
    """, unsafe_allow_html=True)

    # MSE value graph
    # Data
    models = ["SVR", "Random Forest", "LSTM", "XGBoost", "LightGBM", "CatBoost"]
    mse_values = [1.56, 3.81, 1653137682715647.8, 272.86, 340.29, 268.13]

    # Title
    st.markdown("""
    <div class='header'> Model Performance: Mean Squared Error (MSE) Comparison</div>
    """, unsafe_allow_html=True)


    # Interactive Bar Chart with Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=mse_values,
        text=[f"{val:.2f}" for val in mse_values],
        textposition="outside",
        marker=dict(color=["#EB3170", "#56C597", "#2B3036", "#00AADD", "#F6D157", "#454952"])
    ))

    fig.update_layout(
        plot_bgcolor="#242424",  # Black background for the plot
        paper_bgcolor="#242424",  # Black background for the entire figure
        font=dict(color="#EBEBEB"),  # Typography color
        xaxis_title="Model",
        yaxis_title="Mean Squared Error (MSE)",
        yaxis=dict(type="log", showgrid=True, gridcolor="#454952"),
        title="Mean Squared Error by Model",
        title_font=dict(size=16, color="#EBEBEB"),
    )

    # Enable Zoom and Pan
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        <div class='subheader'>Stock Price Predictions Per Model</div>
        <div class='markdown-text'>
        
        1. **SVR (Support Vector Regression)**  
           - **MSE:** 1.56  
             The most accurate model! Predictions are typically within ±1.56% of the true values.  
             Example: For a 100 dollars stock price, predictions range between 98.44 and 101.56.

        2. **Random Forest**  
           - **MSE:** 3.81  
             Solid performance but less accurate than SVR. Predictions deviate by around ±3.81%.  
             Example: For a 100 dollars stock price, estimates range between 96.19 and 103.81.

        3. **LSTM (Long Short-Term Memory)**  
           - **MSE:** 1,653,137,682,715,647.8  
             A major miss! LSTM failed to capture patterns, resulting in astronomically high errors.

        4. **XGBoost**  
           - **MSE:** 272.86  
             Significant errors, with predictions often ±272.86% off.  
             Example: For a 100 dollars stock price, estimates range from -172.86 to 372.86.

        5. **LightGBM** 
           - **MSE:** 340.29  
             Poor accuracy, with deviations of ±340.29%.  
             Example: For a 100 dollars stock price, predictions range from -240.29 to 440.29.

        6. **CatBoost**  
           - **MSE:** 268.13  
             Slightly better than XGBoost and LightGBM, with errors around ±268.13%.  
             _Example_: Predictions for 100 dollars stock price vary from __-168.13__ to __368.13__.
        </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("__Check the box above to learn about Mean Squared Error (MSE) and view the comparison of models.__")

st.markdown("""
<div class='header'> Stock Selection and Predictions</div>
""", unsafe_allow_html=True)


st.markdown("""In this section, we will concentrate on the predictions made by two of the best-performing models: 
**Support Vector Regression (SVR)** and **Random Forest**. These models were selected based on their superior accuracy
and performance in predicting stock prices, as evaluated through various performance metrics
\n\n__Why SVR and Random Forest?__\n\n
\n*    Support Vector Regression (SVR): SVR is known for its ability to handle high-dimensional 
data and find an optimal hyperplane that best fits the data points while minimizing the 
margin of error.\nIts robust nature and generalization ability make it an excellent choice 
for regression tasks such as stock price prediction.\n\n
\n*    Random Forest: Random Forest is a powerful ensemble learning method that builds multiple decision 
trees and combines their outputs to improve accuracy and reduce overfitting.\nIts strength lies in its 
ability to capture complex patterns in data, making it highly effective for predicting future stock prices.
""", unsafe_allow_html=True)

# Attributes Used in Stock Price Prediction Section
st.markdown("<div class='subheader'>Attributes Used in Stock Price Prediction</div>", unsafe_allow_html=True)

st.markdown("""
<div class='markdown-text'>
In predicting stock prices, several key attributes are utilized to capture the market's dynamics and trends. 
These attributes include:

- **Open**: The price at which a stock first trades upon the opening of an exchange on a given trading day. 
It provides an initial indication of the market's sentiment and can be influenced by news and events occurring after 
the previous day's close.

- **High**: The maximum price at which a stock trades during a trading session. This attribute helps in understanding 
the stock's volatility and the upper resistance levels during the day.

- **Low**: The minimum price at which a stock trades during a trading session. It is crucial for identifying the 
stock's support levels and potential buying opportunities.

- **Volume**: The total number of shares traded during a specific period. High trading volumes can indicate strong 
investor interest and are often associated with significant price movements.

- **Return**: This is typically calculated as the percentage change in the stock's price over a specific period. 
It provides insights into the stock's performance and is a critical measure for investors.

- **Rolling Mean**: Also known as the moving average, it smooths out price data by creating a constantly updated 
average price. This helps in identifying trends over time and is useful for reducing noise in volatile data.

- **Rolling Standard Deviation**: This measures the amount of variation or dispersion of a set of values. 
In the context of stock prices, it helps in understanding the volatility over a specific period.

These attributes are essential for building predictive models as they provide insights into the stock's historical 
performance and market behavior. By analyzing these features, machine learning models can identify patterns and make 
informed predictions about future stock prices.
</div>
""", unsafe_allow_html=True)

st.image('media/candlestick.jpg')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('csv/processed_data_with_predictions.csv')


data = load_data()

# Filter data for the selected stock
stock_choice = st.selectbox('Choose a Stock', data['Name'].unique())

# Filter data based on the selected stock
filtered_data = data[data['Name'] == stock_choice]

# Ensure the data is sorted by date
filtered_data = filtered_data.sort_values(by='date')

# Calculate Rolling Mean and Rolling Std if not already in the data
rolling_window = 20  # Example: 20-day rolling window
filtered_data['rolling_mean'] = filtered_data['close'].rolling(window=rolling_window).mean()
filtered_data['rolling_std'] = filtered_data['close'].rolling(window=rolling_window).std()

# Candlestick Chart Visualization
st.subheader(f"Candlestick Chart with Trends and Volatility for {stock_choice}")

# Create the candlestick chart
fig = go.Figure()

# Add candlestick trace
fig.add_trace(go.Candlestick(
    x=filtered_data['date'],
    open=filtered_data['open'],
    high=filtered_data['high'],
    low=filtered_data['low'],
    close=filtered_data['close'],
    name='Candlestick',
    increasing=dict(line=dict(color='#56C597'), fillcolor='#56C597'),  # Green for increasing (rising candles)
    decreasing=dict(line=dict(color='#EB3170'), fillcolor='#EB3170')   # Pink for decreasing (falling candles)
))

# Add Rolling Mean trace
fig.add_trace(go.Scatter(
    x=filtered_data['date'],
    y=filtered_data['rolling_mean'],
    mode='lines',
    name='Rolling Mean',
    line=dict(color='#F6D157', width=2)
))

# Add Rolling Std trace (as a shaded area to indicate volatility)
fig.add_trace(go.Scatter(
    x=filtered_data['date'],
    y=filtered_data['rolling_std'],
    mode='lines',
    name='Rolling Std',
    line=dict(color='#00AADD', width=1, dash='dot'),
    showlegend=True
))

fig.update_layout(
    height=700,
    xaxis_rangeslider_visible=False,
    xaxis_range=[filtered_data['date'].iloc[-300], filtered_data['date'].iloc[-1]],
    xaxis_title="Date",
    yaxis_title="Stock Price",
    plot_bgcolor="#242424",
    paper_bgcolor="#242424",
    font=dict(size=12, color="#EBEBEB"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color='#EBEBEB')
    ),

    title=f'{stock_choice} - Candlestick',
    title_font=dict(size=16, color="#EBEBEB"),
)

# Show the chart in Streamlit
st.plotly_chart(fig)

st.write("""
This chart demonstrates the stock's **Open**, **High**, **Low**, and **Close** prices using candlesticks. 
The **Rolling Mean** (yellow line) highlights the overall trend, while the shaded area formed by **Rolling Std** 
shows the stock's volatility over time.
""")

# Model and Stock Selection
st.markdown("<div class='subheader'>Stock Price Predictions Per Model</div>", unsafe_allow_html=True)
model_choice = st.radio('Choose a Model to see Predictions', options=['Support Vector Regression (SVR)', 'Random Forest'])

# Filter data based on selections
model_column = 'svr_predicted' if model_choice == 'Support Vector Regression (SVR)' else 'rf_predicted'

# Plotting
st.subheader(f'{model_choice} Predictions vs Actual Closing Prices for {stock_choice}')
fig = px.line(filtered_data, x='date', y=['close', model_column],
              labels={'value': 'Stock Price', 'date': 'Date'},
              title=f'{stock_choice} Actual vs Predicted Closing Price ({model_choice})')

fig.update_layout(
    title=f'{stock_choice} - Actual vs Predicted Closing Price ({model_choice})',  # Title for the graph
    title_font=dict(size=12, color="#EBEBEB"),
    legend_title_text='Price Type',
    width=1200,
    height=600,
    plot_bgcolor="#242424",
    paper_bgcolor="#242424",
    font=dict(size=14, color="#EBEBEB"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color='#EBEBEB')
    ),
    xaxis_title="Date",
    yaxis_title="Stock Price"
)
# Customize line colors
fig.update_traces(
    line=dict(color='#EB3170'),  # Actual close prices
    selector=dict(name='close')
)

fig.update_traces(
    line=dict(color='#56C587'),  # Model prediction
    selector=dict(name=model_column)
)

st.plotly_chart(fig)