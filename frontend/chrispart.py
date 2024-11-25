import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go




if st.checkbox("What is an MSE?"):
    # Introduction to MSE
    st.subheader("What is Mean Squared Error (MSE)?")
    st.write("""
    Mean Squared Error (MSE) is a popular metric for evaluating regression models. 
    It measures the average squared difference between the predicted values and the actual values. 
    The formula is:
    """)
    st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
    st.write("""
    - **Lower MSE values** indicate that the model's predictions are closer to the true values.
    - **Higher MSE values** suggest a larger average error in predictions.

    MSE is sensitive to outliers since it squares the differences, giving more weight to larger errors. 
    Thus, it's essential to compare MSE alongside other metrics for a comprehensive evaluation.
    """)

    # Sample MSE values for the models
    st.subheader("MSE Comparison for Different Models")
    st.write("""
    Here are the MSE values for the following models from our testing:
    - Support Vector Regression (SVR)
    - Random Forest
    - Long Short-Term Memory (LSTM)
    - Light Gradient Boosting Machine (LightGBM)
    - CatBoost
    - XGBoost
    """)

    # MSE value graph
    # Data
    models = ["SVR", "Random Forest", "LSTM", "XGBoost", "LightGBM", "CatBoost"]
    mse_values = [1.56, 3.81, 1653137682715647.8, 272.86, 340.29, 268.13]

    # Title
    st.title("Interactive MSE Comparison")

    # Interactive Bar Chart with Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=mse_values,
        text=[f"{val:.2f}" for val in mse_values],
        textposition="outside",
        marker=dict(color=["blue", "green", "orange", "red", "purple", "cyan"])
    ))

    fig.update_layout(
        title="MSE Comparison of Machine Learning Models",
        xaxis_title="Model",
        yaxis_title="Mean Squared Error (MSE)",
        yaxis=dict(type="log", title="Logarithmic Scale (Zoom Enabled)", showgrid=True),
        template="plotly_white"
    )

    # Enable Zoom and Pan
    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    1. **SVR (Support Vector Regression)**:  
       - **MSE: 1.56**  
         The most accurate model! Predictions are typically within ±1.56% of the true values.  
         Example: For a $100 stock, predictions range between $98.44 and $101.56.

    2. **Random Forest**:  
       - **MSE: 3.81**  
         Solid performance but less accurate than SVR. Predictions deviate by around ±3.81%.  
         Example: For a $100 stock, estimates range between $96.19 and $103.81.

    3. **LSTM (Long Short-Term Memory)**:  
       - **MSE: 1,653,137,682,715,647.8**  
         A major miss! LSTM failed to capture patterns, resulting in astronomically high errors.

    4. **XGBoost**:  
       - **MSE: 272.86**  
         Significant errors, with predictions often ±272.86% off.  
         Example: For a $100 stock, estimates range from -$172.86 to $372.86.

    5. **LightGBM**:  
       - **MSE: 340.29**  
         Poor accuracy, with deviations of ±340.29%.  
         Example: For a $100 stock, predictions range from -$240.29 to $440.29.

    6. **CatBoost**:  
       - **MSE: 268.13**  
         Slightly better than XGBoost and LightGBM, with errors around ±268.13%.  
         Example: Predictions for $100 stocks vary from -$168.13 to $368.13.
    """)

else:
    st.write("Check the box above to learn about Mean Squared Error (MSE) and view the comparison of models.")
