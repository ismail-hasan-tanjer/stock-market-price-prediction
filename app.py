import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------
# Data Collection
# ----------------------------
@st.cache_data(show_spinner=True)
def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# ----------------------------
# Feature Engineering
# ----------------------------
def add_technical_indicators(df):
    """
    Add technical indicators: Simple Moving Averages (SMA) and RSI.
    """
    # 20-day and 50-day Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = loss.abs()
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# ----------------------------
# Data Preprocessing
# ----------------------------
def preprocess_data(df):
    """
    Preprocess data by creating the target variable and selecting features.
    The target is the next day's closing price.
    """
    # Create target column: next day's Close price
    df['Target'] = df['Close'].shift(-1)
    # Drop rows with NaN values (from rolling calculations and target shift)
    df = df.dropna()
    
    # Select features for training
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'SMA50', 'RSI']
    X = df[features]
    y = df['Target']
    return X, y, df

# ----------------------------
# Model Training and Evaluation
# ----------------------------
def train_predict_model(X, y):
    """
    Train a Linear Regression model using a time-series–aware split.
    Returns the model, train-test splits, predictions, and evaluation metrics.
    """
    # For time-series data, do not shuffle the data when splitting.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return model, X_train, X_test, y_train, y_test, predictions, mse, mae, r2

# ----------------------------
# Visualization
# ----------------------------
def plot_results(df, X_train, y_test, predictions):
    """
    Create an interactive Plotly graph comparing actual and predicted stock prices.
    """
    # Get the date index for the test set (after the training period)
    test_dates = df.iloc[len(X_train):].index
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=test_dates, y=predictions, mode='lines', name='Predicted Price'))
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title="Legend"
    )
    return fig

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.title("Stock Market Prediction System with Machine Learning")
    st.write("Predict stock prices based on historical data using a Machine Learning model.")
    
    # Sidebar for user inputs
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())
    
    if start_date >= end_date:
        st.error("Error: End date must be after the start date.")
        return

    # Fetch and display raw data
    st.subheader("Fetching Data")
    data_load_state = st.text("Loading data...")
    data = fetch_data(ticker, start_date, end_date)
    data_load_state.text("Loading data... done!")
    
    st.subheader("Raw Data (Last 5 Rows)")
    st.write(data.tail())

    # Feature engineering and preprocessing
    data = add_technical_indicators(data)
    X, y, processed_df = preprocess_data(data)
    
    # Model training and evaluation
    st.subheader("Training the Model")
    model, X_train, X_test, y_train, y_test, predictions, mse, mae, r2 = train_predict_model(X, y)
    
    st.write("### Model Performance Metrics")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
    
    # Visualization of actual vs predicted prices
    st.subheader("Prediction vs Actual Prices")
    fig = plot_results(processed_df, X_train, y_test, predictions)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display test data with predictions in a table
    st.subheader("Test Data Predictions")
    test_dates = processed_df.iloc[len(X_train):].index
    results_df = pd.DataFrame({
        "Date": test_dates,
        "Actual Price": y_test,
        "Predicted Price": predictions
    })
    results_df.reset_index(drop=True, inplace=True)
    st.dataframe(results_df)

if __name__ == "__main__":
    main()
