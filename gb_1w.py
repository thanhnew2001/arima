import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime

# Function to fetch weekly data from Yahoo Finance
def fetch_weekly_data(ticker):
    # Fetching 5 years of data ending today
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=5*365)
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1wk')
    return data['Close']

# Prepare data for a given stock and make a prediction using the Gradient Boosting Classifier
def prepare_and_predict(stock_data, ticker):
    stock_data['Previous Close'] = stock_data.shift(1)
    stock_data['Price Change'] = stock_data - stock_data['Previous Close']
    stock_data['Label'] = (stock_data['Price Change'] > 0).astype(int)
    stock_data.dropna(inplace=True)  # Remove NaN values

    # Define features and labels
    X = stock_data[['Previous Close', 'Price Change']]
    y = stock_data['Label']

    # Training the model with all available data
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X, y)

    # Predicting the trend for the next week (using the most recent 'Previous Close' and assuming 'Price Change' as 0)
    prediction = gbc.predict([[X.iloc[-1]['Previous Close'], 0]])
    predicted_trend = 'Up' if prediction == 1 else 'Down'

    # Return the predicted trend
    return predicted_trend

# Main function to fetch data, predict trends, and print results
def main():
    tickers = ['SPY', 'QQQ', 'IWM']
    predictions = {}

    for ticker in tickers:
        print(f"Fetching weekly data for {ticker}...")
        stock_data = fetch_weekly_data(ticker)
        print(f"Predicting next week's trend for {ticker}...")
        predicted_trend = prepare_and_predict(stock_data, ticker)
        predictions[ticker] = predicted_trend

    # Displaying the predictions
    print("\nPredicted Trends for the Next Week:")
    for ticker, trend in predictions.items():
        print(f"{ticker}: {trend}")

if __name__ == "__main__":
    main()
