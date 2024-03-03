import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to fetch weekly data from Yahoo Finance
def fetch_weekly_data(ticker):
    data = yf.download(ticker, start='2015-01-01', end=pd.to_datetime('today'), interval='1wk')
    return data['Close']
    
# Function to prepare data, train and predict trends
def prepare_predict_visualize(ticker):
    # Fetch data from Yahoo Finance
    stock_data = fetch_weekly_data(ticker)
    
    # Prepare the data
    stock_data = pd.DataFrame(stock_data)
    stock_data['Previous Close'] = stock_data[ticker].shift(1)
    stock_data['Price Change'] = stock_data[ticker] - stock_data['Previous Close']
    stock_data['Up or Down'] = stock_data['Price Change'].apply(lambda x: 'Up' if x > 0 else 'Down')
    stock_data['Label'] = (stock_data['Price Change'] > 0).astype(int)
    stock_data.dropna(inplace=True)  # Remove rows with NaN values
    
    # Define the features and labels
    X = stock_data[['Previous Close', 'Price Change']]
    y = stock_data['Label']
    
    # Split the dataset into training and testing sets
    split_ratio = 0.8
    split_index = int(len(stock_data) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    test_weeks = stock_data.index[split_index:]
    
    # Initialize and train the Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train, y_train)
    
    # Predict the labels for the test set
    y_pred = gbc.predict(X_test)
    
    # Create a DataFrame to visualize test data and predictions
    results = pd.DataFrame({
        'Week': test_weeks,
        'Close Price': X_test['Previous Close'],
        'Actual Trend': stock_data['Up or Down'][split_index:],
        'Predicted Trend': ['Up' if pred == 1 else 'Down' for pred in y_pred]
    })
    
    # Print and plot results for each stock
    print(f"\nResults for {ticker}:")
    print(results)
    plt.figure()
    plt.plot(results['Week'], results['Close Price'], label='Close Price')
    plt.scatter(results['Week'], results['Predicted Trend'] == 'Up', color='green', label='Predicted Up', marker='^')
    plt.scatter(results['Week'], results['Predicted Trend'] == 'Down', color='red', label='Predicted Down', marker='v')
    plt.title(f"Weekly Closing Prices and Predicted Trends for {ticker}")
    plt.xlabel('Week')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Main function to run the analysis for each stock
def main():
    tickers = ['SPY', 'QQQ', 'IWM']
    for ticker in tickers:
        prepare_predict_visualize(ticker)

if __name__ == "__main__":
    main()
