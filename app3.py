import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Function to perform Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    p_value = result[1]  # Extract the p-value
    return p_value < 0.05  # Return True if series is stationary

# Initialize variables for the while loop
stationary = False
max_iterations = 1000
iterations = 0

# Keep generating data until it is stationary or max iterations reached
while not stationary and iterations < max_iterations:
    iterations += 1
    data = np.random.normal(0, 1, 300).cumsum() + 100  # Generate data
    dates = pd.date_range(start='2019-01-01', periods=300, freq='W')
    stock_prices = pd.DataFrame(data, index=dates, columns=['Close'])
    stationary = adf_test(stock_prices['Close'])  # Test for stationarity

if stationary:
    print(f'Series became stationary after {iterations} iterations')
else:
    print('Failed to generate a stationary series within max iterations')
    exit()  # Exit if no stationary series was generated

# Split data into train and test sets
train_size = int(len(stock_prices) * 0.8)
train, test = stock_prices[:train_size], stock_prices[train_size:]

# Predicting increase or decrease
binary_predictions = []
last_value = train['Close'].iloc[-1]  # Last value of the training set for the first comparison

for time_point in range(len(test)):
    endog = pd.concat([train['Close'], test['Close'].iloc[:time_point]])  # Concatenate data for ARIMA input
    model = ARIMA(endog, order=(5, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast()[0][0]  # Forecast the next step
    trend = 'increase' if forecast > last_value else 'decrease'
    binary_predictions.append(trend)
    last_value = forecast  # Update last_value for the next prediction

# Actual binary outcomes
actuals = test['Close'].values
binary_actuals = ['increase' if actuals[i] > train['Close'].iloc[-1] if i == 0 else actuals[i-1] else 'decrease' for i in range(len(actuals))]

# Calculate accuracy of predictions
accuracy = sum(1 for i in range(len(binary_predictions)) if binary_predictions[i] == binary_actuals[i]) / len(binary_predictions)
print(f'Model accuracy on test set (increase or decrease): {accuracy:.2%}')

# Note: No plotting required if you're only interested in the prediction accuracy.
