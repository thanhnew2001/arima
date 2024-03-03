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

# Initialize lists to hold binary outcomes
binary_outcomes = []

# Manually forecast each step in the test set
for time_point in range(len(test)):
    endog = pd.concat([train['Close'], test['Close'].iloc[:time_point]])  # Use concat instead of append
    model = ARIMA(endog, order=(5, 1, 2))
    model_fit = model.fit()
    next_forecast = model_fit.forecast()[0]  # Forecast the next step
    # Determine if the forecast is an increase or decrease
    if time_point == 0:  # If it's the first forecast, compare with last train value
        outcome = 'increase' if next_forecast > train['Close'].iloc[-1] else 'decrease'
    else:  # Otherwise, compare with the previous forecast
        outcome = 'increase' if next_forecast > endog.iloc[-1] else 'decrease'
    binary_outcomes.append(outcome)  # Append binary outcome

# Calculate and print the accuracy if you have actual future data to compare with
actual_changes = ['increase' if test['Close'].iloc[i] > test['Close'].iloc[i - 1] else 'decrease' for i in range(1, len(test))]
predicted_changes = binary_outcomes[1:]  # Exclude the first forecast since there's no previous actual value to compare
accuracy = sum(1 for i in range(len(actual_changes)) if actual_changes[i] == predicted_changes[i]) / len(actual_changes)
print(f'Binary prediction accuracy on test set: {accuracy:.2%}')
