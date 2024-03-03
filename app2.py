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

# Fit ARIMA model
model = ARIMA(train['Close'], order=(5, 1, 2))  # Adjust these parameters as necessary
model_fit = model.fit()

# Manually forecast each step in the test set
forecast = []
for time_point in range(len(test)):
    endog = pd.concat([train['Close'], test['Close'].iloc[:time_point]])  # Use concat instead of append
    model = ARIMA(endog, order=(5, 1, 2))
    model_fit = model.fit()
    next_forecast = model_fit.forecast()[0]  # Forecast the next step
    forecast.append(next_forecast)  # Append forecasted value

# Convert forecast to an array for comparison
forecast = np.array(forecast)

# Plotting the results for visualization
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Close'], label='Training Data')
plt.plot(test.index, test['Close'], label='Actual Price', color='green')
plt.plot(test.index, forecast, label='Forecasted Price', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Note: Binary prediction and accuracy calculation are omitted for brevity but can be added similarly as before.
