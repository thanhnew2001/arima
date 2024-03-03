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

# Initialize variables for while loop
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
model = ARIMA(train['Close'], order=(5, 1, 2))
model_fit = model.fit()

# Forecast
forecast_steps = len(test)  # Number of steps to forecast is same as test set length
forecast = model_fit.forecast(steps=forecast_steps)[0]  # Ensure this is treated as a series

# Correcting binary_predictions generation
if isinstance(forecast, np.ndarray):  # Check if forecast is an array
    binary_predictions = ['up' if forecast[i] > (train['Close'].iloc[-1] if i == 0 else forecast[i-1]) else 'down' for i in range(len(forecast))]
else:  # Single value case (should not normally happen for multiple forecasts)
    binary_predictions = []

# Actual binary outcomes
actuals = test['Close'].values
binary_actuals = ['up' if actuals[i] > (train['Close'].iloc[-1] if i == 0 else actuals[i-1]) else 'down' for i in range(len(actuals))]

# Calculate accuracy
accuracy = sum(1 for i in range(len(binary_actuals)) if i < len(binary_predictions) and binary_predictions[i] == binary_actuals[i]) / len(binary_actuals) if binary_actuals else 0
print(f'Model accuracy on test set: {accuracy:.2%}')

# Plotting the results for visualization
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Close'], label='Training Data')
plt.plot(test.index, test['Close'], label='Actual Price', color='green')
plt.plot(test.index[:len(forecast)], forecast, label='Forecasted Price', color='red')  # Match forecast length
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
