import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA  # Updated import path for ARIMA
from sklearn.metrics import mean_squared_error

# Generate sample weekly stock prices data
np.random.seed(42)  # For reproducibility
data = np.random.normal(0, 1, 300).cumsum() + 100  # Random walk model

# Convert to pandas DataFrame
dates = pd.date_range(start='2019-01-01', periods=300, freq='W')
stock_prices = pd.DataFrame(data, index=dates, columns=['Close'])

# Split data into train and eval
train_size = int(len(stock_prices) * 0.8)
train, test = stock_prices[0:train_size], stock_prices[train_size:len(stock_prices)]

# Function to perform Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    return result[1]  # Return the p-value

# Apply ADF test on the training series
p_value = adf_test(train['Close'])

# Determine if series is stationary and print the result
if p_value < 0.05:
    print('Series is stationary.')
else:
    print('Series is not stationary, differencing might be needed.')

# Fit an ARIMA model to the training set
model = ARIMA(train['Close'], order=(1,1,1))  # Adjust the order based on your data
model_fit = model.fit()

# Forecast the next week's price and extend for the test period
forecast_steps = len(test)
forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)

# Create a pandas series with the forecasted values
forecast_series = pd.Series(forecast, index=test.index)

# Plot the original data, the training part, and the forecasted part
plt.figure(figsize=(10, 6))
plt.plot(train['Close'], label='Training Data')
plt.plot(test['Close'], label='Actual Price', color='green')
plt.plot(forecast_series, label='Forecasted Price', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Week')
plt.ylabel('Price')
plt.legend()
plt.show()
