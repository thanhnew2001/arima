# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# Generate sample weekly stock prices data
np.random.seed(42)  # For reproducibility
data = np.random.normal(0, 1, 300).cumsum() + 100  # Random walk model

# Convert to pandas DataFrame
dates = pd.date_range(start='2019-01-01', periods=300, freq='W')
stock_prices = pd.DataFrame(data, index=dates, columns=['Close'])

# Plot the generated stock price data
plt.figure(figsize=(10, 6))
plt.plot(stock_prices['Close'])
plt.title('Sample Weekly Stock Prices')
plt.xlabel('Week')
plt.ylabel('Price')
plt.show()


# Function to perform Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

# Apply ADF test on the original series
adf_test(stock_prices['Close'])


# Fit an ARIMA model (assuming the data is already stationary)
# These parameters should ideally be optimized based on the data
model = ARIMA(stock_prices['Close'], order=(1,1,1))
model_fit = model.fit(disp=0)

# Forecast the next week's price
forecast, stderr, conf_int = model_fit.forecast(steps=1)
print(f"Next week's predicted price: {forecast[0]}")


