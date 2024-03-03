import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Function to perform Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    p_value = result[1]  # Extract the p-value
    return p_value < 0.05  # Return True if series is stationary

# Initialize variables for the while loop
stationary = False
max_iterations = 1000  # Prevent infinite loop, adjust as necessary
iterations = 0

# Keep generating data until it is stationary or we hit the max number of iterations
while not stationary and iterations < max_iterations:
    iterations += 1
    # Generate sample weekly stock prices data
    data = np.random.normal(0, 1, 300).cumsum() + 100  # Random walk model
    # Convert to pandas DataFrame
    dates = pd.date_range(start='2019-01-01', periods=300, freq='W')
    stock_prices = pd.DataFrame(data, index=dates, columns=['Close'])
    # Check if generated series is stationary
    stationary = adf_test(stock_prices['Close'])

if stationary:
    print(f'Series became stationary after {iterations} iterations')
else:
    print('Failed to generate a stationary series within the maximum number of iterations')
    # Exit if no stationary series was generated
    exit()

# Split data into train and evaluation sets
train_size = int(len(stock_prices) * 0.8)
train, test = stock_prices[0:train_size], stock_prices[train_size:len(stock_prices)]

# Fit an ARIMA model to the training set (adjust the order based on your data)
model = ARIMA(train['Close'], order=(5,1,2))  # Using a general model, might need optimization
model_fit = model.fit()

# Forecast the next week's price
forecast_result = model_fit.forecast(steps=1)
forecast = forecast_result[0]

# Ensure forecast is an array and retrieve the first element safely
predicted_value = forecast[0] if len(forecast) > 0 else None
last_known_value = train['Close'].iloc[-1]

# Check if predicted_value is not None to avoid further errors
if predicted_value is not None:
    predicted_trend = 'up' if predicted_value > last_known_value else 'down'
    print(f"Next week's predicted trend: {predicted_trend} (predicted price: {predicted_value})")
else:
    print("Error in forecasting.")

# Evaluate the model by comparing with the actual values in the test set
# Here we use the model to predict the values for the test set and compare
predictions = model_fit.forecast(steps=len(test))[0]
actuals = test['Close'].values
# Convert predictions and actuals to binary up/down
binary_predictions = ['up' if predictions[i] > actuals[i-1] else 'down' for i in range(1, len(predictions))]
binary_actuals = ['up' if actuals[i] > actuals[i-1] else 'down' for i in range(1, len(actuals))]

# Calculate accuracy
accuracy = sum(1 for i in range(len(binary_predictions)) if binary_predictions[i] == binary_actuals[i]) / len(binary_predictions)
print(f'Model accuracy on test set: {accuracy:.2%}')
