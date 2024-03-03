import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Simulating weekly stock price data
np.random.seed(42)  # Ensuring reproducibility
data = np.random.normal(0, 1, 300).cumsum() + 100  # Simulated stock price data
dates = pd.date_range(start='2019-01-01', periods=300, freq='W')  # Weekly dates
stock_prices = pd.DataFrame(data, index=dates, columns=['Close'])

# Create features: using price differences and percentage changes as examples
stock_prices['Previous Close'] = stock_prices['Close'].shift(1)
stock_prices['Price Change'] = stock_prices['Close'] - stock_prices['Previous Close']
stock_prices['Up or Down'] = stock_prices['Price Change'].apply(lambda x: 'Up' if x > 0 else 'Down')
stock_prices.dropna(inplace=True)  # Removing any rows with NaN values which are resultant of shifts

# Creating binary outcomes: 1 if price increased, 0 if it decreased
stock_prices['Label'] = (stock_prices['Price Change'] > 0).astype(int)

# Define the features and labels
X = stock_prices[['Previous Close', 'Price Change']]
y = stock_prices['Label']

# Split the dataset into training and testing sets
split_ratio = 0.8
split_index = int(len(stock_prices) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
test_weeks = stock_prices.index[split_index:]

# Initialize and train the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = gbc.predict(X_test)

# Create a DataFrame to visualize test data and predictions
results = pd.DataFrame({
    'Week': test_weeks,
    'Close Price': X_test['Previous Close'],
    'Up or Down': stock_prices['Up or Down'][split_index:],
    'Predicted Value': ['Up' if pred == 1 else 'Down' for pred in y_pred],
    'True or False': ['True' if (pred == 1 and actual == 1) or (pred == 0 and actual == 0) else 'False' for pred, actual in zip(y_pred, y_test)]
})

print(results)
