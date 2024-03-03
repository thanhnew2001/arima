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
stock_prices['Percentage Change'] = stock_prices['Price Change'] / stock_prices['Previous Close'] * 100
stock_prices.dropna(inplace=True)  # Removing any rows with NaN values which are resultant of shifts

# Creating binary outcomes: 1 if price increased, 0 if it decreased
stock_prices['Label'] = (stock_prices['Price Change'] > 0).astype(int)

# Define the features and labels
X = stock_prices[['Previous Close', 'Price Change', 'Percentage Change']]
y = stock_prices['Label']

# Split the dataset into training and testing sets
# Let's use the first 80% of the data for training and the remaining 20% for testing
split_ratio = 0.8
split_index = int(len(stock_prices) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Initialize and train the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = gbc.predict(X_test)

# Calculate and print the accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy on test set: {accuracy:.2%}')

# Optional: Plotting feature importance
plt.figure(figsize=(10, 6))
feature_importance = gbc.feature_importances_
plt.barh(X.columns, feature_importance)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance for Stock Price Direction Prediction')
plt.show()
