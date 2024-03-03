import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate sample weekly stock prices data
np.random.seed(42)  # For reproducibility
data = np.random.normal(0, 1, 300).cumsum() + 100
dates = pd.date_range(start='2019-01-01', periods=300, freq='W')
stock_prices = pd.DataFrame(data, index=dates, columns=['Close'])

# Create features and labels for classification
# For simplicity, we use price changes as the features
stock_prices['Previous Close'] = stock_prices['Close'].shift(1)
stock_prices['Price Change'] = stock_prices['Close'] - stock_prices['Previous Close']
stock_prices.dropna(inplace=True)  # Remove rows with NaN values

# The label is 1 if the price increased, 0 if it decreased
stock_prices['Label'] = (stock_prices['Price Change'] > 0).astype(int)

# Split data into features (X) and labels (y)
X = stock_prices[['Previous Close', 'Price Change']]
y = stock_prices['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# Predict on the testing set
y_pred = gbc.predict(X_test)

# Calculate the accuracy
accuracy = gbc.score(X_test, y_test)
print(f'Model accuracy: {accuracy:.2%}')

# Plotting feature importance
plt.figure(figsize=(10, 6))
feature_importance = gbc.feature_importances_
plt.barh(X.columns, feature_importance)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance for Stock Price Direction Prediction')
plt.show()
