import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create a synthetic dataset
dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
prices = np.sin(np.linspace(0, 10, len(dates))) * 100 + 100
data = pd.DataFrame({'Date': dates, 'Close': prices})

# Set the date as the index
data.set_index('Date', inplace=True)

# Use the closing price as the target variable
y = data['Close'].values

# Use the previous day's closing price as a feature
X = data['Close'].shift(1).fillna(method='bfill').values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = np.mean((predictions - y_test) ** 2)
print(f'Mean Squared Error: {mse}')

# Plot the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(data.index[-len(y_test):], predictions, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
