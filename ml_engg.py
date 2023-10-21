import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data.csv')
df=df.drop(['Unnamed: 0'],axis=1)
df['Date'] = pd.to_datetime(df['Date'])
# Extract features (day of the year) and target variable (Receipt_Count)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)
from sklearn.preprocessing import MinMaxScaler
target_variable = 'Receipt_Count'
data = df[[target_variable]]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define the number of previous time steps to use for prediction
look_back = 100

# Create input features and target variable
x, y = [], []
for i in range(look_back, len(scaled_data)):
    x.append(scaled_data[i - look_back:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)


class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, x, y):
        x = self._transform_x(x)
        y = self._transform_y(y)

        betas = self._estimate_coefficients(x, y)

        self.intercept = betas[0]
        self.coefficients = betas[1:]

    def predict(self, x):
        predictions = []
        for values in x:
            pred = np.multiply(values, self.coefficients)
            pred = sum(pred)
            pred += self.intercept
            predictions.append(pred)

        return predictions

    def r2_score(self, y_true, y_pred):
        y_values = y_true
        y_average = np.average(y_values)

        residual_sum_of_squares = 0
        total_sum_of_squares = 0

        for i in range(len(y_values)):
            residual_sum_of_squares += (y_values[i] - y_pred[i])**2
            total_sum_of_squares += (y_values[i] - y_average)**2

        return 1 - (residual_sum_of_squares/total_sum_of_squares)

    def _transform_x(self, x):
        ones_column = np.ones((x.shape[0], 1))
        return np.hstack((ones_column, x))

    def _transform_y(self, y):
        return y

    def _estimate_coefficients(self, x, y):
        xT = x.transpose()
        inversed = np.linalg.inv(xT.dot(x))
        coefficients = inversed.dot(xT).dot(y)
        return coefficients

lr = LinearRegression()
# fit our LR to our data
lr.fit(x, y)

# Create input sequences for 2022
x_2022 = scaled_data[-look_back:].reshape(1, -1)  # Use the last 60 days of available data as initial input for 2022

# Generate forecasts for 2022 (assuming 365 days in a year)
forecasted_values_2022 = []

for _ in range(365):
    # Predict the next day
    next_day_prediction = lr.predict(x_2022)
    
    # Append the prediction to the forecasted values list
    forecasted_values_2022.append(next_day_prediction[0])
    
    # Update the input sequence: remove the first element and append the prediction
    x_2022 = np.append(x_2022[:, 1:], [next_day_prediction], axis=1)

# Inverse transform the forecasted values to get the actual receipt counts
forecasted_values_2022 = scaler.inverse_transform(np.array(forecasted_values_2022).reshape(-1, 1))

# Create a date range for 2022
dates_2022 = pd.date_range(start='2022-01-01', end='2022-12-31')

# Create a DataFrame for 2022 predictions
predictions_2022_df = pd.DataFrame(data=forecasted_values_2022, index=dates_2022, columns=['Predictions'])

import pickle

# Save the model to a file
with open('linear_regression_model.pkl', 'wb') as model_file:
    pickle.dump(lr, model_file)

print('Model saved successfully.')

# Split the data into training and testing sets (80% training, 20% testing)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
# Calculate Mean Squared Error (MSE)
mse = np.mean((predictions - y_test) ** 2)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
#print(f'MSE: {mse}, rmse: {rmse}')


