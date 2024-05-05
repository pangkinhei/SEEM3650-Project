import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Data Preparation
data = pd.read_csv('ap+pop+AQI+AQHI.csv')

# Handle missing or invalid values
data.replace('N.A.', np.nan, inplace=True)
data.dropna(inplace=True)

# Convert date to datetime format
data['DATE'] = pd.to_datetime(data['DATE'])

# Clean non-numeric characters from data
data.replace(to_replace=r'[^0-9.-]+', value='', regex=True, inplace=True)

# Step 2: Feature Engineering
# For AQI prediction, select relevant features
X_aqi = data[['CO', 'FSP', 'NO2', 'NOX', 'O3', 'RSP', 'SO2', '2019 population']]
y_aqi = data['AQI']

# For AQHI prediction, select relevant features
X_aqhi = data[['FSP', 'NOX', 'SO2', '2019 population']]
y_aqhi = data['AQHI']

# Convert data to numeric format
X_aqi = X_aqi.apply(pd.to_numeric, errors='coerce')
y_aqi = pd.to_numeric(y_aqi, errors='coerce')

X_aqhi = X_aqhi.apply(pd.to_numeric, errors='coerce')
y_aqhi = pd.to_numeric(y_aqhi, errors='coerce')

# Step 3: Model Training
# Train model for AQI prediction
X_train_aqi, X_test_aqi, y_train_aqi, y_test_aqi = train_test_split(X_aqi, y_aqi, test_size=0.2, random_state=42)
model_aqi = LinearRegression()
model_aqi.fit(X_train_aqi, y_train_aqi)

# Train model for AQHI prediction
X_train_aqhi, X_test_aqhi, y_train_aqhi, y_test_aqhi = train_test_split(X_aqhi, y_aqhi, test_size=0.2, random_state=42)
model_aqhi = LinearRegression()
model_aqhi.fit(X_train_aqhi, y_train_aqhi)

# Step 4: Model Evaluation
# Evaluate model for AQI prediction
y_pred_aqi = model_aqi.predict(X_test_aqi)
mse_aqi = mean_squared_error(y_test_aqi, y_pred_aqi)
print("Mean Squared Error for AQI Prediction:", mse_aqi)

# Evaluate model for AQHI prediction
y_pred_aqhi = model_aqhi.predict(X_test_aqhi)
mse_aqhi = mean_squared_error(y_test_aqhi, y_pred_aqhi)
print("Mean Squared Error for AQHI Prediction:", mse_aqhi)


# Step 5: Data Visualization
# Visualize the relationship between actual and predicted AQI values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_aqi, y_pred_aqi, color='green')
plt.title('Actual AQI vs Predicted AQI')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.show()

# Visualize the relationship between actual and predicted AQHI values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_aqhi, y_pred_aqhi, color='blue')
plt.title('Actual AQHI vs Predicted AQHI')
plt.xlabel('Actual AQHI')
plt.ylabel('Predicted AQHI')
plt.show()

