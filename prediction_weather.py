import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('weather.csv')

# Data preprocessing
# Replace 'N.A.' with NaN
data.replace('N.A.', np.nan, inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Convert data types
data['O3 * 10'] = data['O3 * 10'].astype(float)

# Visualize data distribution
sns.pairplot(data)
plt.show()

# Visualize correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Split data into features (X) and target (y)
X = data.drop(['Year', 'Month', 'Day', 'AQI'], axis=1)
y_AQI = data['AQI']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_AQI, test_size=0.2, random_state=42)

# Train linear regression model for AQI
model_AQI = LinearRegression()
model_AQI.fit(X_train, y_train)
predictions_AQI = model_AQI.predict(X_test)

# Evaluate model
mse_AQI = mean_squared_error(y_test, predictions_AQI)
print(f"MSE for AQI: {mse_AQI}")

# Predict air pollution levels (e.g., SO2, NO2, O3) using the trained model
predictions_pollutants = model_AQI.predict(X)

# Visualize actual vs. predicted AQI
plt.scatter(y_test, predictions_AQI)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs. Predicted AQI')
plt.show()

# Visualize actual vs. predicted air pollution levels
for col in X.columns:
    plt.scatter(data[col], predictions_pollutants, label='Predicted')
    plt.scatter(data[col], y_AQI, label='Actual')
    plt.xlabel(col)
    plt.ylabel('Pollutant Level')
    plt.title(f'Actual vs. Predicted {col}')
    plt.legend()
    plt.show()
