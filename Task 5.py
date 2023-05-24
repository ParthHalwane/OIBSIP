# -*- coding: utf-8 -*-
"""Taskno5.ipynb"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score

# Load the dataset
data = pd.read_csv('/content/Advertising.csv')

# Display the first few rows of the dataset
data.head()

# Get information about the dataset
print(data.info())

# Summary statistics of the dataset
print(data.describe())

# Split the data into input features (X) and target variable (y)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Linear regression"""

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

"""Residual Plot"""

# Scatter plot of actual vs. predicted sales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Linear Regression: Actual vs. Predicted Sales')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Linear Regression: Residual Plot')
plt.show()

"""Ridge Regression"""

from sklearn.linear_model import Ridge

# Create and train the Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha value
ridge_model.fit(X_train, y_train)

# Make predictions on the testing set using Ridge Regression
ridge_pred = ridge_model.predict(X_test)

# Evaluate the models
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = mean_squared_error(y_test, ridge_pred, squared=False)
ridge_r2 = r2_score(y_test, ridge_pred)

print("Ridge Regression:")
print("Mean Squared Error:", ridge_mse)
print("Root Mean Squared Error:", ridge_rmse)
print("R-squared:", ridge_r2)

"""Lasso Regression"""

from sklearn.linear_model import Lasso

# Create and train the Lasso Regression model
lasso_model = Lasso(alpha=1.0)  # You can adjust the alpha value
lasso_model.fit(X_train, y_train)

# Make predictions on the testing set using Lasso Regression
lasso_pred = lasso_model.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = mean_squared_error(y_test, lasso_pred, squared=False)
lasso_r2 = r2_score(y_test, lasso_pred)

print("\nLasso Regression:")
print("Mean Squared Error:", lasso_mse)
print("Root Mean Squared Error:", lasso_rmse)
print("R-squared:", lasso_r2)

"""Random Forest Regressor """

from sklearn.ensemble import RandomForestRegressor
# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

"""Neural Network"""

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

# Make predictions on the testing set
y_pred = model.predict(X_test).flatten()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

"""Decision Tree Regressor"""

from sklearn.tree import DecisionTreeRegressor
# Create and train the Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

"""K-Nearest Neighbor"""

from sklearn.neighbors import KNeighborsRegressor

# Apply k-Nearest Neighbors (KNN)
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# Evaluate the KNN model
knn_mse = mean_squared_error(y_test, knn_pred)
knn_rmse = mean_squared_error(y_test, knn_pred, squared=False)
knn_r2 = r2_score(y_test, knn_pred)

print("Mean Squared Error:", knn_mse)
print("Root Mean Squared Error:", knn_rmse)
print("R-squared:", knn_r2)

"""Support Vector Machine"""

from sklearn.svm import SVR

# Apply Support Vector Machines (SVM)
svm_model = SVR(kernel='linear')  # You can choose a different kernel if desired
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Evaluate the SVM model
svm_mse = mean_squared_error(y_test, svm_pred)
svm_rmse = mean_squared_error(y_test, svm_pred, squared=False)
svm_r2 = r2_score(y_test, svm_pred)

print("Mean Squared Error:", svm_mse)
print("Root Mean Squared Error:", svm_rmse)
print("R-squared:", svm_r2)

"""SVM Hyperplane and Input classes"""

import numpy as np

# Scatter plot of actual vs. predicted sales
plt.scatter(data['TV'], data['Sales'], c=data['Sales'], cmap='viridis')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('Input Classes and Hyperplane')

# Plot the hyperplane
support_vectors = svm_model.support_vectors_
w = svm_model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(min(data['TV']), max(data['TV']), 100)
yy = a * xx - (svm_model.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-', linewidth=2)

plt.colorbar(label='Sales')
plt.show()


plt.scatter(data['Newspaper'], data['Sales'], c=data['Sales'], cmap='viridis')
plt.xlabel('Newspaper')
plt.ylabel('Sales')
plt.title('Input Classes and Hyperplane')

# Plot the hyperplane
support_vectors = svm_model.support_vectors_
w = svm_model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(min(data['Newspaper']), max(data['Newspaper']), 100)
yy = a * xx - (svm_model.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-', linewidth=2)

plt.colorbar(label='Sales')
plt.show()


plt.scatter(data['Radio'], data['Sales'], c=data['Sales'], cmap='viridis')
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.title('Input Classes and Hyperplane')

# Plot the hyperplane
support_vectors = svm_model.support_vectors_
w = svm_model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(min(data['Radio']), max(data['Radio']), 100)
yy = a * xx - (svm_model.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-', linewidth=2)

plt.colorbar(label='Sales')
plt.show()

"""Scatter Plots"""

import matplotlib.pyplot as plt

# Plotting TV vs. Sales
plt.scatter(data['TV'], data['Sales'])
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('TV vs. Sales')
plt.show()

# Plotting Radio vs. Sales
plt.scatter(data['Radio'], data['Sales'])
plt.xlabel('Radior')
plt.ylabel('Sales')
plt.title('Radio vs. Sales')
plt.show()

# Plotting Newspaper vs. Sales
plt.scatter(data['Newspaper'], data['Sales'])
plt.xlabel('Newspaper')
plt.ylabel('Sales')
plt.title('Newspaper vs. Sales')
plt.show()

"""Histogram"""

plt.hist(data['Sales'], bins=10)
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.title('Distribution of Sales')
plt.show()

plt.figure()
data['Sales'].plot.kde()
plt.xlabel('Sales')
plt.ylabel('Density')
plt.title('Density Plot of Sales')
plt.show()

"""Heatmap and correlation matrix"""

import seaborn as sns

correlation_matrix = data[['TV', 'Radio', 'Newspaper', 'Sales']].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

"""Feature importance plot (Random Forest example)"""

importances = model.feature_importances_
feature_names = ['TV', 'Radio', 'Newspaper']

plt.bar(feature_names, importances)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
