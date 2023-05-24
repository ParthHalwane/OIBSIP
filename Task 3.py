# -*- coding: utf-8 -*-
"""Task 3.ipynb"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,accuracy_score,r2_score

# Step 1: Data Preprocessing
df = pd.read_csv('/content/car.csv')
# Perform necessary preprocessing steps

# Drop irrelevant columns (e.g., car_ID, CarName)
df = df.drop(['car_ID', 'CarName'], axis=1)

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df)

# Separate the features (X) and target variable (y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

X.head()

X.info()

y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Selection
model = LinearRegression()

# Step 4: Model Training
m=model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)

# Example: Make predictions for a new set of data
new_data = {
    'symboling': [3],
    'fueltype': ['gas'],
    'aspiration': ['std'],
    'doornumber': ['two'],
    'carbody': ['convertible'],
    'drivewheel': ['rwd'],
    'enginelocation': ['front'],
    'wheelbase': [100.0],
    'carlength': [170.2],
    'carwidth': [68.2],
    'carheight': [50.1],
    'curbweight': [2700],
    'enginetype': ['dohc'],
    'cylindernumber': ['four'],
    'enginesize': [140],
    'fuelsystem': ['mpfi'],
    'boreratio': [3.5],
    'stroke': [2.8],
    'compressionratio': [10],
    'horsepower': [130],
    'peakrpm': [5500],
    'citympg': [25],
    'highwaympg': [32]
}

new_df = pd.DataFrame(new_data)

# Preprocess the new data
new_df_encoded = pd.get_dummies(new_df.reindex(columns=X.columns, fill_value=0))

# Make predictions on the new data
new_prediction = model.predict(new_df_encoded)

print('Predicted Price:', new_prediction)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree regressor
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("Root Mean Squared Error:", rmse)

