# -*- coding: utf-8 -*-
"""Task 1.ipynb

Load the dataset
"""

import pandas as pd
import numpy as np
# Load the dataset
iris_data = pd.read_csv('/content/Iris.csv')

# Display the first few rows of the dataset
print(iris_data.head())

# Split the dataset into features and target variable
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']

"""Feature variables"""

X.head()

"""Target Variable"""

y.head()

"""Train Test Split"""

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Decision Tree classifier"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a Decision Tree classifier
classifier = DecisionTreeClassifier()

# Train the model
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

"""Logistic Regression """

from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression classifier
model = LogisticRegression(solver='liblinear', max_iter=1000)  # Increase max_iter
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

"""K- Nearest Neighbor (KNN)"""

from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier
clf = KNeighborsClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

"""Support Vector Machines (SVM)"""

from sklearn.svm import SVC

# Create an SVM classifier
clf = SVC()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

"""Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert the dataset into a Pandas DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Visualize the data using pair plots
sns.pairplot(df, hue='species')
plt.show()

