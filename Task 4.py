# -*- coding: utf-8 -*-
"""Task 4.ipynb"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('/content/spam.csv', encoding='latin-1')

data.head()

data.info()

# Preprocessing
data['v2'] = data['v2'].str.lower()  # Convert text to lowercase
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})  # Map 'ham' to 0 and 'spam' to 1

data['v2'].head()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)

"""TF-IDF"""

from sklearn.feature_extraction.text import TfidfVectorizer
# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Selection and Training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on the testing set
y_pred = model.predict(X_test_tfidf)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

"""Count Vectorization"""

from sklearn.feature_extraction.text import CountVectorizer
# Feature Extraction using Count Vectorization
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Model Selection and Training
model = MultinomialNB()
model.fit(X_train_count, y_train)

# Predict on the testing set
y_pred = model.predict(X_test_count)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

"""N-gram Features using TF-IDF"""

# Feature Extraction using TF-IDF with N-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Selection and Training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on the testing set
y_pred = model.predict(X_test_tfidf)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

