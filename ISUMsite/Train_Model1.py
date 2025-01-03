# -*- coding: utf-8 -*-
import scipy.io
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle

print("Start training")

# Read CSV dataset
df = pd.read_csv('./Data/Processed_Data/DR_SMOTE_ENN.csv')

# Assume the last column is the label, and the rest are features
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

# Ten-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Initialize performance metric list
accuracies = []

# Ten-fold cross validation
for train_index, test_index in kf.split(features):
    # Split the training set and test set
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Define classifiers
    rfc1 = RandomForestClassifier(n_estimators=170, random_state=1)
    gbc = GradientBoostingClassifier(n_estimators=580, random_state=1)
    rfc2 = RandomForestClassifier(n_estimators=200, random_state=1)

    # Define ensemble model
    voting_classifier = VotingClassifier(estimators=[('rfc1', rfc1), ('rfc2', rfc2), ('gbc', gbc)], voting='soft')

    # Train ensemble model
    voting_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = voting_classifier.predict_proba(X_test)[:, 1]

    # Compute model accuracy
    accuracy = accuracy_score(y_test, predictions.round())
    accuracies.append(accuracy)
    print("Current ACC:", accuracy)

# Output performance metrics of ten-fold cross validation
print("Ten fold cross verification accuracy:", accuracies)
print("Average ACC:", np.mean(accuracies))

# Save the model to a file
with open("new_ensemble.pkl", "wb") as model_file:
    pickle.dump(voting_classifier, model_file)