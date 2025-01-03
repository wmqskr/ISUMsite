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
# exchange your data
#df = pd.read_csv('./Data/Processed_Data/DR_SMOTE_ENN.csv')

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
    rfc = RandomForestClassifier(n_estimators=440, random_state=1)
    gbc1 = GradientBoostingClassifier(n_estimators=500, random_state=1)
    gbc2 = GradientBoostingClassifier(n_estimators=520, random_state=1)

    # Define ensemble model
    voting_classifier = VotingClassifier(estimators=[('rfc', rfc), ('gbc1', gbc1), ('gbc2', gbc2)], voting='soft')

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