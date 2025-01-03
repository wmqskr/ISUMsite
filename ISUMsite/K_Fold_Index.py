# -*- coding: utf-8 -*-
# Generate cross-validation indices for the given dataset
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold

# Generate cross-validation indices
data = pd.read_csv('./Data/Processed_Data/DR_SMOTE_ENN.csv')

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Count the number of positive and negative samples
numPositive_synthetic = np.sum(y == 1)
numNegative_synthetic = np.sum(y == 0)

# Output the number of positive and negative samples
print('Positive samples:', numPositive_synthetic)
print('Negative samples:', numNegative_synthetic)

# Perform ten-fold cross-validation
k = 10  # Number of folds
n = X.shape[0]  # Number of samples
kf = KFold(n_splits=k, shuffle=True)  # Generate cross-validation indices

# Create feature + label + index matrix
data_with_indices = np.column_stack((X, y, np.zeros(n)))

# Fill in the cross-validation indices
for i, (_, test_index) in enumerate(kf.split(X)):
    data_with_indices[test_index, -1] = i + 1

# Save the feature + label + index matrix
sio.savemat('./Data/Processed_Data/DR_SMOTE_ENN.mat', {'data_with_indices': data_with_indices})