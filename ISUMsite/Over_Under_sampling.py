# -- coding: UTF-8 --
import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NearMiss


# Gaussian Oversampling
def Gaussian_Oversample(input_file, output_file):
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    positive_count = sum(y == 1)
    negative_count = sum(y == 0)
    target_count = (positive_count + negative_count) // 2

    np.random.seed(0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    num_samples_to_generate = target_count - positive_count
    synthetic_samples = pd.DataFrame(np.random.normal(loc=mean, scale=std, size=(num_samples_to_generate, X.shape[1])), columns=X.columns)
    synthetic_labels = pd.Series([1] * num_samples_to_generate)

    balanced_X = pd.concat([X, synthetic_samples], axis=0)
    balanced_y = pd.concat([y, synthetic_labels], axis=0)

    balanced_data = pd.concat([balanced_X, balanced_y], axis=1)
    balanced_data.to_csv(output_file, index=False)



#ADASYN Oversampling
def ADASYN_Oversample(input_file, output_file):
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    positive_count = sum(y == 1)
    negative_count = sum(y == 0)
    target_count = (positive_count + negative_count) // 2

    adasyn = ADASYN(sampling_strategy={1: target_count})
    balanced_X, balanced_y = adasyn.fit_resample(X, y)

    balanced_data = pd.concat([balanced_X, balanced_y], axis=1)
    balanced_data.to_csv(output_file, index=False)



#Borderline Oversampling
def Borderline_Oversample(input_file, output_file):
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    positive_count = sum(y == 1)
    negative_count = sum(y == 0)
    target_count = (positive_count + negative_count) // 2

    smote = BorderlineSMOTE(sampling_strategy={1: target_count})
    balanced_X, balanced_y = smote.fit_resample(X, y)

    balanced_data = pd.concat([balanced_X, balanced_y], axis=1)
    balanced_data.to_csv(output_file, index=False)



#SMOTE Oversampling
def SMOTE_Oversample(input_file, output_file):
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    print("Number of samples before sampling:")
    positive_count = sum(y == 1)
    negative_count = sum(y == 0)
    print("Number of positive samples:" + str(positive_count))
    print("Number of negative samples:" + str(negative_count))
    target_count = (positive_count + negative_count) // 2

    smote = SMOTE(sampling_strategy={1: target_count})
    balanced_X, balanced_y = smote.fit_resample(X, y)

    balanced_data = pd.concat([balanced_X, balanced_y], axis=1)
    balanced_data.to_csv(output_file, index=False)



#ENN Undersampling Edited Nearest Neighbors
def ENN_Undersample(input_file, output_file):
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=3)
    X_resampled, y_resampled = enn.fit_resample(X, y)

    print("Number of positive samples after undersampling:", sum(y_resampled == 1))
    print("Number of negative samples after undersampling:", sum(y_resampled == 0))

    resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['label'])], axis=1)
    resampled_data.to_csv(output_file, index=False)



#Near Miss Undersampling
def Near_Undersample(input_file, output_file):
    data = pd.read_csv(input_file)
    X = data.drop('label', axis=1)
    y = data['label']

    nm = NearMiss()
    X_resampled, y_resampled = nm.fit_resample(X, y)

    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
    resampled_data.to_csv(output_file, index=False)