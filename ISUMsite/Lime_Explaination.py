import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Read the CSV dataset
df = pd.read_csv('./Data/Processed_Data/DR_SMOTE_ENN.csv')

# Assume the last column is the label, and the remaining columns are features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Ten-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Store the results
sn_list = []
sp_list = []
acc_list = []
mcc_list = []
auc_list = []

# Initialize the LIME explainer
explainer = LimeTabularExplainer(X, mode='classification', feature_names=df.columns[:-1], class_names=['0', '1'])

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Define classifiers
    rfc = RandomForestClassifier(n_estimators=170, random_state=1)
    gbc = GradientBoostingClassifier(n_estimators=580, random_state=1)
    rfc1 = RandomForestClassifier(n_estimators=200, random_state=1)

    # Create a voting ensemble classifier
    ensemble = VotingClassifier(estimators=[('rfc', rfc), ('gbc', gbc), ('rfc1', rfc1)], voting='soft')

    # Train the ensemble classifier
    ensemble.fit(X_train, y_train)

    # Predict probabilities on the test set
    ensemble_probabilities = ensemble.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    tn, fp, fn, tp = confusion_matrix(y_test, ensemble_probabilities.round()).ravel()
    sn = tp / (tp + fn)  # Sensitivity or Recall
    sp = tn / (tn + fp)  # Specificity
    acc = accuracy_score(y_test, ensemble_probabilities.round())
    mcc = matthews_corrcoef(y_test, ensemble_probabilities.round())
    auc = roc_auc_score(y_test, ensemble_probabilities)

    # Append the results to the lists
    sn_list.append(sn)
    sp_list.append(sp)
    acc_list.append(acc)
    mcc_list.append(mcc)
    auc_list.append(auc)

    print(f"sn:{sn}, sp:{sp}, acc:{acc}, mcc:{mcc}, auc:{auc}")

    # Choose a sample for explanation
    sample_index = 0  # Choose the first sample
    exp = explainer.explain_instance(X_test[sample_index], ensemble.predict_proba, num_features=len(df.columns[:-1]))
    exp.show_in_notebook(show_table=True, show_all=False)
    # Save the explanation results to an HTML file
    exp.save_to_file(f'lime_explanation_fold_{len(sn_list)}.html')

# Calculate the mean and standard deviation
print(f'Sensitivity (SN): {np.mean(sn_list):.4f} ± {np.std(sn_list):.4f}')
print(f'Specificity (SP): {np.mean(sp_list):.4f} ± {np.std(sp_list):.4f}')
print(f'Accuracy (ACC): {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}')
print(f'Matthews Correlation Coefficient (MCC): {np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f}')
print(f'Area Under Curve (AUC): {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}')