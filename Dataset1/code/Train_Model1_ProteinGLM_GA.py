import pandas as pd
import numpy as np
import random
import pickle
import sys
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score, classification_report
from deap import base, creator, tools, algorithms
PROJECT_ROOT = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"
DATA_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'Change the file name and file path to what you need.csv')
MODEL_FILE = os.path.join(PROJECT_ROOT, "Dataset1", "outputs", "Change the file name and file path to what you need.pkl")
RESULT_LOG_PATH = os.path.join(PROJECT_ROOT, "Dataset1", "outputs", "Change the file name and file path to what you need.txt")
print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Please replace the placeholder path in DATA_FILE: {DATA_FILE}")
    raise
if 'label' in df.columns:
    X = df.drop('label', axis=1).values
    y = df['label'].values
else:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
print(f"Data shape: {X.shape}")
if "FitnessMax" in creator.__dict__:
    del creator.FitnessMax
if "Individual" in creator.__dict__:
    del creator.Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_rfc1", random.randint, 100, 300)
toolbox.register("attr_gbc", random.randint, 300, 800)
toolbox.register("attr_rfc2", random.randint, 100, 300)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_rfc1, toolbox.attr_gbc, toolbox.attr_rfc2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
def evaluate(individual):
    n1, n2, n3 = individual
    if n1 < 10: n1 = 10
    if n2 < 10: n2 = 10
    if n3 < 10: n3 = 10
    rfc1 = RandomForestClassifier(n_estimators=int(n1), random_state=1, n_jobs=-1)
    gbc = GradientBoostingClassifier(n_estimators=int(n2), random_state=1)
    rfc2 = RandomForestClassifier(n_estimators=int(n3), random_state=1, n_jobs=-1)
    eclf = VotingClassifier(estimators=[('rfc1', rfc1), ('rfc2', rfc2), ('gbc', gbc)], voting='soft')
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    scores = cross_val_score(eclf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return (scores.mean(),)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[50, 100, 50], up=[400, 1000, 400], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
def main():
    print("Starting Genetic Algorithm Optimization for ProteinGLM Model...")
    POP_SIZE = 10
    N_GEN = 5
    CXPB = 0.5
    MUTPB = 0.2
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GEN,
                                   stats=stats, halloffame=hof, verbose=True)
    best_ind = hof[0]
    print(f"Best Individual (n_estimators): RFC1={best_ind[0]}, GBC={best_ind[1]}, RFC2={best_ind[2]}")
    print(f"Best Training CV Score (3-fold): {best_ind.fitness.values[0]:.4f}")
    print("\nRunning Final 10-Fold Validation with Best Parameters...")
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    cv_accuracies = []
    cv_sns = []
    cv_sps = []
    cv_mccs = []
    cv_aucs = []
    cv_results_str = []
    rfc1 = RandomForestClassifier(n_estimators=best_ind[0], random_state=1, n_jobs=-1)
    gbc = GradientBoostingClassifier(n_estimators=best_ind[1], random_state=1)
    rfc2 = RandomForestClassifier(n_estimators=best_ind[2], random_state=1, n_jobs=-1)
    voting_classifier = VotingClassifier(estimators=[('rfc1', rfc1), ('rfc2', rfc2), ('gbc', gbc)], voting='soft')
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        voting_classifier.fit(X_train, y_train)
        predictions_proba = voting_classifier.predict_proba(X_test)[:, 1]
        predictions = predictions_proba.round()
        accuracy = accuracy_score(y_test, predictions)
        cv_accuracies.append(accuracy)
        res_str = f"Fold {fold} Accuracy: {accuracy:.4f}"
        print(res_str)
        cv_results_str.append(res_str)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_test, predictions)
        try:
            auc = roc_auc_score(y_test, predictions_proba)
        except ValueError:
            auc = 0.0
        print(f"  SN: {sn:.4f}, SP: {sp:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
        cv_sns.append(sn)
        cv_sps.append(sp)
        cv_mccs.append(mcc)
        cv_aucs.append(auc)
        fold += 1
    avg_cv_acc = np.mean(cv_accuracies)
    avg_cv_sn = np.mean(cv_sns) if len(cv_sns) else 0.0
    avg_cv_sp = np.mean(cv_sps) if len(cv_sps) else 0.0
    avg_cv_mcc = np.mean(cv_mccs) if len(cv_mccs) else 0.0
    avg_cv_auc = np.mean(cv_aucs) if len(cv_aucs) else 0.0
    print("\n" + "="*30)
    print(f"Average 10-Fold CV Accuracy: {avg_cv_acc:.4f}")
    print(f"Average 10-Fold CV SN: {avg_cv_sn:.4f}")
    print(f"Average 10-Fold CV SP: {avg_cv_sp:.4f}")
    print(f"Average 10-Fold CV MCC: {avg_cv_mcc:.4f}")
    print(f"Average 10-Fold CV AUC: {avg_cv_auc:.4f}")
    print("="*30 + "\n")
    with open(RESULT_LOG_PATH, "w") as f:
        f.write("\n".join(cv_results_str))
        f.write(f"\n\nAverage 10-Fold CV Accuracy: {avg_cv_acc:.4f}\n")
        f.write(f"Average 10-Fold CV SN: {avg_cv_sn:.4f}\n")
        f.write(f"Average 10-Fold CV SP: {avg_cv_sp:.4f}\n")
        f.write(f"Average 10-Fold CV MCC: {avg_cv_mcc:.4f}\n")
        f.write(f"Average 10-Fold CV AUC: {avg_cv_auc:.4f}\n")
    print(f"10-Fold CV results saved to {RESULT_LOG_PATH}")
    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump(voting_classifier, model_file)
    print(f"Model saved to {MODEL_FILE}")
if __name__ == "__main__":
    main()
