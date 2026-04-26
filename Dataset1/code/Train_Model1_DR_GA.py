import pandas as pd
import numpy as np
import random
import pickle
import sys
import os
import time
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score, classification_report
from deap import base, creator, tools, algorithms
PROJECT_ROOT = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"
DATA_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'Change the file name and file path to what you need.csv')
MODEL_FILE = os.path.join(PROJECT_ROOT, "Dataset1", "outputs", "Change the file name and file path to what you need.pkl")
RESULT_FILE = os.path.join(PROJECT_ROOT, "Dataset1", "outputs", "Change the file name and file path to what you need.txt")
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
    rfc1 = XGBRFClassifier(
        n_estimators=int(n1),
        learning_rate=1,
        subsample=0.8,
        colsample_bynode=0.8,
        device='cuda',
        tree_method='hist',
        random_state=1,
        n_jobs=1
    )
    gbc = XGBClassifier(
        n_estimators=int(n2),
        device='cuda',
        tree_method='hist',
        random_state=1,
        eval_metric='logloss'
    )
    rfc2 = XGBRFClassifier(
        n_estimators=int(n3),
        learning_rate=1,
        subsample=0.8,
        colsample_bynode=0.8,
        device='cuda',
        tree_method='hist',
        random_state=1,
        n_jobs=1
    )
    eclf = VotingClassifier(estimators=[('rfc1', rfc1), ('rfc2', rfc2), ('gbc', gbc)], voting='soft')
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    scores = cross_val_score(eclf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return (scores.mean(),)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[50, 100, 50], up=[400, 1000, 400], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
def main():
    print("Starting Genetic Algorithm Optimization (GPU Enabled)...")
    start_time = time.time()
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
    print(f"\nBest Individual (n_estimators): RFC1={best_ind[0]}, GBC={best_ind[1]}, RFC2={best_ind[2]}")
    print(f"Best Training CV Score (3-fold): {best_ind.fitness.values[0]:.4f}")
    print("\nRunning Final 10-Fold Validation with Best Parameters...")
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    accuracies = []
    sensitivities = []
    specificities = []
    mccs = []
    aucs = []
    results_log = []
    rfc1 = XGBRFClassifier(
        n_estimators=best_ind[0],
        learning_rate=1,
        subsample=0.8,
        colsample_bynode=0.8,
        device='cuda',
        tree_method='hist',
        random_state=1,
        n_jobs=1
    )
    gbc = XGBClassifier(
        n_estimators=best_ind[1],
        device='cuda',
        tree_method='hist',
        random_state=1,
        eval_metric='logloss'
    )
    rfc2 = XGBRFClassifier(
        n_estimators=best_ind[2],
        learning_rate=1,
        subsample=0.8,
        colsample_bynode=0.8,
        device='cuda',
        tree_method='hist',
        random_state=1,
        n_jobs=1
    )
    voting_classifier = VotingClassifier(estimators=[('rfc1', rfc1), ('rfc2', rfc2), ('gbc', gbc)], voting='soft')
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        voting_classifier.fit(X_train, y_train)
        y_pred_proba = voting_classifier.predict_proba(X_test)[:, 1]
        y_pred = y_pred_proba.round()
        acc = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = 0.0
        accuracies.append(acc)
        sensitivities.append(sn)
        specificities.append(sp)
        mccs.append(mcc)
        aucs.append(auc)
        log_str = f"Fold {fold} - ACC: {acc:.4f}, SN: {sn:.4f}, SP: {sp:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}"
        print(log_str)
        results_log.append(log_str)
        fold += 1
    avg_acc = np.mean(accuracies)
    avg_sn = np.mean(sensitivities)
    avg_sp = np.mean(specificities)
    avg_mcc = np.mean(mccs)
    avg_auc = np.mean(aucs)
    summary_str = f"\nFinal 10-Fold Average Metrics (GPU):\n"
    summary_str += f"ACC: {avg_acc:.4f}\n"
    summary_str += f"SN: {avg_sn:.4f}\n"
    summary_str += f"SP: {avg_sp:.4f}\n"
    summary_str += f"MCC: {avg_mcc:.4f}\n"
    summary_str += f"AUC: {avg_auc:.4f}\n"
    print(summary_str)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total Execution Time: {duration:.2f} seconds")
    with open(RESULT_FILE, "w") as f:
        f.write("Genetic Algorithm Optimization Results (GPU Enabled)\n")
        f.write("================================================\n")
        f.write(f"Best Parameters found:\n")
        f.write(f"  RFC1 n_estimators: {best_ind[0]}\n")
        f.write(f"  GBC n_estimators: {best_ind[1]}\n")
        f.write(f"  RFC2 n_estimators: {best_ind[2]}\n")
        f.write("================================================\n\n")
        f.write("\n".join(results_log))
        f.write("\n" + summary_str)
        f.write(f"\nTotal Execution Time: {duration:.2f} seconds\n")
    print(f"Results saved to {RESULT_FILE}")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(voting_classifier, f)
    print(f"Model saved to {MODEL_FILE}")
if __name__ == "__main__":
    main()
