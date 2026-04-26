
import json
import os
import pickle
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform, uniform
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import contextlib
import joblib

PROJECT_ROOT = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"


@contextlib.contextmanager
def progress_logger(total):
    
    import sys
    
    class ProgressWriter:
        def __init__(self, original_stdout):
            self.original = original_stdout
            self.count = 0
            self.total = total
        
        def write(self, text):
           
            if "[CV] END" in text:
                self.count += 1
                
                text = text.rstrip() + f"  [Progress: {self.count}/{self.total}]\n"
            self.original.write(text)
        
        def flush(self):
            self.original.flush()
    
    old_stdout = sys.stdout
    sys.stdout = ProgressWriter(old_stdout)
    try:
        yield
    finally:
        sys.stdout = old_stdout


def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)
    return X, y


def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, pred)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = 0.0
    return {"ACC": acc, "SN": sn, "SP": sp, "MCC": mcc, "AUC": auc}


def make_base(seed: int = 1) -> VotingClassifier:
    
    rfc = RandomForestClassifier(
        n_estimators=440,
        random_state=seed,
        n_jobs=-1  
    )
    

    gbc1 = XGBClassifier(
        n_estimators=500,
        tree_method='hist',
        device='cuda',
        random_state=seed,
        eval_metric='logloss'
    )
    
  
    gbc2 = XGBClassifier(
        n_estimators=520,
        tree_method='hist',
        device='cuda',
        random_state=seed,
        eval_metric='logloss'
    )
    
    return VotingClassifier(
        estimators=[("rfc", rfc), ("gbc1", gbc1), ("gbc2", gbc2)],
        voting="soft"
    )


def main():
    base_dir = PROJECT_ROOT
    tag = os.environ.get("PGLM_D2_FEATURE_TAG", "NearMiss").strip()
    date_tag = datetime.now().strftime("%Y%m%d")

    train_candidates = [
        os.path.join(base_dir, "Data", "Processed_Data", f"ProteinGLM_Dataset2_Train_Features_{tag}.csv"),
        os.path.join(base_dir, "Dataset2", "intermediate", f"ProteinGLM_Dataset2_Train_Features_{tag}.csv"),
    ]
    test_candidates = [
        os.path.join(base_dir, "Data", "Processed_Data", f"ProteinGLM_Dataset2_Test_Features_{tag}.csv"),
        os.path.join(base_dir, "Dataset2", "intermediate", f"ProteinGLM_Dataset2_Test_Features_{tag}.csv"),
    ]

    train_path = next((p for p in train_candidates if os.path.exists(p)), None)
    test_path = next((p for p in test_candidates if os.path.exists(p)), None)
    if train_path is None:
        raise FileNotFoundError(f"Missing train CSV. Tried: {train_candidates}")
    if test_path is None:
        raise FileNotFoundError(f"Missing test CSV. Tried: {test_candidates}")

    X, y = load_xy(train_path)
    X_test, y_test = load_xy(test_path)
    print(f"Train: {X.shape}, labels={np.bincount(y)}")
    print(f"Test:  {X_test.shape}, labels={np.bincount(y_test)}")

    n_iter = int(os.environ.get("RS_N_ITER", "60"))
    cv_folds = int(os.environ.get("RS_CV_FOLDS", "5"))
    seed = int(os.environ.get("RS_SEED", "1"))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    base = make_base(seed=seed)

   
    param_dist = {
      
        "rfc__n_estimators": randint(200, 800),
        "rfc__max_depth": randint(10, 50),
        "rfc__min_samples_split": randint(2, 10),
        "rfc__min_samples_leaf": randint(1, 5),
        
        "gbc1__n_estimators": randint(300, 800),
        "gbc1__learning_rate": loguniform(0.01, 0.3),
        "gbc1__max_depth": randint(3, 12),
        "gbc1__subsample": uniform(0.6, 0.4),
        "gbc1__colsample_bytree": uniform(0.6, 0.4),
        
        "gbc2__n_estimators": randint(300, 800),
        "gbc2__learning_rate": loguniform(0.01, 0.3),
        "gbc2__max_depth": randint(3, 12),
        "gbc2__subsample": uniform(0.6, 0.4),
        "gbc2__colsample_bytree": uniform(0.6, 0.4),
    }

    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,  
        random_state=seed,
        verbose=2,
        refit=True,
        return_train_score=False,
    )

    print(f"RandomizedSearchCV (GPU): scoring=ACC n_iter={n_iter} cv_folds={cv_folds}")
    
    
    with progress_logger(total=n_iter * cv_folds):
        rs.fit(X, y)

    best = rs.best_estimator_
    best_acc = float(rs.best_score_)
    best_params = rs.best_params_
    print(f"Best CV ACC: {best_acc:.4f}")
    print(f"Best params: {best_params}")

    test_proba = best.predict_proba(X_test)[:, 1]
    test_m = metrics_from_proba(y_test, test_proba)
    print("Test metrics:", test_m)

    out_txt = os.path.join(base_dir, "Dataset2", "outputs", f"CV_Results_Model2_ProteinGLM_Dataset2_{tag}_{date_tag}.txt")
    out_pkl = os.path.join(base_dir, "Dataset2", "outputs", f"new_model_ProteinGLM_Dataset2_{tag}_{date_tag}.pkl")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("ProteinGLM Dataset2 - ACC-first tuning (RandomizedSearchCV + RFC + XGBoost GPU)\n")
        f.write(f"Date: {date_tag}\n")
        f.write(f"Feature tag: {tag}\n")
        f.write(f"Train: {train_path}\n")
        f.write(f"Test:  {test_path}\n")
        f.write(f"RS n_iter={n_iter} cv_folds={cv_folds} seed={seed} scoring=accuracy\n\n")
        f.write(f"Best CV ACC: {best_acc:.6f}\n")
        f.write("Best params:\n")
        f.write(json.dumps(best_params, indent=2) + "\n\n")
        f.write("Test metrics (threshold=0.5):\n")
        for k in ["ACC", "AUC", "SN", "SP", "MCC"]:
            f.write(f"{k}: {test_m[k]:.6f}\n")

    bundle = {
        "date": date_tag,
        "feature_tag": tag,
        "best_cv_acc": best_acc,
        "best_params": best_params,
        "model": best,
        "note": "VotingClassifier soft: RFC(CPU) + GBC1(XGB GPU) + GBC2(XGB GPU) tuned for accuracy.",
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved results to: {out_txt}")
    print(f"Saved model to: {out_pkl}")


if __name__ == "__main__":
    main()
