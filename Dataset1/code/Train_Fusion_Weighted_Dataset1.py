import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
PROJECT_ROOT = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"
@dataclass(frozen=True)
class ModelParams:
    rfc1_n: int
    gbc_n: int
    rfc2_n: int
def build_voting(params: ModelParams, seed: int = 1) -> VotingClassifier:
    rfc1 = RandomForestClassifier(n_estimators=params.rfc1_n, random_state=seed, n_jobs=-1)
    gbc = GradientBoostingClassifier(n_estimators=params.gbc_n, random_state=seed)
    rfc2 = RandomForestClassifier(n_estimators=params.rfc2_n, random_state=seed, n_jobs=-1)
    return VotingClassifier(estimators=[("rfc1", rfc1), ("rfc2", rfc2), ("gbc", gbc)], voting="soft")
def calc_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = 0.0
    return {"ACC": acc, "SN": sn, "SP": sp, "MCC": mcc, "AUC": auc}
def load_aligned_dataset(base_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dir = os.path.join(base_dir, "Data", "Processed_Data")
    pglm_path = os.path.join(data_dir, "Change the file name and file path to what you need.csv")
    pos_dr_path = os.path.join(data_dir, "Change the file name and file path to what you need.csv")
    neg_dr_path = os.path.join(data_dir, "Change the file name and file path to what you need.csv")
    if not (os.path.exists(pglm_path) and os.path.exists(pos_dr_path) and os.path.exists(neg_dr_path)):
        raise FileNotFoundError(
            "Missing required files. Expected:\n"
            f"- {pglm_path}\n- {pos_dr_path}\n- {neg_dr_path}\n"
        )
    df_pglm = pd.read_csv(pglm_path)
    if "label" in df_pglm.columns:
        X_pglm = df_pglm.drop(columns=["label"]).values
        y = df_pglm["label"].values.astype(int)
    else:
        X_pglm = df_pglm.iloc[:, :-1].values
        y = df_pglm.iloc[:, -1].values.astype(int)
    df_pos = pd.read_csv(pos_dr_path)
    df_neg = pd.read_csv(neg_dr_path)
    X_dr = np.vstack([df_pos.iloc[:, :-1].values, df_neg.iloc[:, :-1].values])
    y_dr = np.hstack([np.ones(len(df_pos), dtype=int), np.zeros(len(df_neg), dtype=int)])
    if len(y) != len(y_dr):
        raise ValueError(f"Sample count mismatch: PGLM={len(y)} vs DR={len(y_dr)}")
    if not np.array_equal(y, y_dr):
        raise ValueError(
            "Label alignment mismatch between ProteinGLM and DR.\n"
            "This indicates the row order across feature sets is not aligned."
        )
    return X_dr, X_pglm, y
def load_aligned_balanced_dataset1(base_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dir = os.path.join(base_dir, "Data", "Processed_Data")
    dr_path = os.path.join(data_dir, "Change the file name and file path to what you need.csv")
    pglm_path = os.path.join(data_dir, "Change the file name and file path to what you need.csv")
    if not (os.path.exists(dr_path) and os.path.exists(pglm_path)):
        raise FileNotFoundError(
            "Balanced aligned files not found. Please run:\n"
            "  python Over_Under_sampling.py keepN\n"
            f"Expected:\n- {dr_path}\n- {pglm_path}\n"
        )
    df_dr = pd.read_csv(dr_path)
    df_pglm = pd.read_csv(pglm_path)
    if "label" not in df_dr.columns:
        df_dr.rename(columns={df_dr.columns[-1]: "label"}, inplace=True)
    if "label" not in df_pglm.columns:
        df_pglm.rename(columns={df_pglm.columns[-1]: "label"}, inplace=True)
    y = df_dr["label"].values.astype(int)
    if not np.array_equal(y, df_pglm["label"].values.astype(int)):
        raise ValueError("Balanced DR and ProteinGLM label columns do not match; files are not aligned.")
    X_dr = df_dr.drop(columns=["label"]).values
    X_pglm = df_pglm.drop(columns=["label"]).values
    return X_dr, X_pglm, y
def grid_search_weight(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, step: float = 0.001) -> Tuple[float, Dict[str, float]]:
    best_w = 0.5
    best = None
    ws = np.arange(0.0, 1.0 + 1e-12, step)
    for w in ws:
        p = w * p1 + (1.0 - w) * p2
        m = calc_metrics(y_true, p)
        key = (m["MCC"], m["ACC"], m["AUC"])
        if best is None or key > best:
            best = key
            best_w = float(w)
            best_metrics = m
    return best_w, best_metrics
def main():
    base_dir = PROJECT_ROOT
    dr_params = ModelParams(rfc1_n=261, gbc_n=522, rfc2_n=273)
    pglm_params = ModelParams(rfc1_n=331, gbc_n=770, rfc2_n=160)
    use_balanced = os.environ.get("FUSION_BALANCED", "0").strip() == "1"
    if use_balanced:
        X_dr, X_pglm, y = load_aligned_balanced_dataset1(base_dir)
        print(f"Balanced aligned dataset loaded. DR={X_dr.shape}, PGLM={X_pglm.shape}, y={y.shape}")
    else:
        X_dr, X_pglm, y = load_aligned_dataset(base_dir)
        print(f"Aligned dataset loaded. DR={X_dr.shape}, PGLM={X_pglm.shape}, y={y.shape}")
    out_log = os.path.join(base_dir, "Dataset1", "outputs", "Change the file name and file path to what you need.txt")
    out_model = os.path.join(base_dir, "Dataset1", "outputs", "Change the file name and file path to what you need.pkl")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    oof_p_dr = np.zeros(len(y), dtype=float)
    oof_p_pglm = np.zeros(len(y), dtype=float)
    sampler = None
    if not use_balanced:
        from imblearn.combine import SMOTEENN
        sampler = SMOTEENN(random_state=1)
    fold = 1
    for tr_idx, te_idx in skf.split(X_dr, y):
        print(f"Fold {fold}/10")
        X_dr_tr, X_dr_te = X_dr[tr_idx], X_dr[te_idx]
        X_pglm_tr, X_pglm_te = X_pglm[tr_idx], X_pglm[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if sampler is not None:
            X_dr_rs, y_dr_rs = sampler.fit_resample(X_dr_tr, y_tr)
            X_pglm_rs, y_pglm_rs = sampler.fit_resample(X_pglm_tr, y_tr)
        else:
            X_dr_rs, y_dr_rs = X_dr_tr, y_tr
            X_pglm_rs, y_pglm_rs = X_pglm_tr, y_tr
        m_dr = build_voting(dr_params)
        m_pglm = build_voting(pglm_params)
        m_dr.fit(X_dr_rs, y_dr_rs)
        m_pglm.fit(X_pglm_rs, y_pglm_rs)
        oof_p_dr[te_idx] = m_dr.predict_proba(X_dr_te)[:, 1]
        oof_p_pglm[te_idx] = m_pglm.predict_proba(X_pglm_te)[:, 1]
        m1 = calc_metrics(y_te, oof_p_dr[te_idx])
        m2 = calc_metrics(y_te, oof_p_pglm[te_idx])
        print(f"  DR   ACC={m1['ACC']:.4f} MCC={m1['MCC']:.4f} AUC={m1['AUC']:.4f}")
        print(f"  PGLM ACC={m2['ACC']:.4f} MCC={m2['MCC']:.4f} AUC={m2['AUC']:.4f}")
        fold += 1
    best_w, best_fusion_metrics = grid_search_weight(y, oof_p_dr, oof_p_pglm, step=0.001)
    w_dr = best_w
    w_pglm = 1.0 - best_w
    p_fused = w_dr * oof_p_dr + w_pglm * oof_p_pglm
    m_dr_all = calc_metrics(y, oof_p_dr)
    m_pglm_all = calc_metrics(y, oof_p_pglm)
    m_fused_all = calc_metrics(y, p_fused)
    print("\n" + "=" * 60)
    print(f"Best fusion weights: DR={w_dr:.4f}, PGLM={w_pglm:.4f}")
    print("=" * 60)
    print(f"{'Metric':<8} {'DR':>10} {'PGLM':>10} {'Fusion':>10}")
    for k in ["ACC", "SN", "SP", "MCC", "AUC"]:
        print(f"{k:<8} {m_dr_all[k]:>10.4f} {m_pglm_all[k]:>10.4f} {m_fused_all[k]:>10.4f}")
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(f"Best fusion weights: DR={w_dr:.4f}, PGLM={w_pglm:.4f}\n\n")
        f.write(f"{'Metric':<8} {'DR':>10} {'PGLM':>10} {'Fusion':>10}\n")
        for k in ["ACC", "SN", "SP", "MCC", "AUC"]:
            f.write(f"{k:<8} {m_dr_all[k]:>10.4f} {m_pglm_all[k]:>10.4f} {m_fused_all[k]:>10.4f}\n")
    print(f"\nSaved fusion CV results to: {out_log}")
    if sampler is not None:
        X_dr_rs, y_rs = sampler.fit_resample(X_dr, y)
        X_pglm_rs, _ = sampler.fit_resample(X_pglm, y)
    else:
        X_dr_rs, y_rs = X_dr, y
        X_pglm_rs = X_pglm
    final_dr = build_voting(dr_params).fit(X_dr_rs, y_rs)
    final_pglm = build_voting(pglm_params).fit(X_pglm_rs, y_rs)
    bundle = {
        "dr_params": dr_params,
        "pglm_params": pglm_params,
        "weight_dr": w_dr,
        "weight_pglm": w_pglm,
        "dr_model": final_dr,
        "pglm_model": final_pglm,
        "note": "Predict proba with both models, then p = w_dr*p_dr + w_pglm*p_pglm, threshold=0.5",
    }
    with open(out_model, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved fusion model bundle to: {out_model}")
if __name__ == "__main__":
    main()
