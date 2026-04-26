import os
import pickle
import glob
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score
from lime.lime_tabular import LimeTabularExplainer
from imblearn.combine import SMOTEENN
from dataclasses import dataclass
from typing import Tuple, Dict


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

def load_aligned_balanced_dataset1(project_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    data_dir = os.path.join(project_root, "Dataset1", "intermediate")
    dr_path = os.path.join(data_dir, "Change the file name and file path to what you need.csv")
    pglm_path = os.path.join(data_dir, "Change the file name and file path to what you need.csv")
    if not (os.path.exists(dr_path) and os.path.exists(pglm_path)):
        raise FileNotFoundError(
            "Balanced aligned files not found.\n"
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
    df_dr_features = df_dr.drop(columns=["label"])
    df_dr_features.columns = [f"DR_{i}" for i in range(df_dr_features.shape[1])]
    df_pglm_features = df_pglm.drop(columns=["label"])
    df_pglm_features.columns = [f"PGLM_{col}" for col in df_pglm_features.columns]
    return X_dr, X_pglm, y, df_dr_features, df_pglm_features


def create_fusion_predict_proba(dr_model, pglm_model, weight_dr: float, weight_pglm: float,
                                dr_dim: int, pglm_dim: int):
    def predict_proba(concatenated_features):
        if concatenated_features.ndim == 1:
            concatenated_features = concatenated_features.reshape(1, -1)
        X_dr = concatenated_features[:, :dr_dim]
        X_pglm = concatenated_features[:, dr_dim:]
        prob_dr = dr_model.predict_proba(X_dr)[:, 1]
        prob_pglm = pglm_model.predict_proba(X_pglm)[:, 1]
        prob_fused = weight_dr * prob_dr + weight_pglm * prob_pglm
        prob_class_0 = 1.0 - prob_fused
        prob_class_1 = prob_fused
        return np.column_stack([prob_class_0, prob_class_1])
    return predict_proba


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


def main():
    base_dir = os.path.join(PROJECT_ROOT, "Interpretability")
    project_root = PROJECT_ROOT
    model_dir = os.path.join(project_root, "Dataset1", "outputs")
    dr_params = ModelParams(rfc1_n=261, gbc_n=522, rfc2_n=273)
    pglm_params = ModelParams(rfc1_n=331, gbc_n=770, rfc2_n=160)
    preferred_model = "Change the file name and file path to what you need.pkl"
    fusion_model_path = os.path.join(model_dir, preferred_model)
    if not os.path.exists(fusion_model_path):
        fusion_pattern = os.path.join(model_dir, "Change the file name and file path to what you need.pkl")
        fusion_files = glob.glob(fusion_pattern)
        if fusion_files:
            fusion_model_path = max(fusion_files, key=os.path.getmtime)
            print(f"Preferred model not found, using: {os.path.basename(fusion_model_path)}")
    use_saved_model = os.path.exists(fusion_model_path)
    if use_saved_model:
        print(f"Loading saved fusion model from: {fusion_model_path}")
        with open(fusion_model_path, "rb") as f:
            bundle = pickle.load(f)
        dr_model = bundle["dr_model"]
        pglm_model = bundle["pglm_model"]
        weight_dr = bundle["weight_dr"]
        weight_pglm = bundle["weight_pglm"]
        print(f"Loaded fusion model with weights: DR={weight_dr:.4f}, PGLM={weight_pglm:.4f}")
    else:
        print("Saved fusion model not found. Will train models during CV.")
        weight_dr = None
        weight_pglm = None
        dr_model = None
        pglm_model = None
    print("Loading balanced aligned dataset...")
    try:
        X_dr, X_pglm, y, df_dr, df_pglm_features = load_aligned_balanced_dataset1(project_root)
        print(f"Balanced dataset loaded. DR={X_dr.shape}, PGLM={X_pglm.shape}, y={y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Falling back to unbalanced dataset...")
        raise FileNotFoundError(
            "Please ensure balanced dataset files exist. "
            "The fusion model was trained on balanced data, so LIME analysis should use the same balanced data."
        )
    X_concatenated = np.hstack([X_dr, X_pglm])
    feature_names = list(df_dr.columns) + list(df_pglm_features.columns)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    sn_list = []
    sp_list = []
    acc_list = []
    mcc_list = []
    auc_list = []
    fold = 1

    for train_idx, test_idx in skf.split(X_concatenated, y):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/10")
        print(f"{'='*60}")
        X_train_concat = X_concatenated[train_idx]
        X_test_concat = X_concatenated[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_dr_train = X_train_concat[:, :X_dr.shape[1]]
        X_pglm_train = X_train_concat[:, X_dr.shape[1]:]
        X_dr_test = X_test_concat[:, :X_dr.shape[1]]
        X_pglm_test = X_test_concat[:, X_dr.shape[1]:]
        if use_saved_model:
            fold_dr_model = dr_model
            fold_pglm_model = pglm_model
            fold_weight_dr = weight_dr
            fold_weight_pglm = weight_pglm
        else:
            sampler = SMOTEENN(random_state=1)
            X_dr_res, y_dr_res = sampler.fit_resample(X_dr_train, y_train)
            X_pglm_res, y_pglm_res = sampler.fit_resample(X_pglm_train, y_train)
            fold_dr_model = build_voting(dr_params)
            fold_pglm_model = build_voting(pglm_params)
            fold_dr_model.fit(X_dr_res, y_dr_res)
            fold_pglm_model.fit(X_pglm_res, y_pglm_res)
            p_dr_train = fold_dr_model.predict_proba(X_dr_train)[:, 1]
            p_pglm_train = fold_pglm_model.predict_proba(X_pglm_train)[:, 1]
            best_score = -1
            best_w = 0.5
            for w in np.arange(0.0, 1.01, 0.01):
                p_fused = w * p_dr_train + (1 - w) * p_pglm_train
                m = calc_metrics(y_train, p_fused)
                score = m["MCC"] + m["ACC"]
                if score > best_score:
                    best_score = score
                    best_w = w
            fold_weight_dr = best_w
            fold_weight_pglm = 1.0 - best_w
            print(f"  Optimized weights: DR={fold_weight_dr:.4f}, PGLM={fold_weight_pglm:.4f}")
        fusion_predict_proba = create_fusion_predict_proba(
            fold_dr_model, fold_pglm_model, fold_weight_dr, fold_weight_pglm,
            X_dr.shape[1], X_pglm.shape[1]
        )
        fusion_probabilities = fusion_predict_proba(X_test_concat)[:, 1]
        metrics = calc_metrics(y_test, fusion_probabilities)
        sn = metrics["SN"]
        sp = metrics["SP"]
        acc = metrics["ACC"]
        mcc = metrics["MCC"]
        auc = metrics["AUC"]
        sn_list.append(sn)
        sp_list.append(sp)
        acc_list.append(acc)
        mcc_list.append(mcc)
        auc_list.append(auc)
        print(f"  SN: {sn:.4f}, SP: {sp:.4f}, ACC: {acc:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
        if fold == 1:
            print(f"\n  Performing LIME explanation for Fold {fold}...")
            explainer = LimeTabularExplainer(
                X_train_concat,
                mode='classification',
                feature_names=feature_names,
                class_names=['Negative', 'Positive'],
                discretize_continuous=True
            )
            sample_index = 0
            sample_instance = X_test_concat[sample_index]
            sample_label = y_test[sample_index]
            print(f"  Explaining sample {sample_index} (True label: {sample_label})...")
            num_features_to_show = min(50, len(feature_names))
            exp = explainer.explain_instance(
                sample_instance,
                fusion_predict_proba,
                num_features=num_features_to_show,
                top_labels=1
            )
            output_file = os.path.join(base_dir, 'Change the file name and file path to what you need.html')
            exp.save_to_file(output_file)
            print(f"  LIME explanation saved to: {output_file}")
            pred_proba = fusion_predict_proba(sample_instance.reshape(1, -1))[0]
            pred_label = np.argmax(pred_proba)
            print(f"  Predicted label: {pred_label} (probabilities: Negative={pred_proba[0]:.4f}, Positive={pred_proba[1]:.4f})")
            try:
                explanation_list = exp.as_list(label=pred_label)
            except KeyError:
                try:
                    explanation_list = exp.as_list(label=sample_label)
                    print(f"  Warning: Using true label {sample_label} for explanation (predicted label {pred_label} not available)")
                except KeyError:
                    available_labels = list(exp.available_labels())
                    if available_labels:
                        explanation_list = exp.as_list(label=available_labels[0])
                        print(f"  Warning: Using first available label {available_labels[0]} for explanation")
                    else:
                        raise ValueError("No explanation available for any label")
            dr_features = [(f, imp) for f, imp in explanation_list if f.startswith('DR_')]
            pglm_features = [(f, imp) for f, imp in explanation_list if f.startswith('PGLM_')]
            print(f"\n  Top contributing features (showing top 20):")
            print(f"    Total features analyzed: {len(explanation_list)}")
            print(f"    DR features: {len(dr_features)}, ProteinGLM features: {len(pglm_features)}")
            print(f"\n    Top 10 DR features:")
            for i, (feature, importance) in enumerate(dr_features[:10], 1):
                print(f"      {i}. {feature}: {importance:.4f}")
            print(f"\n    Top 10 ProteinGLM features:")
            if pglm_features:
                for i, (feature, importance) in enumerate(pglm_features[:10], 1):
                    print(f"      {i}. {feature}: {importance:.4f}")
            else:
                print(f"      (No ProteinGLM features in top {num_features_to_show} features)")
            dr_total_importance = sum(abs(imp) for _, imp in dr_features)
            pglm_total_importance = sum(abs(imp) for _, imp in pglm_features)
            total_importance = dr_total_importance + pglm_total_importance
            if total_importance > 0:
                dr_ratio = dr_total_importance / total_importance * 100
                pglm_ratio = pglm_total_importance / total_importance * 100
                print(f"\n    Feature importance ratio (in top {num_features_to_show}):")
                print(f"      DR: {dr_ratio:.2f}% (total: {dr_total_importance:.4f})")
                print(f"      ProteinGLM: {pglm_ratio:.2f}% (total: {pglm_total_importance:.4f})")
            print(f"\n    Fusion model weights:")
            print(f"      DR weight: {fold_weight_dr:.4f}, ProteinGLM weight: {fold_weight_pglm:.4f}")
        fold += 1
    print(f"\n{'='*60}")
    print("10-Fold Cross-Validation Results:")
    print(f"{'='*60}")
    print(f'Sensitivity (SN): {np.mean(sn_list):.4f} ± {np.std(sn_list):.4f}')
    print(f'Specificity (SP): {np.mean(sp_list):.4f} ± {np.std(sp_list):.4f}')
    print(f'Accuracy (ACC): {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}')
    print(f'Matthews Correlation Coefficient (MCC): {np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f}')
    print(f'Area Under Curve (AUC): {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}')
    result_file = os.path.join(base_dir, "Change the file name and file path to what you need.txt")

    
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("LIME Explanation - DR and ProteinGLM Weighted Fusion Model\n")
        f.write("="*60 + "\n\n")
        f.write("10-Fold Cross-Validation Results:\n")
        f.write(f"Sensitivity (SN): {np.mean(sn_list):.4f} ± {np.std(sn_list):.4f}\n")
        f.write(f"Specificity (SP): {np.mean(sp_list):.4f} ± {np.std(sp_list):.4f}\n")
        f.write(f"Accuracy (ACC): {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f}\n")
        f.write(f"Area Under Curve (AUC): {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}\n")
    print(f"\nResults saved to: {result_file}")



if __name__ == "__main__":
    main()
