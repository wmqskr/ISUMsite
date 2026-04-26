# -- coding: UTF-8 --
import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
import os
from datetime import datetime


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


def _load_dataset1_aligned_pair(base_dir: str):
    """
    Dataset1 alignment rule used in this repo:
      - ProteinGLM PCA420 file contains label column
      - DR features are split into 775pos.csv and 17807neg.csv
      - The aligned order is: [pos] then [neg]
    Returns:
      df_dr (n, d_dr+1), df_pglm (n, d_pglm+1), y (n,)
    """
    proc_dir = os.path.join(base_dir, "Data", "Processed_Data")
    pglm_path = os.path.join(proc_dir, "ProteinGLM_features_Dataset1_pca420.csv")
    pos_dr_path = os.path.join(proc_dir, "775pos.csv")
    neg_dr_path = os.path.join(proc_dir, "17807neg.csv")

    if not (os.path.exists(pglm_path) and os.path.exists(pos_dr_path) and os.path.exists(neg_dr_path)):
        raise FileNotFoundError(
            "Missing Dataset1 aligned input files. Expected:\n"
            f"- {pglm_path}\n- {pos_dr_path}\n- {neg_dr_path}\n"
        )

    df_pglm = pd.read_csv(pglm_path)
    if "label" not in df_pglm.columns:
        # assume last column
        df_pglm = df_pglm.copy()
        df_pglm.rename(columns={df_pglm.columns[-1]: "label"}, inplace=True)

    df_pos = pd.read_csv(pos_dr_path)
    df_neg = pd.read_csv(neg_dr_path)
    # DR files include label, but we rebuild labels explicitly to avoid any mismatch
    df_dr = pd.concat([df_pos.iloc[:, :-1], df_neg.iloc[:, :-1]], axis=0).reset_index(drop=True)
    y = np.hstack([np.ones(len(df_pos), dtype=int), np.zeros(len(df_neg), dtype=int)])
    df_dr["label"] = y

    if len(df_pglm) != len(df_dr):
        raise ValueError(f"Row count mismatch: PGLM={len(df_pglm)} vs DR={len(df_dr)}")
    if not np.array_equal(df_pglm["label"].values.astype(int), y):
        raise ValueError("Label alignment mismatch between ProteinGLM and DR. Cannot produce aligned balanced data.")

    return df_dr, df_pglm, y


def Aligned_Balance_Dataset1(method: str = "under", date_tag: str = None, random_state: int = 1):
    """
    Create an aligned and balanced Dataset1 for fusion training.
    - method='under': RandomUnderSampler to 1:1 (size = 2 * minority)
    - method='over' : RandomOverSampler  to 1:1 (size = 2 * majority)

    Outputs are saved into ISUMsite/Data/Processed_Data without method/date suffixes.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    proc_dir = os.path.join(base_dir, "Data", "Processed_Data")

    if date_tag is None:
        date_tag = datetime.now().strftime("%Y%m%d")

    df_dr, df_pglm, y = _load_dataset1_aligned_pair(base_dir)

    method = (method or "").strip().lower()

    # resample by ID to guarantee alignment (sampling duplicates/selects existing rows only)
    # Supported:
    # - under: RandomUnderSampler to 1:1 (total becomes 2*minority)
    # - over : RandomOverSampler  to 1:1 (total becomes 2*majority)
    # - keepn: keep total N (close to original), target pos=neg=N/2 via (over pos w/ replacement) + (under neg w/o replacement)
    if method in {"over", "under"}:
        ids = np.arange(len(y)).reshape(-1, 1)
        if method == "over":
            sampler = RandomOverSampler(random_state=random_state)
        else:
            sampler = RandomUnderSampler(random_state=random_state)
        ids_res, y_res = sampler.fit_resample(ids, y)
        ids_res = ids_res.reshape(-1)
    elif method in {"keepn", "keep", "match_total"}:
        rng = np.random.RandomState(random_state)
        n_total = len(y)
        target_each = n_total // 2
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(neg_idx) < target_each or len(pos_idx) == 0:
            raise ValueError(f"Cannot keepN balance: pos={len(pos_idx)} neg={len(neg_idx)} total={n_total}")

        # Negatives: downsample without replacement to target_each
        neg_pick = rng.choice(neg_idx, size=target_each, replace=False)
        # Positives: oversample with replacement to target_each
        pos_pick = rng.choice(pos_idx, size=target_each, replace=True)

        ids_res = np.concatenate([pos_pick, neg_pick], axis=0)
        rng.shuffle(ids_res)
        y_res = y[ids_res]
    else:
        raise ValueError("method must be 'under', 'over', or 'keepN'")

    df_dr_bal = df_dr.iloc[ids_res].reset_index(drop=True)
    df_pglm_bal = df_pglm.iloc[ids_res].reset_index(drop=True)

    # Save using stable names so downstream scripts do not depend on method/date suffixes.
    method_tag = "keepN" if method in {"keepn", "keep", "match_total"} else method
    out_dr = os.path.join(proc_dir, "DR_Dataset1_balanced.csv")
    out_pglm = os.path.join(proc_dir, "ProteinGLM_Dataset1_pca420_balanced.csv")
    out_ids = os.path.join(proc_dir, "Dataset1_balanced_indices.npy")

    df_dr_bal.to_csv(out_dr, index=False)
    df_pglm_bal.to_csv(out_pglm, index=False)
    np.save(out_ids, ids_res)

    print(f"[Aligned Balance] method={method_tag} date={date_tag}")
    print(f"Saved DR balanced -> {out_dr} shape={df_dr_bal.shape}")
    print(f"Saved PGLM balanced -> {out_pglm} shape={df_pglm_bal.shape}")
    print(f"Saved indices -> {out_ids} len={len(ids_res)}")
    print(f"Label counts -> pos={int((y_res==1).sum())}, neg={int((y_res==0).sum())}")


if __name__ == "__main__":
    # Example usage:
    #   python Over_Under_sampling.py under
    #   python Over_Under_sampling.py over
    #   python Over_Under_sampling.py keepN
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "under"
    Aligned_Balance_Dataset1(method=m)
