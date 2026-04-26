import pandas as pd
from imblearn.under_sampling import NearMiss
import os

PROJECT_ROOT = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"
INPUT_CSV = os.path.join(PROJECT_ROOT, "Data", "Processed_Data", "Change the file name and file path to what you need.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "Data", "Processed_Data", "Change the file name and file path to what you need.csv")

def Near_Undersample(input_file, output_file):
    print(f"Reading data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return
    data = pd.read_csv(input_file)

    if 'label' not in data.columns:
        print("Warning: 'label' column not found. Assuming last column is label.")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        y.name = 'label'

    else:
        X = data.drop('label', axis=1)
        y = data['label']
    print("Original dataset shape:")
    print(y.value_counts())
    positive_count = sum(y == 1)
    negative_count = sum(y == 0)
    print(f"Original Positive samples: {positive_count}")
    print(f"Original Negative samples: {negative_count}")
    print("Running NearMiss undersampling (version 1)...")
    nm = NearMiss(version=1)
    X_resampled, y_resampled = nm.fit_resample(X, y)
    print("Resampled dataset shape:")

    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

    if not isinstance(y_resampled, pd.Series):
        y_resampled = pd.Series(y_resampled, name='label')
    print(y_resampled.value_counts())
    final_pos = sum(y_resampled == 1)
    final_neg = sum(y_resampled == 0)
    print(f"Final Positive samples: {final_pos}")
    print(f"Final Negative samples: {final_neg}")
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
    resampled_data = resampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Saving to {output_file}...")
    out_dir = os.path.dirname(output_file)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    resampled_data.to_csv(output_file, index=False)
    print("Done.")


if __name__ == "__main__":
    Near_Undersample(INPUT_CSV, OUTPUT_CSV)
