import os
import sys
import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import random

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

PROJECT_ROOT = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Change the file name and file path to what you need')

if not os.path.exists(MODEL_PATH):
    print("Error: Model path not found at: {}".format(MODEL_PATH))
    sys.exit(1)
print("Loading ProteinGLM-1b model from: {}".format(MODEL_PATH))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.to(device)
    model.eval()
except Exception as e:
    print("Failed to load model: {}".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)


def read_mixed_fasta(file_path):
    sequences = []
    labels = []
    original_lengths = []
    if not os.path.exists(file_path):
        print("Warning: File not found: {}".format(file_path))
        return [], [], []
    print(f"Reading data from {file_path}...")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            header = lines[i].strip()
            seq_line = lines[i+1].strip()
            if header.startswith('>'):
                if '|P' in header:
                    label = 1
                elif '|N' in header:
                    label = 0
                else:
                    print(f"Warning: Unknown label in header: {header}. Skipping.")
                    continue
                seq = seq_line.upper()
                seq = re.sub(r"[UZOB]", "X", seq)
                seq_len = len(seq)
                if seq:
                    sequences.append(seq)
                    labels.append(label)
                    original_lengths.append(seq_len)
            else:
                pass
    return sequences, labels, original_lengths


def extract_and_save(input_file, output_file):
    all_seqs, all_labels, all_lens = read_mixed_fasta(input_file)
    print("Total samples in {}: {} (Positive: {}, Negative: {})".format(
        os.path.basename(input_file),
        len(all_seqs),
        sum(1 for l in all_labels if l == 1),
        sum(1 for l in all_labels if l == 0)
    ))
    if len(all_seqs) == 0:
        print("No sequences found. Skipping.")
        return
    batch_size = 4
    embeddings = []
    print(f"Extracting features for {os.path.basename(input_file)}...")
    with torch.no_grad():
        for i in tqdm(range(0, len(all_seqs), batch_size)):
            batch_seqs = all_seqs[i : i + batch_size]
            output = tokenizer(batch_seqs, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in output.items()}
            model_output = model(**inputs, output_hidden_states=True, return_last_hidden_state=True)
            if hasattr(model_output, 'hidden_states'):
                hs = model_output.hidden_states
                if isinstance(hs, tuple):
                    last_hidden_state = hs[-1]
                else:
                    last_hidden_state = hs
            else:
                raise ValueError("Model output does not contain hidden_states. Ensure output_hidden_states=True.")
            if last_hidden_state.shape[0] == inputs['input_ids'].shape[1] and last_hidden_state.shape[1] == inputs['input_ids'].shape[0]:
                 last_hidden_state = last_hidden_state.permute(1, 0, 2)
            attention_mask = inputs['attention_mask']
            for j in range(len(batch_seqs)):
                seq_len = attention_mask[j].sum().item()
                if seq_len > 1:
                     valid_embeddings = last_hidden_state[j, :seq_len-1, :]
                else:
                     valid_embeddings = last_hidden_state[j, :seq_len, :]
                mean_emb = valid_embeddings.mean(dim=0)
                embeddings.append(mean_emb.float().cpu().numpy())
    if embeddings:
        X = np.vstack(embeddings)
        y = np.array(all_labels)
        print("Features shape: {}".format(X.shape))
        cols = ["feature_{}".format(k) for k in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df['label'] = y
        out_dir = os.path.dirname(output_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Saving to {}...".format(output_file))
        df.to_csv(output_file, index=False)
        print("Saved.")

        
train_input = os.path.join(PROJECT_ROOT, 'Data', 'dataset2', 'Change the file name and file path to what you need.txt')
train_output = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'Change the file name and file path to what you need.csv')
extract_and_save(train_input, train_output)
test_input = os.path.join(PROJECT_ROOT, 'Data', 'dataset2', 'Change the file name and file path to what you need.txt')
test_output = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'Change the file name and file path to what you need.csv')
extract_and_save(test_input, test_output)
print("All extractions complete.")
