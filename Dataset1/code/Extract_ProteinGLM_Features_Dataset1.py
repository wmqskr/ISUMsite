import os
import sys
import torch
import pandas as pd
import numpy as np
import random
import re
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm


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

MODEL_PATH = os.path.join(PROJECT_ROOT, 'biomap-research', 'proteinglm-1b-mlm')
POS_FILE = os.path.join(PROJECT_ROOT, 'Data', 'dataset1', '775positive_samples.txt')
NEG_FILE = os.path.join(PROJECT_ROOT, 'Data', 'dataset1', '17807negative_samples.txt')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'ProteinGLM_features_Dataset1.csv')

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



def read_fasta(file_path, label_value):
    sequences = []
    labels = []
    if not os.path.exists(file_path):
        print("Warning: File not found: {}".format(file_path))
        return [], []
    print(f"Reading data from {file_path}...")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:

            continue

        if line.startswith('>'):

            continue 

        else:

            

            

            seq = line.upper()

            

            

            seq = re.sub(r"[UZOB]", "X", seq)

            

            

            

            

            if seq:

                sequences.append(seq)

                labels.append(label_value)

    

    return sequences, labels





all_seqs = []

all_labels = []



print("Reading positive samples...")

pos_seqs, pos_labels = read_fasta(POS_FILE, 1)

print("Reading negative samples...")

neg_seqs, neg_labels = read_fasta(NEG_FILE, 0)



all_seqs = pos_seqs + neg_seqs

all_labels = pos_labels + neg_labels



print("Total samples: {} (Positive: {}, Negative: {})".format(

    len(all_seqs), len(pos_seqs), len(neg_seqs)

))



if len(all_seqs) == 0:

    print("No sequences found. Exiting.")

    sys.exit(1)





batch_size = 4 

embeddings = []



print("Extracting ProteinGLM features...")



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

            raise ValueError("Model output does not contain hidden_states.")



        

        

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



    print("Features extracted. Shape: {}".format(X.shape))



    

    cols = ["feature_{}".format(i) for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=cols)

    df['label'] = y



    

    out_dir = os.path.dirname(OUTPUT_FILE)

    if not os.path.exists(out_dir):

        os.makedirs(out_dir)



    print("Saving to {}...".format(OUTPUT_FILE))

    df.to_csv(OUTPUT_FILE, index=False)

    print("Done! Features extracted for Dataset1 using ProteinGLM.")

else:

    print("No embeddings extracted.")

