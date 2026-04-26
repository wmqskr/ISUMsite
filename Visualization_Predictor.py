import codecs
import csv
import pickle
import os
import sys
import glob
import numpy as np
import pandas as pd


BASE_DIR = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"
DATASET1_CODE_DIR = os.path.join(BASE_DIR, "Dataset1", "code")
DATASET1_OUTPUT_DIR = os.path.join(BASE_DIR, "Dataset1", "outputs")
PREDICT_DATA_DIR = os.path.join(BASE_DIR, "Change the file name and file path to what you need")


if DATASET1_CODE_DIR not in sys.path:
    sys.path.append(DATASET1_CODE_DIR)
try:
    import Tkinter as tk
    import tkFileDialog
    import tkMessageBox
except ImportError:
    import tkinter as tk
    from tkinter import filedialog as tkFileDialog
    from tkinter import messagebox as tkMessageBox

import Data_Process_Tools as Tools

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    PROTEINGLM_AVAILABLE = True
except ImportError:
    PROTEINGLM_AVAILABLE = False
    print("Warning: PyTorch or transformers not available. ProteinGLM features cannot be extracted.")


from dataclasses import dataclass
@dataclass(frozen=True)
class ModelParams:
    rfc1_n: int
    gbc_n: int
    rfc2_n: int
def select_file():
    file_path = tkFileDialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        print("Selected file path:", file_path)
        Tools.process_file(file_path)
        root.destroy()
root = tk.Tk()
root.title("Protein Sequence Prediction")
window_width = 512
window_height = 512
root.geometry("{}x{}".format(window_width, window_height))
button = tk.Button(root, text="Select the .fsa file", command=select_file, width=20, height=2)
button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
root.mainloop()


def load_fusion_model():
    import sys
    if '__main__' not in sys.modules:
        import __main__
        sys.modules['__main__'] = __main__
    if not hasattr(sys.modules['__main__'], 'ModelParams'):
        sys.modules['__main__'].ModelParams = ModelParams
    if 'Train_Fusion_Weighted_Dataset1' not in sys.modules:
        import types
        train_module = types.ModuleType('Train_Fusion_Weighted_Dataset1')
        sys.modules['Train_Fusion_Weighted_Dataset1'] = train_module
    if not hasattr(sys.modules['Train_Fusion_Weighted_Dataset1'], 'ModelParams'):
        sys.modules['Train_Fusion_Weighted_Dataset1'].ModelParams = ModelParams
    fusion_pattern = os.path.join(DATASET1_OUTPUT_DIR, "Change the file name and file path to what you need.pkl")
    fusion_files = glob.glob(fusion_pattern)
    if not fusion_files:
        raise FileNotFoundError(
            f"Fusion model not found. Expected file matching pattern: {fusion_pattern}\n"
            "Please ensure the fusion model file exists."
        )
    fusion_model_path = fusion_files[0]
    print(f"Loading fusion model from: {fusion_model_path}")
    with open(fusion_model_path, "rb") as f:
        bundle = pickle.load(f)
    dr_model = bundle["dr_model"]
    pglm_model = bundle["pglm_model"]
    weight_dr = bundle["weight_dr"]
    weight_pglm = bundle["weight_pglm"]
    print(f"Loaded fusion model with weights: DR={weight_dr:.4f}, PGLM={weight_pglm:.4f}")
    return dr_model, pglm_model, weight_dr, weight_pglm


def load_proteinglm_pca():
    candidate_paths = [
        os.path.join(PREDICT_DATA_DIR, "Change the file name and file path to what you need.pkl"),
        os.path.join(DATASET1_OUTPUT_DIR, "Change the file name and file path to what you need.pkl"),
        os.path.join(BASE_DIR, "Change the file name and file path to what you need.pkl"),
    ]
    pca_model_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if pca_model_path is None:
        raise FileNotFoundError(
            "ProteinGLM PCA model not found.\n"
            "Checked:\n- " + "\n- ".join(candidate_paths)
        )
    with open(pca_model_path, "rb") as f:
        pca_model = pickle.load(f)
    print(f"Loaded ProteinGLM PCA model from: {pca_model_path}")
    return pca_model


def extract_proteinglm_features(sequences):
    if not PROTEINGLM_AVAILABLE:
        raise RuntimeError("ProteinGLM feature extraction requires PyTorch and transformers")
    model_path = os.path.join(BASE_DIR, 'Change the file name and file path to what you need')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ProteinGLM model not found at: {model_path}\n"
            "Please ensure the ProteinGLM model is available."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for ProteinGLM feature extraction")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.to(device)
    model.eval()
    batch_size = 4
    embeddings = []
    print(f"Extracting ProteinGLM features for {len(sequences)} sequences...")
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            output = tokenizer(
                batch_seqs,
                add_special_tokens=True,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            )
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
        print(f"ProteinGLM features extracted. Shape: {X.shape}")
        return X
    else:
        raise ValueError("No embeddings extracted")
    

def read_sequences_from_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines), 2):
            seq = lines[i].strip()
            sequences.append(seq)
    return sequences


def read_csv(file_path):
    features = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            features.append([float(x) for x in row])
    return np.array(features)


print("Loading models...")
dr_model, pglm_model, weight_dr, weight_pglm = load_fusion_model()
pca_model = load_proteinglm_pca()
print("Loading DR features...")
dr_features = read_csv(os.path.join(PREDICT_DATA_DIR, "Change the file name and file path to what you need.csv"))
print("Extracting ProteinGLM features...")
copy_file = os.path.join(PREDICT_DATA_DIR, "Change the file name and file path to what you need.txt")
if not os.path.exists(copy_file):
    raise FileNotFoundError(f"Sequence file not found: {copy_file}")
sequences = read_sequences_from_fasta(copy_file)
if len(sequences) != len(dr_features):
    raise ValueError(
        f"Sequence count ({len(sequences)}) does not match DR feature count ({len(dr_features)})"
    )
pglm_features_raw = extract_proteinglm_features(sequences)
print("Reducing ProteinGLM features to 420 dimensions...")
pglm_features = pca_model.transform(pglm_features_raw)
print(f"DR features shape: {dr_features.shape}")
print(f"ProteinGLM features shape: {pglm_features.shape}")
print("Making predictions with fusion model...")
prob_dr = dr_model.predict_proba(dr_features)[:, 1]
prob_pglm = pglm_model.predict_proba(pglm_features)[:, 1]
prob_fused = weight_dr * prob_dr + weight_pglm * prob_pglm
predictions = (prob_fused >= 0.5).astype(int)
probabilities = np.column_stack([1.0 - prob_fused, prob_fused])



class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.page_index = 0
        self.init_window()
    def init_window(self):
        self.master.title("Protein Sequence Prediction")
        self.master.geometry("{}x{}".format(window_width, window_height))
        self.text_box = tk.Text(self.master, height=30, width=60)
        self.text_box.pack()
        self.prev_page_button = tk.Button(self.master, text="Previous Page", command=self.prev_page, padx=0, pady=5)
        self.prev_page_button.pack(side=tk.LEFT, padx=25, pady=5)
        self.next_page_button = tk.Button(self.master, text="Next Page", command=self.next_page, padx=0, pady=5)
        self.next_page_button.pack(side=tk.LEFT, padx=25, pady=5)
        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit_program, padx=20, pady=5)
        self.quit_button.pack(side=tk.RIGHT, padx=25, pady=5)
        self.save_button = tk.Button(self.master, text="Save All", command=self.save_all_to_file, padx=20, pady=5)
        self.save_button.pack(side=tk.RIGHT, padx=25, pady=5)
        self.update_content()
    def update_content(self):
        self.text_box.delete(1.0, tk.END)
        start_index = self.page_index * 20
        end_index = min(start_index + 20, len(lines))
        for i in range(start_index, end_index):
            line_number = i + 1
            if line_number % 2 == 0:
                self.text_box.insert(tk.END, lines[i])
    def save_all_to_file(self):
        file_path = tkFileDialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with codecs.open(file_path, "w", encoding="utf-8") as file:
                for page_index in range(len(lines) // 20 + 1):
                    self.page_index = page_index
                    self.update_content()
                    content = self.text_box.get("1.0", "end")
                    file.write(content)
            tkMessageBox.showinfo("Saved successfully", "Data has been saved to file {}".format(file_path))
    def next_page(self):
        if (self.page_index + 1) * 20 < len(lines):
            self.page_index += 1
            self.update_content()
    def prev_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.update_content()
    def quit_program(self):
        self.master.destroy()
def print_even_lines(filename):
    try:
        with open(filename, 'r') as file:
            global lines
            lines = file.readlines()
            protein_count = 1
            for i in range(1, len(lines), 2):
                line = lines[i].strip() + '\n'
                line = "Protein {}: ".format(protein_count) + line
                protein_count += 1
                line += print_probabilities(i)
                lines[i] = line
            root = tk.Tk()
            app = Application(master=root)
            app.mainloop()
    except IOError:
        print("File not found or unreadable")
def print_probabilities(start_index):
    start_index = float(start_index) / 2
    start_index = int(start_index)
    if probabilities[start_index][1] >= 0.5:
        return "{:.3f} -->positive sample\n\n".format(probabilities[start_index][1])
    else:
        return "{:.3f} -->negative sample\n\n".format(probabilities[start_index][1])
    
print_even_lines(os.path.join(PREDICT_DATA_DIR, "copy.txt"))
count_0 = sum(1 for prediction in predictions if prediction == 0)
count_1 = sum(1 for prediction in predictions if prediction == 1)
print("Number of negative samples in the prediction results: {}".format(count_0))
print("Number of positive samples in the prediction results: {}".format(count_1))
