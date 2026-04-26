import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

PROJECT_ROOT = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"
TRAIN_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'Change the file name and file path to what you need.csv')
TEST_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'Change the file name and file path to what you need.csv')
TRAIN_OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'Change the file name and file path to what you need.csv')
TEST_OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'Change the file name and file path to what you need.csv')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'Dataset2', 'outputs', 'Change the file name and file path to what you need.pth')
INPUT_DIM = 2048
LATENT_DIM = 1280
HIDDEN_DIM = 1536
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return None, None
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values
    return X, y


def save_features(features, labels, output_file):
    cols = [f"feature_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=cols)
    df['label'] = labels
    df.to_csv(output_file, index=False)
    print(f"Saved features to {output_file}")


def main():
    X_train, y_train = load_data(TRAIN_FILE)
    if X_train is None: return
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    train_dataset = TensorDataset(torch.tensor(X_train_scaled))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = Autoencoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(DEVICE)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
    print("Training complete.")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    model.eval()
    with torch.no_grad():
        print("Extracting features for Training set...")
        X_train_tensor = torch.tensor(X_train_scaled).to(DEVICE)
        encoded_train, _ = model(X_train_tensor)
        X_train_encoded = encoded_train.cpu().numpy()
        save_features(X_train_encoded, y_train, TRAIN_OUTPUT_FILE)
        X_test, y_test = load_data(TEST_FILE)
        if X_test is not None:
            print("Extracting features for Testing set...")
            X_test_scaled = scaler.transform(X_test)
            X_test_tensor = torch.tensor(X_test_scaled).to(DEVICE)
            encoded_test, _ = model(X_test_tensor)
            X_test_encoded = encoded_test.cpu().numpy()
            save_features(X_test_encoded, y_test, TEST_OUTPUT_FILE)

            
if __name__ == "__main__":
    main()
