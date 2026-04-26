import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

import pickle

import os



PROJECT_ROOT = "WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH"

INPUT_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'ProtBert_features_Dataset1.csv')

OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Processed_Data', 'ProtBert_features_Dataset1_pca420.csv')

PCA_MODEL_FILE = os.path.join(PROJECT_ROOT, 'Dataset1', 'outputs', 'ProtBert_pca_model_420Dataset1.pkl')

TARGET_DIM = 420



def reduce_dimension():

    print("Loading original ProGen2 features from {}...".format(INPUT_FILE))

    if not os.path.exists(INPUT_FILE):

        print("Error: Input file not found. Please run feature extraction first.")

        return



    df = pd.read_csv(INPUT_FILE)

    

    

    X = df.iloc[:, :-1].values

    y = df.iloc[:, -1].values

    

    print("Original shape: {}".format(X.shape))

    

    

    print("Fitting PCA to reduce dimensions to {}...".format(TARGET_DIM))

    pca = PCA(n_components=TARGET_DIM, random_state=1)

    X_pca = pca.fit_transform(X)

    

    print("Reduced shape: {}".format(X_pca.shape))

    print("Explained variance ratio sum: {:.4f}".format(np.sum(pca.explained_variance_ratio_)))

    

    

    with open(PCA_MODEL_FILE, "wb") as f:

        pickle.dump(pca, f)

    print("PCA model saved to {}".format(PCA_MODEL_FILE))

    

    

    cols = ["feature_{}".format(i) for i in range(TARGET_DIM)]

    df_pca = pd.DataFrame(X_pca, columns=cols)

    df_pca['label'] = y

    

    df_pca.to_csv(OUTPUT_FILE, index=False)

    print("Reduced features saved to {}".format(OUTPUT_FILE))



if __name__ == "__main__":

    reduce_dimension()



