# Project Documentation

## Overview

This project contains two main workflows:

1. `Dataset1`: DR feature extraction + ProteinGLM feature extraction + dimensionality reduction + sampling/alignment + single-model optimization + fusion modeling
2. `Dataset2`: ProteinGLM feature extraction + autoencoder-based dimensionality reduction + NearMiss sampling + model training and testing

All file paths in the current codebase have been converted to placeholder style for easier reuse by other researchers. The main placeholders are:

- `WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH`
- `Change the file name and file path to what you need.*`

Before reproducing the project, replace these placeholders with your own actual paths and filenames.

## Dataset1 Workflow

### 1. DR Feature Extraction

Script:

- `Dataset1/code/Data_Process_Tools.py`

Function:

- Uses the DR feature extraction pipeline to generate DR features.

### 2. ProteinGLM Feature Extraction

Script:

- `Dataset1/code/Extract_ProteinGLM_Features_Dataset1.py`

Raw input data:

- `Data/dataset1/775positive_samples.txt`
- `Data/dataset1/17807negative_samples.txt`

Function:

- Loads the ProteinGLM model
- Reads positive and negative sample sequences
- Extracts ProteinGLM embedding features
- Saves the ProteinGLM feature table for Dataset1

### 3. Dimensionality Reduction to 420 Dimensions

Script:

- `Dataset1/code/Reduce_Dimension_PCA.py`

Function:

- Reduces ProteinGLM features to 420 dimensions
- Saves the reduced feature file and the PCA model

Purpose:

- Aligns the ProteinGLM feature dimension with the DR feature dimension used in the fusion stage

### 4. Sampling and Alignment

Script:

- `Dataset1/code/Over_Under_sampling.py`

Function:

- Supports oversampling, undersampling, and balanced alignment
- Generates aligned balanced DR and ProteinGLM data for the downstream fusion model

### 5. Single-Model Training and Parameter Optimization

Scripts:

- `Dataset1/code/Train_Model1_DR_GA.py`
- `Dataset1/code/Train_Model1_ProteinGLM_GA.py`

Function:

- Trains the DR submodel and the ProteinGLM submodel separately
- Uses a genetic algorithm to search for better parameters
- Saves each model and its cross-validation results

### 6. Final Fusion Model

Script:

- `Dataset1/code/Train_Fusion_Weighted_Dataset1.py`

Function:

- Loads aligned DR and ProteinGLM data for Dataset1
- Uses the optimized parameters from the two submodels
- Generates OOF predictions
- Searches for the best fusion weights
- Saves the final fusion results and fusion model

## Dataset2 Workflow

### 1. ProteinGLM Feature Extraction

Script:

- `Dataset2/code/Extract_ProteinGLM_Features.py`

Raw input data:

- `Data/dataset2/Dataset2_Train.txt`
- `Data/dataset2/Dataset2_Test.txt`

Function:

- Extracts ProteinGLM features for the Dataset2 training and test sets

### 2. Autoencoder-Based Dimensionality Reduction

Script:

- `Dataset2/code/Reduce_Dimension_Autoencoder.py`

Function:

- Trains an autoencoder on Dataset2 ProteinGLM features
- Exports compressed training and test features
- Saves the autoencoder model

### 3. NearMiss Sampling

Script:

- `Dataset2/code/Undersample_NearMiss.py`

Function:

- Applies NearMiss undersampling to the specified Dataset2 feature file

### 4. Model Training and Testing

Script:

- `Dataset2/code/Train_Model2_Dataset2.py`

Function:

- Uses `RandomizedSearchCV` to optimize the Dataset2 ProteinGLM ensemble model
- Evaluates the model on the test set
- Saves the results and trained model

## Interpretability and Prediction Scripts

### 1. `Interpretability/Lime_Explaination.py`

Purpose:

- Performs LIME interpretability analysis on the final fusion model for `Dataset1`
- Analyzes the contribution of DR features and ProteinGLM features in the fusion prediction
- Outputs an explanation file and prints cross-validation metrics

Required inputs:

- The saved fusion model file in `Dataset1/outputs`
- The aligned balanced DR data in `Dataset1/intermediate`
- The aligned balanced ProteinGLM data in `Dataset1/intermediate`

Main outputs:

- An `.html` file containing the LIME explanation result
- Cross-validation metrics and feature contribution information printed in the console

How to use:

1. Replace the placeholder path `WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH` with your own project root path.
2. Replace the placeholder filename `Change the file name and file path to what you need.*` with your actual filenames.
3. Make sure `lime`, `imbalanced-learn`, `scikit-learn`, `pandas`, and `numpy` are installed.
4. Run:

```bash
python Interpretability/Lime_Explaination.py
```

Notes:

- This script is designed for the `Dataset1` fusion model, not for `Dataset2`.
- The DR data and ProteinGLM data must already be aligned and balanced; otherwise, label consistency checks will fail.
- If no saved fusion model is found, the script falls back to retraining the submodels during cross-validation.
- The current code still uses placeholder filenames, so users must replace them with local paths before running the script.

### 2. `Visualization_Predictor.py`

Purpose:

- Provides a graphical user interface for loading protein sequence files for prediction
- Automatically generates DR features, extracts ProteinGLM features, and reduces their dimensionality
- Uses the `Dataset1` fusion model to predict positive or negative labels for the input sequences
- Displays prediction results in the GUI and supports exporting results to a text file

Required inputs:

- A user-selected sequence file for prediction
- The saved fusion model file in `Dataset1/outputs`
- The ProteinGLM PCA model file
- The ProteinGLM pretrained model directory
- `Dataset1/code/Data_Process_Tools.py`

Main outputs:

- Intermediate files generated during prediction, such as the DR feature file and copied sequence file
- Paginated prediction results in the GUI
- A user-exported prediction result `.txt` file

Workflow:

1. Open the GUI and select the sequence file to be predicted.
2. Call `Data_Process_Tools.py` to generate DR features and intermediate sequence files.
3. Load the fusion model and PCA model.
4. Use ProteinGLM to extract sequence embedding features.
5. Reduce the ProteinGLM features to 420 dimensions.
6. Combine the DR submodel predictions and ProteinGLM submodel predictions using the fusion weights.
7. Display the prediction probability and label of each sequence in a paginated GUI view.

How to use:

1. Replace `WRITE_YOUR_OWN_FILE_PATH_IN_ENGLISH` in the script with your own project root path.
2. Replace all `Change the file name and file path to what you need.*` placeholders with your own input, model, and output filenames.
3. Make sure `tkinter`, `torch`, `transformers`, `numpy`, and `pandas` are available in your environment.
4. Run:

```bash
python Visualization_Predictor.py
```

Notes:

- This script requires a graphical desktop environment and may not run directly on headless servers.
- If `torch` or `transformers` is not installed, ProteinGLM feature extraction cannot be executed.
- If the fusion model, PCA model, or ProteinGLM pretrained model paths are not configured correctly, the script will fail immediately.
- This script is intended for prediction with the `Dataset1` fusion model and is not designed for direct use with `Dataset2`.
- The number of DR feature rows generated by `Data_Process_Tools.py` must match the number of input sequences; otherwise, the script will fail due to sample count mismatch.
