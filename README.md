# Maize Yield Prediction Model (nn02_optimized)

This document provides an overview of the `nn02_optimized.ipynb` notebook, which builds and trains a neural network to predict maize yield based on genotypic and environmental data.

## Description

The notebook implements a feed-forward neural network using PyTorch. It includes data preprocessing, feature scaling, model training with validation, and final evaluation on a held-out test set.

## How to Run

1.  **Install Dependencies**: Make sure you have a Python environment with the necessary packages installed. You can install them using the `requirements.txt` file in the root directory.

    ```bash
    pip install -r ../requirements.txt
    # On macOS, you may also need to install the graphviz library system-wide
    brew install graphviz
    ```

2.  **Run the Notebook**: Open `nn02_optimized.ipynb` in a Jupyter environment and execute the cells sequentially from top to bottom.

## Key Steps in the Notebook

1.  **Load and Preprocess Data**: Loads the `C1.{selected_line|ALL}_PHENOnENV_PCA.csv` dataset, handles missing values via mean imputation, and separates features from the target variable (`YLD_BE`).
2.  **Data Splitting**: Splits the data into training (80%), validation (10%), and test (10%) sets.
3.  **Feature Scaling**: Applies `StandardScaler` to normalize the feature values, fitting only on the training data to prevent data leakage.
4.  **Model Definition**: Defines a `MaizeYieldPredictor` class, which is a sequential neural network with three hidden layers. It includes `BatchNorm1d` and `Dropout` for regularization.
5.  **Training**: Trains the model for a set number of epochs, using the validation set to monitor for overfitting.
6.  **Evaluation**: Evaluates the model's performance on the test set using Root Mean Squared Error (RMSE) and R-squared (R²) metrics. It also visualizes the results.
7.  **Visualization**: Includes code to visualize the model architecture using `torchviz` and `torchview`.
8.  **Save Model**: Saves the trained model's state dictionary to `maize_yield_predictor_nn02.pth`.

## Model Architecture

The model is a fully connected neural network with the following structure:

-   **Input Layer**: Takes 21 features.
-   **Hidden Layer 1**: 256 nodes, followed by `BatchNorm`, `ReLU`, and `Dropout`.
-   **Hidden Layer 2**: 128 nodes, followed by `BatchNorm`, `ReLU`, and `Dropout`.
-   **Hidden Layer 3**: 64 nodes, followed by `BatchNorm` and `ReLU`.
-   **Output Layer**: 1 node for the final yield prediction.
