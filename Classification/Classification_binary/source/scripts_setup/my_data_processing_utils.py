import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

def scale_dataset(df_param, random_state=None, oversample = False):
  X = df_param[df_param.columns[:-1]].values
  y = df_param[df_param.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler(random_state=random_state)
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y

def save_datasets_to_csv(X_train, y_train, X_test, y_test, X_valid, y_valid, data_path="../../data/processed/"):
    """
    Save datasets to CSV files and return their file paths.

    Args:
    - X_train: DataFrame or array-like, training features
    - y_train: DataFrame or array-like, training labels
    - X_test: DataFrame or array-like, testing features
    - y_test: DataFrame or array-like, testing labels
    - X_valid: DataFrame or array-like, validation features
    - y_valid: DataFrame or array-like, validation labels
    - data_path: str, path where the CSV files will be saved (default: '../../data/processed/')

    Returns:
    - dict: A dictionary containing the file paths of the saved datasets
    """

    # Ensure data_path is an absolute path
    data_path = os.path.abspath(data_path)

    # Save datasets to CSV files
    datasets = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_valid": X_valid,
        "y_valid": y_valid
    }

    dataset_paths = {}  # Dictionary to store the file paths

    for dataset_name, dataset in datasets.items():
        file_path = os.path.join(data_path, f"{dataset_name}.csv")
        dataset_paths[dataset_name + "_path"] = file_path
        pd.DataFrame(dataset).to_csv(file_path, index=False)

    return dataset_paths

def split_data(df, test_size=0.4, valid_size=0.5, random_state=None):
    """
    Splits the input DataFrame into training, testing, and validation sets.

    Parameters:
    - df: DataFrame: The input DataFrame to split.
    - test_size: float, default=0.4: The proportion of the data to include in the testing split.
    - valid_size: float, default=0.5: The proportion of the data to include in the validation split.
    - random_state: int or RandomState instance, default=None: Controls the randomness of the shuffle.

    Returns:
    - df_train: DataFrame: The training set.
    - df_test: DataFrame: The testing set.
    - df_valid: DataFrame: The validation set.
    """
    # Shuffle with a fixed random state for reproducibility
    data_shuffled = df.sample(frac=1, random_state=random_state)  

    # Split data into training and temporary sets
    df_train, df_temp = train_test_split(data_shuffled, test_size=test_size, random_state=random_state)

    # Split the temporary set into testing and validation sets
    df_test, df_valid = train_test_split(df_temp, test_size=valid_size, random_state=random_state)

    # Reset the indices of the resulting DataFrames
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    return df_train, df_test, df_valid