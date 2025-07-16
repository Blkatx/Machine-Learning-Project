import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


def read_dataset(file_path):
    """
    Reads the dataset from the specified file path.
    
    Args:
        file_path (str): The path to the dataset file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the file path.")
    
    data = pd.read_csv(file_path)

    data_summary = {
        "describe": data.describe().to_dict(),
        "info": {
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "null_counts": data.isnull().sum().to_dict(),
            "shape": data.shape
        },
        "personality_ratio": (data["Personality"].value_counts() * 100/data.value_counts().sum()).to_dict()
    }
    return data, data_summary

def get_feature_columns(data_summary):
    """
    Determines target, numerical, and categorical columns from data_summary.

    Args:
        data_summary (dict): Summary dictionary from read_dataset.

    Returns:
        target_column (str): Name of the target column.
        num_columns (list): List of numerical feature columns.
        cat_columns (list): List of categorical feature columns.
    """
    target_column = "Personality"
    num_columns = [col for col, dtype in data_summary["info"]["dtypes"].items() if dtype == "float64"]
    cat_columns = [col for col, dtype in data_summary["info"]["dtypes"].items() if dtype == "object" and col != target_column]
    return target_column, num_columns, cat_columns

def categorical_encoding(data, cat_columns, target_column):
    """
    Encodes categorical columns and the target column in a DataFrame as binary integers.
    For each column in `cat_columns`, values equal to 'Yes' are encoded as 1, others as 0.
    For the `target_column`, values equal to 'Extrovert' are encoded as 1, others as 0.
    Args:
        data (pd.DataFrame): The input DataFrame containing categorical columns.
        cat_columns (list of str): List of column names to encode as binary features.
        target_column (str): The name of the target column to encode.
    Returns:
        pd.DataFrame: The DataFrame with encoded categorical and target columns.
    """
    for col in cat_columns:
        data[col] = (data[col] == 'Yes').astype(int)

    data[target_column] = (data[target_column] == 'Extrovert').astype(int)

    return data

def scale_numerical_features(data, num_columns):
    """
    Scales the specified numerical columns of a DataFrame using Min-Max normalization.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to be scaled.
        num_columns (list of str): List of column names in 'data' to be scaled.

    Returns:
        pd.DataFrame: The DataFrame with the specified numerical columns scaled to the range [0, 1].
    """
    scaler = MinMaxScaler()
    data[num_columns] = scaler.fit_transform(data[num_columns])
    return data

def load_and_clean_data(file_path):
    """
    Loads and cleans the dataset from the specified file path.
    
    Args:
        file_path (str): The path to the dataset file.
        
    Returns:
        X (pd.DataFrame): Features after encoding and scaling.
        y (pd.Series): Target variable.
    """
    data, data_summary = read_dataset(file_path)
    
    target_column, num_columns, cat_columns = get_feature_columns(data_summary)
    
    encoded_data = categorical_encoding(data, cat_columns, target_column)
    
    scaled_encoded_data = scale_numerical_features(encoded_data, num_columns)
    
    return scaled_encoded_data, target_column