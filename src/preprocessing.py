import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from config import RANDOM_STATE, TEST_SIZE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.feature_engineering import create_engineered_features

FAILURE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


# --------------------------------------------------
# Target Construction
# --------------------------------------------------

def create_multiclass_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Failure Type multiclass target.
    """
    df = df.copy()

    def assign_failure_type(row):
        failures = [col for col in FAILURE_COLUMNS if row[col] == 1]

        if len(failures) == 0:
            return "No Failure"
        elif len(failures) == 1:
            return failures[0]
        else:
            return "Multiple Failure"

    df["Failure Type"] = df.apply(assign_failure_type, axis=1)

    return df


# --------------------------------------------------
# Leakage Removal
# --------------------------------------------------

def drop_failure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove failure indicator columns to prevent leakage.
    """
    return df.drop(columns=FAILURE_COLUMNS)


# --------------------------------------------------
# Dataset Builders
# --------------------------------------------------

def build_binary_dataset(df: pd.DataFrame):
    X = df.drop(columns=["Machine failure", "Failure Type"])
    y = df["Machine failure"]
    return X, y


def build_multiclass_dataset(df: pd.DataFrame):
    X = df.drop(columns=["Machine failure", "Failure Type"])
    y = df["Failure Type"]
    return X, y


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


# --------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------

def build_preprocessing_pipeline(X: pd.DataFrame):

    categorical_cols = ["Type"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    return preprocessor

  


# --------------------------------------------------
# Transfer feature engineering Pipeline
# --------------------------------------------------

def build_full_preprocessing_pipeline(X: pd.DataFrame):

    categorical_cols = ["Type"]

    numerical_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temp_diff",
        "Power"
    ]

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    full_pipeline = Pipeline(steps=[
        ("feature_engineering",
         FunctionTransformer(create_engineered_features, validate=False)),
        ("preprocessing", column_transformer)
    ])

    return full_pipeline

'''

def build_full_preprocessing_pipeline(X: pd.DataFrame):

    categorical_cols = ["Type"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    full_pipeline = Pipeline(steps=[
        ("feature_engineering",
         FunctionTransformer(create_engineered_features, validate=False)),
        ("preprocessing", column_transformer)
    ])

    return full_pipeline

'''
