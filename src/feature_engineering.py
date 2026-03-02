import pandas as pd


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed engineered features.
    """
    df = df.copy()

    # Temperature difference
    df["Temp_diff"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    # Mechanical power proxy
    df["Power"] = (
        df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    )

    return df