"""Data access and validation for the AI4I predictive-maintenance dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "Dataset" / "ai4i2020.csv"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN = "Machine failure"
FAILURE_INDICATOR_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load AI4I data and fail early when its required columns are missing."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataframe = pd.read_csv(data_path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN] + FAILURE_INDICATOR_COLUMNS)
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    return dataframe


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a deduplicated, complete copy suitable for reproducible training."""
    cleaned = dataframe.drop_duplicates().dropna().copy()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned
