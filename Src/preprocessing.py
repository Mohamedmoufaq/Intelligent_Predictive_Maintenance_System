"""Shared feature preprocessing. Keep this logic inside the saved model pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Src.data_loader import FEATURE_COLUMNS


CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]


def create_preprocessor() -> ColumnTransformer:
    """Encode machine type and scale numeric sensors without data leakage."""
    return ColumnTransformer(
        transformers=[
            ("machine_type", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("sensors", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def create_pipeline(model) -> Pipeline:
    """Bundle preprocessing and a classifier so prediction always matches training."""
    return Pipeline([( "preprocessor", create_preprocessor()), ("model", model)])
