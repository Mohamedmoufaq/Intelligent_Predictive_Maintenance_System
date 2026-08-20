"""Inference functions used by the dashboard and tests."""

from pathlib import Path

import joblib
import pandas as pd

from Src.data_loader import FEATURE_COLUMNS
from Src.utils import calculate_risk, estimate_costs, maintenance_decision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "Models"


def load_artifacts(models_dir: Path = MODELS_DIR) -> dict:
    """Load trained production artifacts. Run `python -m Src.train_model` first."""
    model_path = models_dir / "failure_model.joblib"
    type_model_path = models_dir / "failure_type_model.joblib"
    metadata_path = models_dir / "model_metadata.joblib"
    missing_artifacts = [
        path.name for path in (model_path, metadata_path) if not path.exists()
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            f"Required trained artifact(s) missing: {', '.join(missing_artifacts)}. "
            "Run: python -m Src.train_model"
        )
    return {
        "failure_model": joblib.load(model_path),
        "failure_type_model": joblib.load(type_model_path) if type_model_path.exists() else None,
        "metadata": joblib.load(metadata_path),
    }


def predict_machine(input_data: dict, artifacts: dict | None = None) -> dict:
    """Produce model output plus transparent risk, maintenance, and cost decisions."""
    missing_features = set(FEATURE_COLUMNS) - set(input_data)
    if missing_features:
        raise ValueError(f"Missing input features: {sorted(missing_features)}")

    artifacts = artifacts or load_artifacts()
    sample = pd.DataFrame([{feature: input_data[feature] for feature in FEATURE_COLUMNS}])
    failure_model = artifacts["failure_model"]
    probability = float(failure_model.predict_proba(sample)[0, 1])
    threshold = artifacts["metadata"]["decision_threshold"]
    predicted_failure = probability >= threshold
    risk_score, risk_level = calculate_risk(probability)

    likely_failure_type = "No failure predicted"
    type_model = artifacts["failure_type_model"]
    if predicted_failure and type_model is not None:
        likely_failure_type = str(type_model.predict(sample)[0])

    return {
        "machine_failure": bool(predicted_failure),
        "failure_probability": probability,
        "decision_threshold": threshold,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "likely_failure_type": likely_failure_type,
        "maintenance_recommendation": maintenance_decision(probability, threshold, risk_level),
        **estimate_costs(probability),
    }
