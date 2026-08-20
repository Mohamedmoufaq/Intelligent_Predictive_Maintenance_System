"""Optional real SHAP explanations for supported tree production models."""

import pandas as pd


def local_shap_explanation(failure_pipeline, sample: pd.DataFrame, top_n: int = 3) -> list[dict[str, float | str]]:
    """Return the strongest local SHAP feature effects; never fabricate explanations."""
    try:
        import shap

        preprocessor = failure_pipeline.named_steps["preprocessor"]
        classifier = failure_pipeline.named_steps["model"]
        if not hasattr(classifier, "feature_importances_"):
            return []
        transformed = preprocessor.transform(sample)
        values = shap.TreeExplainer(classifier).shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, 1]
        feature_names = preprocessor.get_feature_names_out()
        ranked = sorted(zip(feature_names, values[0]), key=lambda item: abs(item[1]), reverse=True)[:top_n]
        return [{"feature": name.replace("sensors__", "").replace("machine_type__", "Type: "), "shap_value": float(value)} for name, value in ranked]
    except Exception:
        return []
