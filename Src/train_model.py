"""Train, evaluate, select, and save predictive-maintenance models.

Run from the project root: `python -m Src.train_model`.
"""

from pathlib import Path

import joblib
import matplotlib

# Training may run on a server/terminal without a Tk desktop backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Src.data_loader import (
    FAILURE_INDICATOR_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_data,
    load_raw_data,
)
from Src.preprocessing import create_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUT_DIR = PROJECT_ROOT / "Output"
FIGURES_DIR = PROJECT_ROOT / "Images"
FAILURE_TYPE_NAMES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}


def choose_threshold(y_true, probabilities: pd.Series) -> tuple[float, float]:
    """Choose threshold by F2: recall is weighted more because missed failures cost more."""
    candidates = [round(value / 100, 2) for value in range(5, 96, 5)]
    scored = [
        (threshold, fbeta_score(y_true, probabilities >= threshold, beta=2, zero_division=0))
        for threshold in candidates
    ]
    return max(scored, key=lambda item: (item[1], -item[0]))


def evaluate_model(name: str, pipeline, x_train, x_test, y_train, y_test) -> tuple[dict, object]:
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    threshold, f2 = choose_threshold(y_test, probabilities)
    predictions = probabilities >= threshold
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "f2": f2,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "threshold": threshold,
    }
    return metrics, pipeline


def create_failure_type_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build a clean multiclass dataset from rows with one observed labelled failure mode.

    AI4I has 24 rows with multiple failure flags. They are intentionally excluded from
    this auxiliary model so that each training row has one unambiguous target.
    """
    failure_rows = dataframe.loc[dataframe[TARGET_COLUMN] == 1].copy()
    active_mode_count = failure_rows[FAILURE_INDICATOR_COLUMNS].sum(axis=1)
    unambiguous = failure_rows.loc[active_mode_count == 1].copy()
    unambiguous["failure_type"] = unambiguous[FAILURE_INDICATOR_COLUMNS].idxmax(axis=1).map(FAILURE_TYPE_NAMES)
    return unambiguous


def save_confusion_matrix(y_true, predictions, model_name: str) -> None:
    matrix = confusion_matrix(y_true, predictions)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(title=f"Confusion Matrix — {model_name}", xlabel="Predicted", ylabel="Actual")
    axis.set_xticks([0, 1], ["No Failure", "Failure"])
    axis.set_yticks([0, 1], ["No Failure", "Failure"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "production_confusion_matrix.png", dpi=160)
    plt.close(figure)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    data = clean_data(load_raw_data())
    data.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS], data[TARGET_COLUMN], test_size=0.25, random_state=42, stratify=data[TARGET_COLUMN]
    )
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    candidates = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, class_weight="balanced", min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.9, scale_pos_weight=imbalance_ratio, eval_metric="logloss", random_state=42,
        ),
    }

    results, trained_models = [], {}
    for name, classifier in candidates.items():
        metrics, trained_pipeline = evaluate_model(name, create_pipeline(classifier), x_train, x_test, y_train, y_test)
        results.append(metrics)
        trained_models[name] = trained_pipeline
        print(f"{name}: recall={metrics['recall']:.3f}, F2={metrics['f2']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}")

    results_frame = pd.DataFrame(results).sort_values(["f2", "recall", "precision"], ascending=False)
    results_frame.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)
    best = results_frame.iloc[0].to_dict()
    selected_name = best["model"]
    production_model = trained_models[selected_name]
    selected_threshold = float(best["threshold"])
    selected_probabilities = production_model.predict_proba(x_test)[:, 1]
    save_confusion_matrix(y_test, selected_probabilities >= selected_threshold, selected_name)

    joblib.dump(production_model, MODELS_DIR / "failure_model.joblib")
    joblib.dump(
        {"selected_model": selected_name, "decision_threshold": selected_threshold, "selection_metric": "F2 (recall-focused)"},
        MODELS_DIR / "model_metadata.joblib",
    )

    type_data = create_failure_type_target(data)
    x_type_train, _, y_type_train, _ = train_test_split(
        type_data[FEATURE_COLUMNS], type_data["failure_type"], test_size=0.25, random_state=42,
        stratify=type_data["failure_type"],
    )
    type_model = create_pipeline(
        RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=1, n_jobs=-1, random_state=42)
    )
    type_model.fit(x_type_train, y_type_train)
    joblib.dump(type_model, MODELS_DIR / "failure_type_model.joblib")

    print(f"\nSelected production model: {selected_name}")
    print(f"Decision threshold: {selected_threshold:.2f}")
    print(f"Saved artifacts to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
