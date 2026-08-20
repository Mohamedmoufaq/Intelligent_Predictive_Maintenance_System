# Intelligent Predictive Maintenance System

An educational predictive-maintenance project built with the AI4I 2020 Predictive Maintenance Dataset. The system predicts machine-failure risk, identifies a likely failure type when a failure is predicted, provides local SHAP-based explanations when supported, assigns a risk level, and recommends a maintenance action.

## Project Overview

The project combines a reproducible machine-learning pipeline with a Streamlit dashboard. It accepts machine type and sensor readings, estimates the probability of machine failure, applies the saved decision threshold, and presents the result as a risk and maintenance recommendation.

## Problem Statement

Unexpected equipment failure can interrupt operations and make maintenance reactive. This project explores how machine-learning predictions from machine and sensor measurements can help identify elevated failure risk before a maintenance decision is made.

## Objectives

- Clean and prepare the AI4I machine data for repeatable modelling.
- Compare several classification approaches using recall-focused evaluation.
- Select and save a failure-prediction model with an explicit decision threshold.
- Present predictions with risk, maintenance guidance, and local explanations.
- Demonstrate how model outputs can be connected to example cost estimates without presenting them as real financial results.

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**. The failure model uses these machine and sensor features:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

The dataset also contains `Machine failure` and failure-indicator columns including `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`.

## Machine Failure Target

The `Machine failure` column is a binary target:

- `0` = No Failure
- `1` = Failure

## Project Workflow

```text
Data → Cleaning → Preprocessing → Model Training → Model Comparison → XGBoost
	→ Threshold → Prediction → SHAP → Risk → Maintenance Decision
```

## Machine Learning Models

The project compares:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Model Selection

The saved production model is XGBoost. It was selected using recall-focused F2 evaluation because missed failures can be costly in a maintenance context.

Verified results from the recorded evaluation output:

- **Recall:** `0.847`
- **F2:** `0.774`
- **ROC-AUC:** `0.972`
- **Decision threshold:** `0.55`

The threshold is applied to the predicted failure probability: a probability at or above `0.55` is treated as a predicted failure.

## Explainability

The project can generate local SHAP factors for supported tree-based failure models. These factors show which transformed input features had the strongest positive or negative contribution for an individual prediction. If a reliable explanation is unavailable, the application displays no explanation rather than inventing one.

## Risk & Maintenance Decision

The predicted failure probability is converted into a presentation-friendly risk score from `0` to `100` and a risk level: `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`.

The maintenance recommendation uses the saved threshold and risk rules:

- Probability at least `0.55`, or `CRITICAL` risk → **Immediate Maintenance**
- `HIGH` risk below the threshold → **Prepare Maintenance**
- `MEDIUM` risk → **Schedule Maintenance**
- `LOW` risk → **Continue Monitoring**

## Cost Estimation

The dashboard displays an illustrative expected-failure-loss calculation based on the predicted probability and configurable assumptions. The current documented assumptions include:

- **₹120,000** potential failure cost
- **₹8,000** preventive maintenance cost
- **₹2,000** unnecessary maintenance cost

These are demonstrative project assumptions, not real company costs, measured savings, or deployment results.

## Streamlit Dashboard

The Streamlit dashboard accepts machine type and sensor readings and displays:

- Failure status and probability
- Risk score and risk level
- Likely failure type
- Maintenance recommendation
- Demonstrative cost estimates
- Available local SHAP factors

## Project Structure

```text
.
├── App/
│   └── app.py
├── data/
│   ├── ai4i2020.csv
│   └── raw/
├── Dataset/
│   └── ai4i2020.csv
├── Documents/
│   ├── Interview_QA.docx
│   ├── LinkedIn_Post.docx
│   └── Resume_Project.txt
├── Images/
├── Models/
│   ├── decision_tree.pkl
│   ├── failure_model.joblib
│   ├── failure_type_model.joblib
│   ├── logistic_model.pkl
│   ├── model_metadata.joblib
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── Notebooks/
├── Output/
│   ├── cleaned_data.csv
│   ├── evaluation_results.csv
│   ├── feature_importance.csv
│   └── predictions.csv
├── Presentation/
│   └── Project_Presentation.pptx
├── Reports/
│   ├── Final_Report.pdf
│   └── Project_Report.docx
├── Requirements/
│   └── requirements.txt
├── Src/
│   ├── data_loader.py
│   ├── explainability.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── utils.py
├── tests/
│   └── test_prediction.py
└── README.md
```

## Installation

From the project root, install the verified dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Requirements\requirements.txt
```

## Run Training

```powershell
.\.venv\Scripts\python.exe -m Src.train_model
```

Training cleans the dataset, compares the configured models, records evaluation results, selects the production model, and saves the model artifacts.

## Run Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run App\app.py
```

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prediction
```

## Output Files

The `Output/` folder contains:

- `cleaned_data.csv` — cleaned project data.
- `evaluation_results.csv` — recorded comparison metrics and thresholds for the candidate models.
- `feature_importance.csv` — generated feature-importance output for the trained model.
- `predictions.csv` — generated prediction output from the current prediction pipeline and project data.

## Screenshots

The `Images/` folder contains the available dashboard screenshots and project charts, including:

- `dashboard_high_risk.png`
- `dashboard_low_risk.png`
- `production_confusion_matrix.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `bar_chart.png`
- `heatmap.png`

## Limitations

- The AI4I 2020 data is a benchmark dataset and does not establish performance for a particular company or plant.
- The recorded metrics depend on the project’s data split, preprocessing, model configuration, and threshold-selection procedure.
- Cost values are illustrative assumptions and are not validated financial outcomes.
- Local SHAP explanations are conditional on model and runtime support.
- A model prediction is decision support and does not replace maintenance expertise or inspection.

## Future Enhancements

Potential future work includes validation on newly collected machine data, probability calibration, threshold tuning with verified maintenance costs, model-drift monitoring, broader failure-type coverage, and integration with a verified maintenance workflow.

## Author

**Author:** [MOHAMED MOUFAQ S]
