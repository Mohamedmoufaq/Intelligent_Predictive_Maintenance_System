"""Streamlit dashboard for the Intelligent Predictive Maintenance System."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Src.explainability import local_shap_explanation
from Src.predict import load_artifacts, predict_machine


st.set_page_config(page_title="Predictive Maintenance", page_icon=":material/precision_manufacturing:", layout="wide")
st.title("Intelligent Predictive Maintenance System")
st.caption("AI4I 2020 dataset · Costs and decision thresholds are demonstrative project assumptions.")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    artifacts = get_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.code("python -m Src.train_model")
    st.stop()

with st.sidebar:
    st.header("Machine sensor readings")
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=0)
    air_temperature = st.number_input("Air Temperature (K)", min_value=250.0, max_value=400.0, value=300.0, step=0.1)
    process_temperature = st.number_input("Process Temperature (K)", min_value=250.0, max_value=450.0, value=310.0, step=0.1)
    rotational_speed = st.number_input("Rotational Speed (rpm)", min_value=0, max_value=10_000, value=1_500, step=1)
    torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=200.0, value=40.0, step=0.1)
    tool_wear = st.number_input("Tool Wear (min)", min_value=0, max_value=1_000, value=100, step=1)
    run_prediction = st.button(
        "Analyze machine",
        icon=":material/analytics:",
        type="primary",
        width="stretch",
    )

if run_prediction:
    try:
        machine = {
            "Type": machine_type,
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }
        result = predict_machine(machine, artifacts)
        failure_text = "YES — warning" if result["machine_failure"] else "NO — below warning threshold"
        risk_colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "red"}

        st.subheader("Machine health result")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Machine failure", failure_text, border=True)
        metric_columns[1].metric("Failure probability", f"{result['failure_probability']:.1%}", border=True)
        metric_columns[2].metric("Risk score", f"{result['risk_score']:.1f} / 100", border=True)
        metric_columns[3].metric("Risk level", result["risk_level"], border=True)
        st.badge(result["risk_level"], color=risk_colors[result["risk_level"]])

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.subheader("Maintenance decision")
                st.write(f"**Likely failure type:** {result['likely_failure_type']}")
                st.write(f"**Recommendation:** {result['maintenance_recommendation']}")
                st.caption(f"Model decision threshold: {result['decision_threshold']:.0%}")
        with right:
            with st.container(border=True):
                st.subheader("Example cost estimate")
                st.caption("Demonstrative / project assumptions — not real company costs.")
                st.write(f"Potential failure cost: ₹{result['potential_failure_cost']:,.0f}")
                st.write(f"Expected failure loss: ₹{result['expected_failure_loss']:,.0f}")
                st.write(f"Preventive maintenance cost: ₹{result['preventive_maintenance_cost']:,.0f}")
                st.info(result["business_decision"], icon=":material/account_balance:")

        with st.container(border=True):
            st.subheader("Main contributing factors")
            explanation = local_shap_explanation(artifacts["failure_model"], pd.DataFrame([machine]))
            if explanation:
                for item in explanation:
                    direction = "increased" if item["shap_value"] > 0 else "reduced"
                    st.write(f"- **{item['feature']}** {direction} predicted failure risk (SHAP: {item['shap_value']:+.3f}).")
            else:
                st.caption("A local SHAP explanation is unavailable for the selected model. No explanation is displayed rather than generating a fake one.")
    except (ValueError, KeyError, OSError) as error:
        st.error(f"Unable to analyze this machine: {error}", icon=":material/error:")
else:
    st.info("Enter sensor readings in the sidebar and select **Analyze machine**.", icon=":material/info:")
