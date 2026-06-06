"""
NEC Surgical Risk Prediction System
CatBoost + SHAP interpretability analysis
Usage: streamlit run streamlit_app.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import catboost as cb
import shap
import matplotlib.pyplot as plt

# ==================== Page Config ====================
st.set_page_config(
    page_title="NEC Surgical Risk Prediction",
    page_icon=":hospital:",
    layout="wide",
)

# ==================== Plotting Support ====================
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== File Paths ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "catboost_model.cbm")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "catboost_feature_names.pkl")
SHAP_PATH = os.path.join(BASE_DIR, "catboost_shap_result.pkl")


# ==================== Load Model & Data (Cached) ====================
@st.cache_resource
def load_model_and_data():
    """Load CatBoost model, feature names, and SHAP results."""
    try:
        model = cb.CatBoostClassifier()
        model.load_model(MODEL_PATH)

        with open(FEATURE_NAMES_PATH, "rb") as f:
            feature_names = pickle.load(f)

        with open(SHAP_PATH, "rb") as f:
            shap_result = pickle.load(f)

        return model, feature_names, shap_result
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e.filename}")
        return None, None, None
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None


model, feature_names, shap_result = load_model_and_data()

# ==================== UI ====================
st.title("Necrotizing Enterocolitis (NEC) Surgical Risk Prediction Model")
st.markdown("#### CatBoost Machine Learning + SHAP Interpretability")
st.markdown("---")

# ---------- Sidebar: Raw Value Inputs ----------
with st.sidebar:
    st.header("Patient Characteristics")

    raw_inputs = {}
    raw_labels = {
        "CRP": "C-Reactive Protein CRP (mg/L)",
        "RBC": "Red Blood Cell Count RBC (x10^12/L)",
        "sbp": "Systolic Blood Pressure SBP (mmHg)",
        "dbp": "Diastolic Blood Pressure DBP (mmHg)",
        "neutrophil": "Neutrophil Count (x10^9/L)",
        "lymphocyte": "Lymphocyte Count Lymph_Count (x10^9/L)",
        "platelet": "Platelet Count (x10^9/L)",
        "Preterm_baby": "Preterm Infant",
        "asphyxia": "History of Asphyxia",
        "sepsis": "Sepsis",
    }

    # Continuous variables
    raw_inputs["CRP"] = st.number_input(raw_labels["CRP"], value=0.0, step=0.01, format="%.2f")
    raw_inputs["RBC"] = st.number_input(raw_labels["RBC"], value=0.0, step=0.01, format="%.2f")
    raw_inputs["sbp"] = st.number_input(raw_labels["sbp"], value=0.0, step=1.0, format="%.0f")
    raw_inputs["dbp"] = st.number_input(raw_labels["dbp"], value=0.0, step=1.0, format="%.0f")
    raw_inputs["neutrophil"] = st.number_input(raw_labels["neutrophil"], value=0.0, step=0.01, format="%.2f")
    raw_inputs["lymphocyte"] = st.number_input(raw_labels["lymphocyte"], value=0.0, step=0.01, format="%.2f")
    raw_inputs["platelet"] = st.number_input(raw_labels["platelet"], value=0.0, step=1.0, format="%.0f")

    # Binary variables
    for feat in ("Preterm_baby", "asphyxia", "sepsis"):
        raw_inputs[feat] = st.selectbox(
            raw_labels[feat],
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key=feat,
        )

    st.markdown("---")
    predict_btn = st.button("Predict Risk", type="primary", use_container_width=True)

# ==================== Compute Derived Ratios ====================
lymph = raw_inputs["lymphocyte"]
neutrophil = raw_inputs["neutrophil"]
platelet = raw_inputs["platelet"]

nlr = neutrophil / lymph if lymph != 0 else 0.0
plr = platelet / lymph if lymph != 0 else 0.0

# Build the final model input DataFrame
model_input = {
    "CRP": raw_inputs["CRP"],
    "RBC": raw_inputs["RBC"],
    "sbp": raw_inputs["sbp"],
    "dbp": raw_inputs["dbp"],
    "NLR": nlr,
    "Lymph_Count": lymph,
    "PLR": plr,
    "Preterm_baby": raw_inputs["Preterm_baby"],
    "asphyxia": raw_inputs["asphyxia"],
    "sepsis": raw_inputs["sepsis"],
}
input_df = pd.DataFrame([model_input])

# ==================== Prediction Logic ====================
if predict_btn and model is not None:
    pred_prob = model.predict_proba(input_df)[0]
    pred_label = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        risk_text = "High Risk" if pred_label == 1 else "Low Risk"
        st.metric("Risk Level", risk_text)
    with col2:
        st.metric("Surgical Risk Probability", f"{pred_prob[1]:.2%}")

    st.progress(float(pred_prob[1]), text=f"Risk Score: {pred_prob[1]:.1%}")

    # ---- Derived Ratios Display ----
    st.markdown("---")
    st.subheader("Derived Ratios (Computed from Raw Inputs)")
    ratio_df = pd.DataFrame({
        "Ratio": ["NLR", "PLR"],
        "Formula": [
            "Neutrophil Count / Lymphocyte Count",
            "Platelet Count / Lymphocyte Count",
        ],
        "Raw Values": [
            f"{neutrophil:.2f} / {lymph:.2f}",
            f"{platelet:.0f} / {lymph:.2f}",
        ],
        "Calculated Value": [f"{nlr:.2f}", f"{plr:.2f}"],
    })
    st.table(ratio_df)

    st.markdown("---")

    # ==================== SHAP Interpretability ====================
    st.subheader("SHAP Feature Contribution Analysis")

    explainer = shap.TreeExplainer(model)
    shap_values_single = explainer.shap_values(input_df)

    # Waterfall plot for this patient
    st.markdown("#### Patient-Specific Feature Contributions (Waterfall)")
    fig_wf, ax_wf = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_single[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=feature_names,
        ),
        show=False,
    )
    st.pyplot(fig_wf)

    # Global feature importance bar
    st.markdown("---")
    st.markdown("#### Global Feature Importance (SHAP Mean |Value|)")

    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_result["shap_values"],
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    st.pyplot(fig_bar)

    # SHAP summary beeswarm
    st.markdown("#### SHAP Summary Plot (All Samples)")
    fig_summary, ax_summary = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_result["shap_values"],
        feature_names=feature_names,
        show=False,
    )
    st.pyplot(fig_summary)

elif predict_btn and model is None:
    st.error("Model not loaded. Prediction unavailable. Please check model files.")

# ==================== Footer ====================
st.markdown("---")
st.markdown(
    "**Disclaimer**: This model is for clinical decision support only. "
    "Final clinical decisions should be made by qualified healthcare professionals."
)
