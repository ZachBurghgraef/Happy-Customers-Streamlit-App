import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import pickle

# data imports
import matplotlib.pyplot as plt

# ML imports
import shap

# -----------------------------------------------------------------------

# Globals

SMALLPLOT_WIDTH = 500
TEXT_WIDTH = 700
model = None
prediction = None

# -----------------------------------------------------------------------

# Helpful defs

def wide_centered_layout():
    with st.container(horizontal_alignment="center"):
        return st.container(
            width=2 * SMALLPLOT_WIDTH + 16, horizontal_alignment="center"
        )
    
@st.cache_resource
def load_model_file(filepath: str):
    """Loads a single model file and its corresponding metadata file."""
    ext = filepath.split(".")[-1]
    match ext:
        case "joblib":
            model_obj = joblib.load(filepath)
            try:
                metadata = joblib.load(filepath.split(".joblib")[0] + "_metadata.joblib")
            except Exception:
                metadata = {"score": -1, "info": "No metadata on selected model"}
            return model_obj, metadata

        case "pkl":
            with open(filepath, 'rb') as file:
                model_obj = pickle.load(file)
            try:
                with open(filepath.split(".pkl")[0] + "_metadata.pkl", 'rb') as file:
                    metadata = pickle.load(file)
            except Exception:
                metadata = {"score": -1, "info": "No metadata on selected model"}
            return model_obj, metadata

        case _:
            raise ValueError(f"Unable to handle file extension: {ext}")

@st.cache_resource
def get_available_models_sorted(model_folder: str):
    """
    Scans the model folder, loads metadata for all available models,
    and returns a list of model filenames sorted by:
    1. Highest 'score' (descending)
    2. Lowest 'CV_SD' (ascending, to break ties)
    """
    path = Path(model_folder)
    model_files = [f.name for f in path.iterdir() if f.is_file() and "_metadata" not in f.name]
    
    scored_models = []
    for m_file in model_files:
        filepath = str(path / m_file)
        _, meta = load_model_file(filepath)
        
        # Safely extract score (higher is better) and CV_SD (lower is better)
        if isinstance(meta, dict):
            score = meta.get("score", float("-inf"))
            cv_sd = meta.get("CV_SD", float("inf"))
        else:
            score = float("-inf")
            cv_sd = float("inf")
            
        scored_models.append((m_file, score, cv_sd))
    
    # Sort key: (-score, cv_sd) places highest score first, then lowest CV_SD first
    scored_models.sort(key=lambda x: (-x[1], x[2]))
    
    return [m[0] for m in scored_models]

# Caching the data gathering side of shap
@st.cache_data
def _calculate_shap_values(_treeModel, dataInput):
    explainer = shap.TreeExplainer(_treeModel)
    shap_explanation = explainer(dataInput)
    return shap_explanation

def explain_tree_model(treeModel, dataInput, targetClass = None, title = "Key Drivers Behind This Prediction"):

    shap_explanation = _calculate_shap_values(treeModel, dataInput)

    if len(shap_explanation.shape) == 3:
        if targetClass is None:
            # Default to explaining the class the model actually predicted
            predicted_class = model.predict(dataInput)[0]
            single_shap_values = shap_explanation[0, :, predicted_class]
        else:
            single_shap_values = shap_explanation[0, :, targetClass]
    else:
        # Regression models yield a standard 2D shape (samples, features)
        single_shap_values = shap_explanation[0]
        
    # Visualize the contribution using a Waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(single_shap_values, show=False)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()

    # Render the plot in your Streamlit UI
    st.pyplot(fig)
    
    # Clean up the matplotlib memory buffer
    plt.close(fig)

    return single_shap_values

def explain_cohort_dependence(treeModel, test_data, feature_x, feature_y):
    """Generates a SHAP dependence plot to show cohort interactions."""
    explainer = shap.TreeExplainer(treeModel)
    shap_values_all = explainer.shap_values(test_data)
    
    # Handle multi-class output tracking (extract array for Happy/Class 1)
    if isinstance(shap_values_all, list) and len(shap_values_all) == 2:
        shap_disp = shap_values_all[1]
    elif len(shap_values_all.shape) == 3:
        shap_disp = shap_values_all[:, :, 1]
    else:
        shap_disp = shap_values_all

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature_x, 
        shap_disp, 
        test_data, 
        interaction_index=feature_y, 
        ax=ax, 
        show=False
    )
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Risk Threshold')
    plt.title(f"Feature Interaction Analysis: {feature_x} vs. {feature_y}", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------------------

# Draw app

with wide_centered_layout():
    with st.container(width=TEXT_WIDTH):
        st.title("Customer Happiness Intelligence")

        st.caption("AI-powered insights to understand, predict, and elevate your customer experience.")

        st.space()

        modelFolder = r"src/models/saved_models/"
        availableModels = get_available_models_sorted(modelFolder)
        
        if not availableModels:
            st.warning("⚠️ **No trained models found.** Please ensure your model files are saved in the models directory.")
        else:
            # Model selection and technical specs placed inside the insight dropdown
            with st.expander("🔍 Model Selection & Performance Insights", expanded=False):
                # Highest accuracy / lowest CV_SD model is index 0 and set as default
                modelSelected = st.segmented_control(
                    "**Choose AI Model Engine**",
                    availableModels,
                    default=availableModels[0],
                )
                
                if modelSelected is not None:
                    model, metadata = load_model_file(modelFolder + modelSelected)
                    
                    st.markdown("---")
                    st.markdown("### Model Architecture")
                    st.write(model)
                    st.markdown("### Performance Metrics & Metadata")
                    st.write(metadata)

        # Create tabs to split Single Inference and Cohort Data Discovery
        tab1, tab2 = st.tabs(["👤 Predict Single Customer", "🎯 Cohort Risk Explorer"])

        with tab1:
            st.markdown("### Customer Experience Survey")
            st.markdown(
                "Adjust the scores below based on customer feedback (*1 = Strongly Disagree* to *5 = Strongly Agree*)."
            )

            inputs = {}
            X1 = st.slider(
                "1. Timely Delivery",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1 = Very Late, 5 = On Time / Early",
            )
            inputs["X1"] = X1

            X2 = st.slider(
                "2. Order Accuracy",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1 = Wrong Items, 5 = Exactly as Expected",
            )
            inputs["X2"] = X2

            X3 = st.slider(
                "3. Item Availability",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1 = Missing Items, 5 = Everything Available",
            )
            inputs["X3"] = X3

            X4 = st.slider(
                "4. Value & Pricing",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1 = Overpriced, 5 = Great Value",
            )
            inputs["X4"] = X4

            X5 = st.slider(
                "5. Courier & Hand-Off Satisfaction",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1 = Poor Service, 5 = Excellent Service",
            )
            inputs["X5"] = X5

            X6 = st.slider(
                "6. App & Ordering Experience",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1 = Difficult to Use, 5 = Effortless",
            )
            inputs["X6"] = X6
        
            # Prediction zone
            if model is not None:
                features = metadata.get("features", ["X1", "X2", "X3", "X4", "X5", "X6"]) if isinstance(metadata, dict) else ["X1", "X2", "X3", "X4", "X5", "X6"]
                predictionInputs = [inputs[f] for f in features if f in inputs]
                predictionInputs = pd.DataFrame([predictionInputs], columns=features)

                st.markdown("---")
                st.subheader("Predicted Sentiment Outcome")

                prediction = model.predict(predictionInputs)

                if prediction[0] == 1:
                    st.success("🎉 **Satisfied Customer** — High likelihood of a positive experience.")
                else:
                    st.error("⚠️ **At-Risk Customer** — High risk of dissatisfaction or churn.")

                st.space()

                explain_tree_model(
                    model, 
                    predictionInputs, 
                    1, 
                    title=f"Feature Impact Analysis ({modelSelected})"
                )

        with tab2:
            st.subheader("Analyze & Isolate At-Risk Customer Segments")
            st.markdown(
                "Upload batch survey data to identify key friction points and generate targeted outreach lists for at-risk customers."
            )
            
            uploaded_file = st.file_uploader(
                "Upload Customer Survey Data (CSV)", 
                type=["csv"],
                help="Upload a CSV containing ratings for columns X1 through X6."
            )
            
            if uploaded_file is not None and model is not None:
                # Load up data asset
                batch_df = pd.read_csv(uploaded_file)
                features_list = metadata.get("features", ["X1", "X2", "X3", "X4", "X5", "X6"]) if isinstance(metadata, dict) else ["X1", "X2", "X3", "X4", "X5", "X6"]
                
                # Check for feature conformity
                if not all(col in batch_df.columns for col in features_list):
                    st.error(f"⚠️ **Incompatible CSV Format:** The uploaded file must include the following column headers: `{features_list}`")
                else:
                    # Clean input space for prediction evaluation
                    eval_df = batch_df[features_list].dropna()
                    
                    st.markdown("### 1. Feature Interaction Explorer")
                    col1, col2 = st.columns(2)
                    with col1:
                        feat_x = st.selectbox("Primary Factor (X-Axis)", features_list, index=0)
                    with col2:
                        feat_y = st.selectbox("Secondary Comparison Factor (Color)", features_list, index=4 if len(features_list) > 4 else 1)
                    
                    # Generate dependencies
                    explain_cohort_dependence(model, eval_df, feat_x, feat_y)
                    
                    st.markdown("### 2. Isolate & Export High-Risk Cohort")
                    threshold = st.slider(
                        "Dissatisfaction Sensitivity Threshold", 
                        0.50, 0.95, 0.73, step=0.01,
                        help="Higher thresholds narrow down the list to only the most severely dissatisfied customers."
                    )
                    
                    # Compute probabilities [:, 0] tracks likelihood of being Unhappy (Class 0)
                    probabilities = model.predict_proba(eval_df)[:, 0]
                    
                    # Isolate target cohort rows
                    at_risk_indices = np.where(probabilities >= threshold)[0]
                    at_risk_cohort = batch_df.iloc[at_risk_indices].copy()
                    at_risk_cohort["Unhappy_Probability"] = probabilities[at_risk_indices]
                    
                    st.metric(label="Total Flagged Customers At Risk", value=len(at_risk_cohort))
                    
                    if len(at_risk_cohort) > 0:
                        st.dataframe(at_risk_cohort, use_container_width=True)
                        
                        # Direct downloadable utility for user business systems
                        csv_data = at_risk_cohort.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Export At-Risk Cohort (CSV)",
                            data=csv_data,
                            file_name="at_risk_customer_cohort.csv",
                            mime="text/csv",
                        )