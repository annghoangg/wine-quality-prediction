import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
from streamlit_shap import st_shap

st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

st.title("Wine Quality Prediction")
st.markdown("A simple model to predict wine quality based on chemical properties.")

# Initialize session state for prediction history
if 'history' not in st.session_state:
    st.session_state.history = []

quality_descriptions = {
    3: "Very Poor: Noticeable flaws, highly unbalanced, or faulty.",
    4: "Poor: Unbalanced, lacks character, or has minor flaws.",
    5: "Fair: Average commercial wine, acceptable but unexciting.",
    6: "Good: Well-made, balanced, and enjoyable everyday wine.",
    7: "Very Good: High quality, complex, and highly enjoyable.",
    8: "Excellent: Outstanding complexity, balance, and character.",
    9: "Exceptional: World-class, flawless, phenomenal wine."
}
# Add simple styling
st.markdown(
    """
    <style>
    .result-box {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .high-quality { color: #28a745; font-weight: bold; font-size: 24px; }
    .med-quality { color: #ffc107; font-weight: bold; font-size: 24px; }
    .low-quality { color: #dc3545; font-weight: bold; font-size: 24px; }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_models():
    base_path = "models"
    try:
        imputer = joblib.load(os.path.join(base_path, 'imputer.joblib'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.joblib'))
        label_encoder = joblib.load(os.path.join(base_path, 'label_encoder.joblib'))
        top_features = joblib.load(os.path.join(base_path, 'top_features.joblib'))
        model = joblib.load(os.path.join(base_path, 'best_model.joblib'))
        
        # Load best base models in case blending is used and it somehow needs them
        # (Usually joblib.dump on blending saves everything, but just to be safe)
        try:
            with open(os.path.join(base_path, 'metadata.txt'), 'r') as f:
                model_name = f.read().strip()
        except:
            model_name = "Unknown Model"
            
        # Explainer for XAI
        best_et = joblib.load(os.path.join(base_path, 'best_et.joblib'))
        explainer = shap.TreeExplainer(best_et)
            
        return imputer, scaler, label_encoder, top_features, model, model_name, explainer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None, None

def create_safe_wine_features(df_input):
    df_engineered = df_input.copy()
    
    df_engineered['acid ratio'] = df_engineered['fixed acidity'] / (df_engineered['volatile acidity'] + 1e-8)
    df_engineered['sulfur ratio'] = df_engineered['free sulfur dioxide'] / (df_engineered['total sulfur dioxide'] + 1e-8)
    df_engineered['alcohol sugar ratio'] = df_engineered['alcohol'] / (df_engineered['residual sugar'] + 1e-8)
    
    df_engineered['density alcohol interaction'] = df_engineered['density'] * df_engineered['alcohol']
    df_engineered['sulphates alcohol'] = df_engineered['sulphates'] * df_engineered['alcohol']
    
    df_engineered['alcohol squared'] = df_engineered['alcohol'] ** 2
    df_engineered['volatile acidity squared'] = df_engineered['volatile acidity'] ** 2
    
    df_engineered['log volatile acidity'] = np.log1p(df_engineered['volatile acidity'])
    df_engineered['log residual sugar'] = np.log1p(df_engineered['residual sugar'])
    df_engineered['log chlorides'] = np.log1p(df_engineered['chlorides'])
    
    df_engineered['chemical_balance_index'] = (
        df_engineered['alcohol'] * 0.35 +
        (1 / (df_engineered['volatile acidity'] + 1e-8)) * 0.20 +
        df_engineered['sulphates'] * 0.20 +
        df_engineered['citric acid'] * 0.15 +
        (1 / (df_engineered['chlorides'] + 1e-8)) * 0.10
    )
    
    if 'type' in df_engineered.columns:
        df_engineered['type'] = df_engineered['type'].map({'white': 0, 'red': 1})
        
    return df_engineered

imputer, scaler, label_encoder, top_features, model, model_name, explainer = load_models()

if model is not None:
    st.sidebar.header("Wine Characteristics")
    
    # Input fields
    wine_type = st.sidebar.selectbox("Type", ["white", "red"])
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 3.0, 16.0, 7.0, 0.1)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.0, 2.0, 0.3, 0.01)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 2.0, 0.3, 0.01)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.0, 70.0, 5.0, 0.1)
    chlorides = st.sidebar.slider("Chlorides", 0.0, 0.6, 0.05, 0.001)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 0.0, 300.0, 30.0, 1.0)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 0.0, 450.0, 115.0, 1.0)
    density = st.sidebar.slider("Density", 0.98, 1.04, 0.99, 0.001)
    pH = st.sidebar.slider("pH", 2.5, 4.5, 3.2, 0.01)
    sulphates = st.sidebar.slider("Sulphates", 0.0, 2.0, 0.5, 0.01)
    alcohol = st.sidebar.slider("Alcohol (%)", 7.0, 15.0, 10.5, 0.1)
    
    input_data = pd.DataFrame({
        'type': [wine_type],
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })
    
    st.write("### Input Data")
    st.dataframe(input_data)
    
    if st.button("Predict Quality", type="primary"):
        with st.spinner("Processing..."):
            # Feature engineering
            engineered_data = create_safe_wine_features(input_data)
            
            # Select only top features from training
            X_selected = engineered_data[top_features]
            
            # Impute and scale
            X_imputed = imputer.transform(X_selected)
            X_scaled = scaler.transform(X_imputed)
            
            # Predict
            if model_name == "Blending":
                # For blending, we need to get predict_proba from base models
                try:
                    best_rf = joblib.load(os.path.join("models", 'best_rf.joblib'))
                    best_et = joblib.load(os.path.join("models", 'best_et.joblib'))
                    best_xgb = joblib.load(os.path.join("models", 'best_xgb.joblib'))
                    
                    rf_proba = best_rf.predict_proba(X_scaled)
                    et_proba = best_et.predict_proba(X_scaled)
                    xgb_proba = best_xgb.predict_proba(X_scaled)
                    
                    blend_X = np.hstack((rf_proba, et_proba, xgb_proba))
                    pred_encoded = model.predict(blend_X)
                except Exception as e:
                    st.error(f"Error in Blending prediction: {e}")
                    pred_encoded = None
            else:
                pred_encoded = model.predict(X_scaled)
                
            if pred_encoded is not None:
                # Decode prediction
                predicted_quality = label_encoder.inverse_transform(pred_encoded)[0]
                
                # Determine styling
                if predicted_quality >= 7:
                    q_class = "high-quality"
                elif predicted_quality <= 4:
                    q_class = "low-quality"
                else:
                    q_class = "med-quality"
                
                st.markdown(f'<div class="result-box"><span style="font-size: 18px;">Predicted Quality Rating:</span><br><span class="{q_class}">{predicted_quality}</span> / 9</div>', unsafe_allow_html=True)
                
                explanation = quality_descriptions.get(predicted_quality, "Unknown rating.")
                st.success(f"**What this means:** {explanation}")
                
                st.info(f"Using Model: {model_name}")
                
                # --- XAI Block ---
                st.markdown("---")
                st.markdown("### Explainable AI (XAI)")
                st.markdown("Here is how each feature contributed to this specific prediction according to our strongest base model (Extra Trees):")
                
                # Calculate SHAP values
                shap_values = explainer(X_scaled)
                
                # shap_values shape for multiclass XGB is (samples, features, classes)
                # We extract the explanation for the predicted class
                prediction_idx = int(pred_encoded[0])
                shap_explanation = shap_values[0, :, prediction_idx]
                
                # Give proper feature names to the Explanation object
                shap_explanation.feature_names = top_features
                
                st_shap(shap.plots.waterfall(shap_explanation), height=500)
                
                # Save to history
                history_entry = input_data.copy()
                history_entry['Predicted Quality'] = predicted_quality
                st.session_state.history.append(history_entry)

# Display Prediction History
if st.session_state.history:
    st.markdown("---")
    st.write("### Prediction History")
    history_df = pd.concat(st.session_state.history, ignore_index=True)
    st.dataframe(history_df)
