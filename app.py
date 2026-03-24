import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import datetime
from streamlit_shap import st_shap

# --- Page Config ---
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon=":wine_glass:",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
<style>
    :root {
        --wine-dark: #4A0E2E;
        --wine-mid: #722F37;
        --wine-light: #A0455A;
        --gold: #C9A84C;
        --cream: #F5F0EB;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        color: var(--wine-dark);
        font-size: 2.2rem;
        margin-bottom: 0;
    }
    .main-header p {
        color: var(--wine-mid);
        font-size: 1.05rem;
        margin-top: 0.2rem;
    }

    .result-box {
        padding: 24px;
        background: linear-gradient(135deg, #f8f0f4 0%, #f0e4ea 100%);
        border-radius: 14px;
        text-align: center;
        margin-top: 16px;
        border: 1px solid #e0c8d4;
        box-shadow: 0 2px 8px rgba(74, 14, 46, 0.08);
    }
    .high-quality { color: #1a7a3a; font-weight: bold; font-size: 28px; }
    .med-quality { color: #b8860b; font-weight: bold; font-size: 28px; }
    .low-quality { color: #c0392b; font-weight: bold; font-size: 28px; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fdf8fa 0%, #f5eef1 100%);
    }

    .feedback-success {
        padding: 16px;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 10px;
        border: 1px solid #a5d6a7;
        text-align: center;
        margin: 10px 0;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        font-size: 0.9rem;
    }

    .warning-box {
        padding: 12px 16px;
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        border-radius: 0 8px 8px 0;
        margin: 4px 0;
        font-size: 0.9rem;
    }

    .compare-box {
        padding: 20px;
        background: linear-gradient(135deg, #f8f0f4 0%, #f0e4ea 100%);
        border-radius: 12px;
        border: 1px solid #e0c8d4;
        text-align: center;
    }

    .pipeline-step {
        padding: 12px 16px;
        background: #f8f0f4;
        border-radius: 8px;
        border-left: 3px solid var(--wine-mid);
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>Wine Quality Prediction</h1>
    <p>Machine learning-powered wine quality assessment using chemical properties</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# --- Session State Init ---
for key, default in [
    ('history', []),
    ('last_prediction', None),
    ('last_input', None),
    ('last_probabilities', None),
    ('last_shap', None),
    ('llm_response', None),
    ('feedback_submitted', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

quality_descriptions = {
    3: "Very Poor: Noticeable flaws, highly unbalanced, or faulty.",
    4: "Poor: Unbalanced, lacks character, or has minor flaws.",
    5: "Fair: Average commercial wine, acceptable but unexciting.",
    6: "Good: Well-made, balanced, and enjoyable everyday wine.",
    7: "Very Good: High quality, complex, and highly enjoyable.",
    8: "Excellent: Outstanding complexity, balance, and character.",
    9: "Exceptional: World-class, flawless, phenomenal wine."
}

# --- Feature Config ---
FEATURE_CONFIGS = [
    ("Fixed Acidity",    "fixed_acidity",        3.0,  16.0,  7.0,   0.1),
    ("Volatile Acidity", "volatile_acidity",     0.0,  2.0,   0.3,   0.01),
    ("Citric Acid",      "citric_acid",          0.0,  2.0,   0.3,   0.01),
    ("Residual Sugar",   "residual_sugar",       0.0,  70.0,  5.0,   0.1),
    ("Chlorides",        "chlorides",            0.0,  0.6,   0.05,  0.001),
    ("Free SO2",         "free_sulfur_dioxide",  0.0,  300.0, 30.0,  1.0),
    ("Total SO2",        "total_sulfur_dioxide", 0.0,  450.0, 115.0, 1.0),
    ("Density",          "density",              0.98, 1.04,  0.99,  0.001),
    ("pH",               "pH_val",               2.5,  4.5,   3.2,   0.01),
    ("Sulphates",        "sulphates",            0.0,  2.0,   0.5,   0.01),
    ("Alcohol (%)",      "alcohol",              7.0,  15.0,  10.5,  0.1),
]

DF_COLUMN_MAP = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "chlorides": "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "density": "density",
    "pH_val": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}


# =============================================
# MODEL / DATA LOADING
# =============================================
@st.cache_resource
def load_models():
    base_path = "models"
    try:
        imputer = joblib.load(os.path.join(base_path, 'imputer.joblib'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.joblib'))
        label_encoder = joblib.load(os.path.join(base_path, 'label_encoder.joblib'))
        top_features = joblib.load(os.path.join(base_path, 'top_features.joblib'))
        model = joblib.load(os.path.join(base_path, 'best_model.joblib'))

        try:
            with open(os.path.join(base_path, 'metadata.txt'), 'r') as f:
                model_name = f.read().strip()
        except Exception:
            model_name = "Unknown Model"

        best_et = joblib.load(os.path.join(base_path, 'best_et.joblib'))
        explainer = shap.TreeExplainer(best_et)

        return imputer, scaler, label_encoder, top_features, model, model_name, explainer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None, None


@st.cache_data
def load_dataset():
    """Load the training dataset for stats and exploration."""
    try:
        df = pd.read_csv("wine_quality_dataset.csv")
        return df
    except Exception:
        return None


# =============================================
# HELPER FUNCTIONS
# =============================================
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


def run_prediction(input_df, imputer, scaler, label_encoder, top_features, model, model_name):
    """Run a single prediction and return (quality, probabilities, pred_encoded)."""
    engineered = create_safe_wine_features(input_df)
    X_selected = engineered[top_features]
    X_imputed = imputer.transform(X_selected)
    X_scaled = scaler.transform(X_imputed)

    if model_name == "Blending":
        try:
            best_rf = joblib.load(os.path.join("models", 'best_rf.joblib'))
            best_et = joblib.load(os.path.join("models", 'best_et.joblib'))
            best_xgb = joblib.load(os.path.join("models", 'best_xgb.joblib'))
            blend_X = np.hstack((
                best_rf.predict_proba(X_scaled),
                best_et.predict_proba(X_scaled),
                best_xgb.predict_proba(X_scaled)
            ))
            pred_encoded = model.predict(blend_X)
            probabilities = model.predict_proba(blend_X)[0] if hasattr(model, 'predict_proba') else None
        except Exception:
            return None, None, None, None
    else:
        pred_encoded = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None

    quality = label_encoder.inverse_transform(pred_encoded)[0]
    return quality, probabilities, pred_encoded, X_scaled


def get_gemini_description(api_key, wine_params, predicted_quality, wine_type):
    """Generate a Vietnamese wine description using Gemini."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel('gemini-2.0-flash')

        prompt = f"""Bạn là một chuyên gia rượu vang (sommelier) giàu kinh nghiệm. Hãy mô tả và nhận xét về chai rượu vang dựa trên các thông số hóa học sau đây. Viết bằng tiếng Việt, ngắn gọn nhưng chuyên nghiệp (khoảng 150-200 từ).

Loại rượu: {"Vang đỏ" if wine_type == "red" else "Vang trắng"}
Chất lượng dự đoán bởi mô hình ML: {predicted_quality}/9

Thông số hóa học:
- Độ axit cố định (Fixed Acidity): {wine_params['fixed acidity']}
- Độ axit bay hơi (Volatile Acidity): {wine_params['volatile acidity']}
- Axit citric (Citric Acid): {wine_params['citric acid']}
- Đường dư (Residual Sugar): {wine_params['residual sugar']} g/L
- Clorua (Chlorides): {wine_params['chlorides']}
- SO2 tự do (Free SO2): {wine_params['free sulfur dioxide']}
- SO2 tổng (Total SO2): {wine_params['total sulfur dioxide']}
- Tỷ trọng (Density): {wine_params['density']}
- Độ pH: {wine_params['pH']}
- Sunfat (Sulphates): {wine_params['sulphates']}
- Nồng độ cồn (Alcohol): {wine_params['alcohol']}%

Hãy bao gồm:
1. Nhận xét tổng quan về chất lượng
2. Mô tả hương vị và đặc điểm dự kiến dựa trên thông số hóa học
3. Gợi ý món ăn kết hợp (food pairing)
4. Một lời khuyên ngắn cho người thưởng thức"""

        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"


FEEDBACK_FILE = "user_feedback.csv"

def save_feedback(input_data, predicted_quality, user_quality, user_note):
    """Append user feedback to CSV file."""
    feedback_row = input_data.copy()
    feedback_row['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_row['predicted_quality'] = predicted_quality
    feedback_row['user_quality'] = user_quality
    feedback_row['user_note'] = user_note

    if os.path.exists(FEEDBACK_FILE):
        existing = pd.read_csv(FEEDBACK_FILE)
        updated = pd.concat([existing, feedback_row], ignore_index=True)
    else:
        updated = feedback_row

    updated.to_csv(FEEDBACK_FILE, index=False)
    return True


def load_feedback():
    """Load existing feedback from CSV."""
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return None


def check_out_of_range(input_values, dataset_df):
    """Check which input values fall outside the training data observed range."""
    warnings = []
    numeric_cols = dataset_df.select_dtypes(include=[np.number]).columns

    for label, key, min_v, max_v, default, step in FEATURE_CONFIGS:
        col_name = DF_COLUMN_MAP[key]
        if col_name in numeric_cols:
            ds_min = dataset_df[col_name].min()
            ds_max = dataset_df[col_name].max()
            val = input_values[key]
            if val < ds_min:
                warnings.append(f"**{label}** = {val} is below the training data minimum ({ds_min:.3f})")
            elif val > ds_max:
                warnings.append(f"**{label}** = {val} is above the training data maximum ({ds_max:.3f})")
    return warnings


# =============================================
# SLIDER + NUMBER INPUT SYNC CALLBACKS
# =============================================
def sync_from_slider(feature_key):
    """When slider changes, update the canonical value and the number input."""
    val = st.session_state[f"slider_{feature_key}"]
    st.session_state[f"val_{feature_key}"] = val
    st.session_state[f"num_{feature_key}"] = val


def sync_from_number(feature_key):
    """When number input changes, update the canonical value and the slider."""
    val = st.session_state[f"num_{feature_key}"]
    st.session_state[f"val_{feature_key}"] = val
    st.session_state[f"slider_{feature_key}"] = val


# =============================================
# LOAD EVERYTHING
# =============================================
imputer, scaler, label_encoder, top_features, model, model_name, explainer = load_models()
dataset_df = load_dataset()

if model is not None:

    # =============================================
    # SIDEBAR
    # =============================================
    st.sidebar.markdown("## Wine Characteristics")
    st.sidebar.caption("Adjust sliders or type values directly")

    # Gemini API Key
    st.sidebar.markdown("---")
    st.sidebar.markdown("### AI Description (Gemini)")
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey"
    )
    st.sidebar.markdown("---")

    # Wine type
    wine_type = st.sidebar.selectbox("Type", ["white", "red"])

    # Synced sliders + number inputs
    input_values = {}
    for label, key, min_val, max_val, default_val, step in FEATURE_CONFIGS:
        # Initialize canonical value if not set
        state_key = f"val_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = default_val

        current_val = st.session_state[state_key]

        st.sidebar.markdown(f"**{label}**")
        col1, col2 = st.sidebar.columns([3, 1])
        col1.slider(
            label, min_val, max_val,
            value=current_val, step=step,
            key=f"slider_{key}",
            on_change=sync_from_slider, args=(key,),
            label_visibility="collapsed"
        )
        col2.number_input(
            label, min_val, max_val,
            value=current_val, step=step,
            key=f"num_{key}",
            on_change=sync_from_number, args=(key,),
            label_visibility="collapsed"
        )
        input_values[key] = st.session_state[state_key]

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset to Defaults", use_container_width=True):
        for _, key, _, _, default_val, _ in FEATURE_CONFIGS:
            for prefix in ['val_', 'slider_', 'num_']:
                sk = f"{prefix}{key}"
                if sk in st.session_state:
                    del st.session_state[sk]
        st.session_state.last_prediction = None
        st.session_state.llm_response = None
        st.session_state.feedback_submitted = False
        st.rerun()

    # --- Build input dataframe ---
    input_data = pd.DataFrame({
        'type': [wine_type],
        **{DF_COLUMN_MAP[key]: [input_values[key]] for key in input_values}
    })

    # --- Input summary ---
    with st.expander("Current Input Parameters", expanded=False):
        st.dataframe(input_data, use_container_width=True)

    # --- Predict Button ---
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_clicked = st.button("Predict Quality", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analyzing wine characteristics..."):
            quality, probabilities, pred_encoded, X_scaled = run_prediction(
                input_data, imputer, scaler, label_encoder, top_features, model, model_name
            )

            if quality is not None:
                all_classes = label_encoder.classes_

                # SHAP
                shap_values = explainer(X_scaled)
                prediction_idx = int(pred_encoded[0])
                shap_explanation = shap_values[0, :, prediction_idx]
                shap_explanation.feature_names = top_features

                # Save to session state
                st.session_state.last_prediction = quality
                st.session_state.last_input = input_data.copy()
                st.session_state.last_probabilities = (probabilities, all_classes) if probabilities is not None else None
                st.session_state.last_shap = shap_explanation
                st.session_state.feedback_submitted = False
                st.session_state.llm_response = None

                # Save to history
                history_entry = input_data.copy()
                history_entry['Predicted Quality'] = quality
                history_entry['Timestamp'] = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.history.append(history_entry)

    # =============================================
    # MAIN CONTENT — TABS
    # =============================================
    if st.session_state.last_prediction is not None:
        predicted_quality = st.session_state.last_prediction

        tab_predict, tab_xai, tab_batch, tab_compare, tab_explorer, tab_model, tab_feedback, tab_history = st.tabs([
            "Prediction", "Explainability", "Batch Predict",
            "Comparison", "Dataset Explorer", "Model Info",
            "User Feedback", "History"
        ])

        # =========================================
        # TAB 1: PREDICTION
        # =========================================
        with tab_predict:
            if predicted_quality >= 7:
                q_class = "high-quality"
            elif predicted_quality <= 4:
                q_class = "low-quality"
            else:
                q_class = "med-quality"

            st.markdown(
                f'<div class="result-box">'
                f'<span style="font-size: 16px; color: #666;">Predicted Quality Rating</span><br>'
                f'<span class="{q_class}">{predicted_quality}</span>'
                f'<span style="font-size: 16px; color: #888;"> / 9</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            explanation = quality_descriptions.get(predicted_quality, "Unknown rating.")
            st.success(f"**What this means:** {explanation}")
            st.caption(f"Model: **{model_name}**")

            # Out-of-range warnings
            if dataset_df is not None:
                oor_warnings = check_out_of_range(input_values, dataset_df)
                if oor_warnings:
                    st.markdown("#### Out-of-Range Warnings")
                    st.warning(
                        "Some input values fall outside the training data range. "
                        "Predictions may be less reliable."
                    )
                    for w in oor_warnings:
                        st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

            # Confidence chart
            if st.session_state.last_probabilities is not None:
                st.markdown("#### Prediction Confidence")
                probabilities, all_classes = st.session_state.last_probabilities
                prob_df = pd.DataFrame({
                    'Quality Score': [str(c) for c in all_classes],
                    'Probability': probabilities
                }).sort_values('Quality Score')

                st.bar_chart(prob_df.set_index('Quality Score'), color="#722F37")
                max_prob = probabilities.max() * 100
                st.caption(f"Confidence: **{max_prob:.1f}%** for quality score **{predicted_quality}**")

            # LLM Vietnamese Description
            st.markdown("---")
            st.markdown("#### AI Wine Description (Tieng Viet)")

            if not gemini_key:
                st.info("Nhap Gemini API Key o thanh ben trai de nhan mo ta ruou bang tieng Viet tu AI.")
            else:
                gen_col1, gen_col2 = st.columns([1, 1])
                with gen_col1:
                    generate_clicked = st.button("Generate", use_container_width=True)
                with gen_col2:
                    regenerate_clicked = st.button("Regenerate", use_container_width=True)

                if generate_clicked or regenerate_clicked:
                    with st.spinner("Generating AI description..."):
                        wine_params = st.session_state.last_input.iloc[0].to_dict()
                        response = get_gemini_description(
                            gemini_key, wine_params, predicted_quality,
                            wine_params.get('type', 'white')
                        )
                        st.session_state.llm_response = response

                if st.session_state.llm_response:
                    st.markdown(
                        f'<div style="background: linear-gradient(135deg, #fef9f0 0%, #fdf3e3 100%); '
                        f'padding: 20px; border-radius: 12px; border: 1px solid #f0dcc0; '
                        f'line-height: 1.7;">'
                        f'{st.session_state.llm_response}</div>',
                        unsafe_allow_html=True
                    )

        # =========================================
        # TAB 2: EXPLAINABILITY (XAI)
        # =========================================
        with tab_xai:
            st.markdown("### Explainable AI (SHAP)")
            st.markdown(
                "How each feature contributed to this specific prediction "
                "according to our strongest base model (**Extra Trees**):"
            )

            if st.session_state.last_shap is not None:
                st_shap(shap.plots.waterfall(st.session_state.last_shap), height=500)

                st.markdown("---")
                st.caption(
                    "**How to read this:** Red bars push the prediction higher, "
                    "blue bars push it lower. The length shows how much each feature contributed."
                )

        # =========================================
        # TAB 3: BATCH PREDICTION
        # =========================================
        with tab_batch:
            st.markdown("### Batch Prediction")
            st.markdown(
                "Upload a CSV file with multiple wine samples to predict quality for all of them at once."
            )

            # Show expected format
            with st.expander("Expected CSV format"):
                st.markdown(
                    "The CSV should contain these columns (order doesn't matter):"
                )
                expected_cols = ['type', 'fixed acidity', 'volatile acidity', 'citric acid',
                                 'residual sugar', 'chlorides', 'free sulfur dioxide',
                                 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
                st.code(", ".join(expected_cols))
                st.caption("The 'type' column should contain 'red' or 'white'.")

            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.markdown(f"**Loaded {len(batch_df)} samples**")
                    st.dataframe(batch_df.head(10), use_container_width=True)
                    if len(batch_df) > 10:
                        st.caption(f"Showing first 10 of {len(batch_df)} rows.")

                    if st.button("Run Batch Prediction", type="primary"):
                        with st.spinner(f"Predicting {len(batch_df)} samples..."):
                            results = []
                            for idx, row in batch_df.iterrows():
                                row_df = pd.DataFrame([row])
                                q, probs, _, _ = run_prediction(
                                    row_df, imputer, scaler, label_encoder,
                                    top_features, model, model_name
                                )
                                confidence = probs.max() * 100 if probs is not None else None
                                results.append({
                                    'Sample': idx + 1,
                                    'Type': row.get('type', 'N/A'),
                                    'Predicted Quality': q if q is not None else 'Error',
                                    'Confidence (%)': f"{confidence:.1f}" if confidence else 'N/A'
                                })

                            results_df = pd.DataFrame(results)
                            st.markdown("#### Results")
                            st.dataframe(results_df, use_container_width=True)

                            # Summary stats
                            valid = results_df[results_df['Predicted Quality'] != 'Error']
                            if len(valid) > 0:
                                quality_counts = valid['Predicted Quality'].value_counts().sort_index()
                                st.markdown("#### Quality Distribution")
                                st.bar_chart(quality_counts, color="#722F37")

                            # Download
                            full_results = batch_df.copy()
                            full_results['Predicted Quality'] = [r['Predicted Quality'] for r in results]
                            full_results['Confidence (%)'] = [r['Confidence (%)'] for r in results]
                            csv_out = full_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Results CSV",
                                csv_out,
                                "batch_predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"Error processing CSV: {str(e)}")

        # =========================================
        # TAB 4: WINE COMPARISON
        # =========================================
        with tab_compare:
            st.markdown("### Wine Comparison")
            st.markdown(
                "Compare two wines side by side. **Wine A** uses the current sidebar values. "
                "Adjust **Wine B** below."
            )

            st.markdown("#### Wine B Parameters")
            wine_b_type = st.selectbox("Type (Wine B)", ["white", "red"], key="compare_type")

            # Layout: 3 columns of inputs for Wine B
            wine_b_values = {}
            cols_per_row = 3
            for i in range(0, len(FEATURE_CONFIGS), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(FEATURE_CONFIGS):
                        label, key, min_v, max_v, default_v, step = FEATURE_CONFIGS[i + j]
                        wine_b_values[key] = col.number_input(
                            label, min_v, max_v, default_v, step,
                            key=f"compare_{key}"
                        )

            if st.button("Compare Wines", type="primary", use_container_width=True):
                with st.spinner("Comparing..."):
                    # Wine A = current sidebar input
                    wine_a_df = st.session_state.last_input.copy()

                    # Wine B
                    wine_b_df = pd.DataFrame({
                        'type': [wine_b_type],
                        **{DF_COLUMN_MAP[k]: [v] for k, v in wine_b_values.items()}
                    })

                    q_a, prob_a, _, _ = run_prediction(
                        wine_a_df, imputer, scaler, label_encoder, top_features, model, model_name
                    )
                    q_b, prob_b, _, _ = run_prediction(
                        wine_b_df, imputer, scaler, label_encoder, top_features, model, model_name
                    )

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("#### Wine A (Sidebar)")
                        if q_a is not None:
                            cls_a = "high-quality" if q_a >= 7 else ("low-quality" if q_a <= 4 else "med-quality")
                            st.markdown(
                                f'<div class="compare-box"><span style="font-size:14px;color:#666;">Quality</span><br>'
                                f'<span class="{cls_a}">{q_a}</span><span style="color:#888;"> / 9</span></div>',
                                unsafe_allow_html=True
                            )
                            if prob_a is not None:
                                st.caption(f"Confidence: {prob_a.max()*100:.1f}%")
                        st.dataframe(wine_a_df.T.rename(columns={0: "Value"}), use_container_width=True)

                    with col_b:
                        st.markdown("#### Wine B (Custom)")
                        if q_b is not None:
                            cls_b = "high-quality" if q_b >= 7 else ("low-quality" if q_b <= 4 else "med-quality")
                            st.markdown(
                                f'<div class="compare-box"><span style="font-size:14px;color:#666;">Quality</span><br>'
                                f'<span class="{cls_b}">{q_b}</span><span style="color:#888;"> / 9</span></div>',
                                unsafe_allow_html=True
                            )
                            if prob_b is not None:
                                st.caption(f"Confidence: {prob_b.max()*100:.1f}%")
                        st.dataframe(wine_b_df.T.rename(columns={0: "Value"}), use_container_width=True)

                    # Difference summary
                    if q_a is not None and q_b is not None:
                        diff = q_b - q_a
                        if diff > 0:
                            st.info(f"Wine B is rated **{diff} point(s) higher** than Wine A.")
                        elif diff < 0:
                            st.info(f"Wine A is rated **{abs(diff)} point(s) higher** than Wine B.")
                        else:
                            st.info("Both wines received the **same quality rating**.")

        # =========================================
        # TAB 5: DATASET EXPLORER
        # =========================================
        with tab_explorer:
            st.markdown("### Dataset Explorer")

            if dataset_df is not None:
                st.markdown(
                    f"Explore the training dataset used to build this model. "
                    f"**{len(dataset_df):,} samples** with **{len(dataset_df.columns)} features**."
                )

                # Summary statistics
                st.markdown("#### Summary Statistics")
                numeric_df = dataset_df.select_dtypes(include=[np.number])
                st.dataframe(numeric_df.describe().round(3), use_container_width=True)

                # Quality distribution
                st.markdown("#### Quality Score Distribution")
                quality_counts = dataset_df['quality'].value_counts().sort_index()
                st.bar_chart(quality_counts, color="#722F37")

                # Wine type breakdown
                st.markdown("#### Wine Type Breakdown")
                type_counts = dataset_df['type'].value_counts()
                type_df = pd.DataFrame({
                    'Type': type_counts.index,
                    'Count': type_counts.values,
                    'Percentage': (type_counts.values / len(dataset_df) * 100).round(1)
                })
                st.dataframe(type_df, use_container_width=True, hide_index=True)

                # Feature distributions
                st.markdown("#### Feature Distributions")
                selected_feature = st.selectbox(
                    "Select a feature to visualize",
                    numeric_df.columns.tolist(),
                    key="explorer_feature"
                )
                if selected_feature:
                    st.bar_chart(
                        pd.cut(numeric_df[selected_feature], bins=30).value_counts().sort_index(),
                        color="#722F37"
                    )

                    # Show where current input falls
                    col_key_reverse = {v: k for k, v in DF_COLUMN_MAP.items()}
                    if selected_feature in col_key_reverse:
                        fkey = col_key_reverse[selected_feature]
                        current_val = input_values.get(fkey)
                        if current_val is not None:
                            percentile = (numeric_df[selected_feature] < current_val).mean() * 100
                            st.caption(
                                f"Your current input ({current_val}) is at the "
                                f"**{percentile:.0f}th percentile** of the training data."
                            )

                # Correlation with quality
                st.markdown("#### Correlation with Quality")
                correlations = numeric_df.corr()['quality'].drop('quality').sort_values(ascending=False)
                corr_df = pd.DataFrame({
                    'Feature': correlations.index,
                    'Correlation': correlations.values.round(3)
                })
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Could not load the training dataset (wine_quality_dataset.csv).")

        # =========================================
        # TAB 6: MODEL INFO
        # =========================================
        with tab_model:
            st.markdown("### Model Information")
            st.markdown("Overview of the machine learning pipeline used in this system.")

            st.markdown("#### Pipeline Architecture")

            steps = [
                ("1. Data Collection",
                 "Combined red and white wine datasets from the UCI ML Repository (6,497 samples, 12 features + quality label)."),
                ("2. Preprocessing",
                 "Handled missing values with median imputation. Applied StandardScaler for feature normalization."),
                ("3. Feature Engineering",
                 "Created 11 derived features including acid ratio, sulfur ratio, alcohol-sugar ratio, "
                 "polynomial features (squared terms), log transforms, and a composite chemical balance index."),
                ("4. Feature Selection",
                 "Selected top features using importance-based ranking to reduce dimensionality "
                 "and improve model generalization."),
                ("5. Class Balancing",
                 "Applied SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance "
                 "in quality scores (most wines scored 5-6)."),
                ("6. Model Training",
                 "Trained multiple base models: Random Forest, Extra Trees, and XGBoost. "
                 "Hyperparameters optimized using Optuna (Bayesian optimization)."),
                ("7. Ensemble (Stacking)",
                 "Combined base models using a Stacking ensemble with a Logistic Regression meta-learner. "
                 "This leverages the strengths of each base model for more robust predictions."),
            ]

            for title, desc in steps:
                st.markdown(f'<div class="pipeline-step"><strong>{title}</strong><br>{desc}</div>', unsafe_allow_html=True)

            st.markdown("#### Current Model Details")
            details_col1, details_col2 = st.columns(2)
            with details_col1:
                st.markdown(f"**Model Type:** {model_name}")
                st.markdown(f"**Number of Features:** {len(top_features)}")
                st.markdown(f"**Quality Classes:** {list(label_encoder.classes_)}")
            with details_col2:
                st.markdown(f"**Training Samples:** 6,497")
                st.markdown(f"**Base Models:** Random Forest, Extra Trees, XGBoost")
                st.markdown(f"**XAI Method:** SHAP (TreeExplainer)")

            st.markdown("#### Selected Features")
            feat_df = pd.DataFrame({'Feature': top_features, '#': range(1, len(top_features) + 1)})
            feat_df = feat_df.set_index('#')
            st.dataframe(feat_df, use_container_width=True)

        # =========================================
        # TAB 7: USER FEEDBACK
        # =========================================
        with tab_feedback:
            st.markdown("### Rate This Wine")
            st.markdown("Provide your own quality assessment to help improve the model.")

            if st.session_state.feedback_submitted:
                st.markdown(
                    '<div class="feedback-success">'
                    '<strong>Feedback saved successfully!</strong><br>'
                    'Your rating has been recorded for future model training.'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"**Model predicted:** {predicted_quality}/9 — "
                    f"What would **you** rate this wine?"
                )

                fb_col1, fb_col2 = st.columns([1, 1])
                with fb_col1:
                    user_quality = st.slider(
                        "Your Quality Rating",
                        min_value=3, max_value=9, value=predicted_quality,
                        help="Rate based on the wine's chemical profile"
                    )
                with fb_col2:
                    user_note = st.text_area(
                        "Notes (optional)",
                        placeholder="Any observations about this wine...",
                        height=100
                    )

                if st.button("Submit Feedback", type="primary", use_container_width=True):
                    save_feedback(
                        st.session_state.last_input,
                        predicted_quality,
                        user_quality,
                        user_note
                    )
                    st.session_state.feedback_submitted = True
                    st.rerun()

            # Show existing feedback
            st.markdown("---")
            feedback_data = load_feedback()
            if feedback_data is not None and len(feedback_data) > 0:
                with st.expander(f"Saved Feedback ({len(feedback_data)} entries)", expanded=False):
                    st.dataframe(feedback_data, use_container_width=True)

                    csv_data = feedback_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Feedback CSV",
                        csv_data,
                        "user_feedback.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.caption("No feedback submitted yet.")

        # =========================================
        # TAB 8: HISTORY
        # =========================================
        with tab_history:
            st.markdown("### Prediction History")
            if st.session_state.history:
                history_df = pd.concat(st.session_state.history, ignore_index=True)
                st.dataframe(history_df, use_container_width=True)

                csv_history = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download History CSV",
                    csv_history,
                    "prediction_history.csv",
                    "text/csv",
                    use_container_width=True
                )

                st.caption(f"Total predictions this session: **{len(st.session_state.history)}**")
            else:
                st.info("No predictions yet. Use the 'Predict Quality' button to get started.")

    else:
        # No prediction yet
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #888;">
            <p style="font-size: 28px; margin-bottom: 10px; font-weight: 600; color: #722F37;">Wine Quality Prediction</p>
            <p style="font-size: 1.1rem;">Adjust the wine characteristics in the sidebar<br>and click <strong>Predict Quality</strong> to begin.</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.caption("Wine Quality Prediction Dashboard | Built with Streamlit | Powered by ML & SHAP")

else:
    st.error("Could not load models. Please ensure all model files exist in the `models/` directory.")
