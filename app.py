import streamlit as st
import xgboost as xgb
import pandas as pd
import json
import time

# ==========================================
# 1. Global Page Configuration
# ==========================================
st.set_page_config(
    page_title="Clinical Early Screening System",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. Model & Thresholds Loading
# ==========================================
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("xgb_clinical_model.json")
    return model

@st.cache_resource
def load_thresholds():
    with open("thresholds.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
thresholds = load_thresholds()
T_LOW = thresholds["T_low"]
T_HIGH = thresholds["T_high"]

# ==========================================
# 3. Custom UI Styles & Header
# ==========================================
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>⚕️ Early Screening & Risk Assessment System</h1>
    <p style='text-align: center; color: #7F8C8D;'>
    Built upon XGBoost and SHAP-RFE algorithms, this system achieves precise risk stratification using only 4 core clinical indicators.
    </p >
    <hr style="border:1px solid #E5E8E8">
""", unsafe_allow_html=True)

# ==========================================
# 4. Form Input Section (Dual-column Layout)
# ==========================================
st.markdown("### 📋 Enter Patient Clinical Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Demographics")
    # Assuming: 0 for Male, 1 for Female (adjust if your encoding is different)
    gender_input = st.radio("Gender", options=["Male", "Female"], horizontal=True)
    gender_val = 0 if gender_input == "Male" else 1
    
    age_val = st.number_input("Age", min_value=10, max_value=120, value=50, step=1)

with col2:
    st.markdown("#### Clinical Indicators")
    night_pain_input = st.selectbox("Night Sleep Pain", options=["Absent", "Present"])
    night_pain_val = 0 if night_pain_input == "Absent" else 1

    labour_input = st.selectbox("Regular Hard Physical Labour", options=["Yes", "No"])
    labour_pain_val = 1 if labour_input == "Yes" else 0

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. Core Computation & Result Presentation
# ==========================================
# Center and enlarge the button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🚀 EXECUTE RISK ASSESSMENT", use_container_width=True, type="primary")

if predict_button:
    input_df = pd.DataFrame({
        'Gender': [gender_val],
        'Age': [age_val],
        'Night Sleep Pain': [night_pain_val],
        'Hard Manual Labour Pain': [labour_pain_val]
    })

    with st.spinner('Running AI model calculation, please wait...'):
        time.sleep(0.5)
        prob = model.predict_proba(input_df)[0][1]

    st.markdown("<hr style='border:1px dashed #E5E8E8'>", unsafe_allow_html=True)
    st.markdown("### 📊 Screening Results")

    # Risk stratification based on data-driven thresholds
    if prob >= T_HIGH:
        st.markdown("""
            <div style='background-color:#FDEDEC; border-left:6px solid #E74C3C; padding:20px; border-radius:8px; margin:20px 0;'>
                <h2 style='color:#C0392B; margin:0;'>⚠️ High Risk</h2>
                <p style='color:#2C3E50; font-size:16px; margin-top:10px;'>
                This patient is classified as <strong>high risk</strong> based on the clinical indicators provided.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **Clinical Recommendation:**
        - Immediate further imaging examination (e.g., DXA bone density scan) is strongly recommended
        - Consider pathological examination if clinically indicated
        - Arrange close follow-up and monitoring plan
        - Evaluate the need for early intervention or preventive treatment
        """)
    elif prob >= T_LOW:
        st.markdown("""
            <div style='background-color:#FEF9E7; border-left:6px solid #F39C12; padding:20px; border-radius:8px; margin:20px 0;'>
                <h2 style='color:#E67E22; margin:0;'>⚡ Moderate Risk</h2>
                <p style='color:#2C3E50; font-size:16px; margin-top:10px;'>
                This patient is classified as <strong>moderate risk</strong> and warrants further clinical attention.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **Clinical Recommendation:**
        - Combine with other clinical tests for comprehensive evaluation
        - Schedule a follow-up assessment within 3-6 months
        - Provide lifestyle and dietary guidance for bone health
        - Monitor for symptom progression
        """)
    else:
        st.markdown("""
            <div style='background-color:#EAFAF1; border-left:6px solid #27AE60; padding:20px; border-radius:8px; margin:20px 0;'>
                <h2 style='color:#1E8449; margin:0;'>✅ Low Risk</h2>
                <p style='color:#2C3E50; font-size:16px; margin-top:10px;'>
                This patient is classified as <strong>low risk</strong> based on current clinical indicators.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **Clinical Recommendation:**
        - Routine health guidance and annual check-ups are advised
        - Encourage regular physical activity and adequate calcium/vitamin D intake
        - Re-assess if new symptoms develop
        """)

# ==========================================
# 6. Footer & Disclaimer
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("ℹ️ About the Model & Medical Disclaimer"):
    st.markdown("""
    - **Model Core**: Built on the eXtreme Gradient Boosting (XGBoost) algorithm.
    - **Feature Selection**: Utilizes SHAP values and Recursive Feature Elimination (RFE) to identify the 4 most discriminative clinical indicators from dozens of original variables.
    - **Risk Stratification**: Thresholds are derived from Youden index optimization and high-sensitivity analysis on the training cohort, validated on an independent external dataset.
    - **Medical Disclaimer**: This tool is designed to assist healthcare professionals in early screening. The risk classification does not constitute a definitive medical diagnosis. Clinical judgment should be based on comprehensive patient history and other clinical indications.
    """)