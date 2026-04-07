import streamlit as st
import xgboost as xgb
import pandas as pd
import time
import plotly.graph_objects as go

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
# 2. Model Loading and Caching
# ==========================================
@st.cache_resource
def load_model():
    # Ensure your best model is saved as xgb_clinical_model.json in the same directory
    model = xgb.XGBClassifier()
    model.load_model("xgb_clinical_model.json")
    return model

model = load_model()

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
    st.markdown("#### Core Symptoms")
    # Assuming: 0 for Absent, 1 for Present
    night_pain_input = st.selectbox("Night Sleep Pain", options=["Absent", "Present"])
    night_pain_val = 0 if night_pain_input == "Absent" else 1
    
    labour_pain_input = st.selectbox("Hard Manual Labour Pain", options=["Absent", "Present"])
    labour_pain_val = 0 if labour_pain_input == "Absent" else 1

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. Core Computation & Result Presentation
# ==========================================
# Center and enlarge the button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🚀 EXECUTE RISK ASSESSMENT", use_container_width=True, type="primary")

if predict_button:
    # Construct input DataFrame (Columns must strictly match your training 'core_features')
    # Strict order: ['Gender', 'Age', 'Night Sleep Pain', 'Hard Manual Labour Pain']
    input_df = pd.DataFrame({
        'Gender': [gender_val],
        'Age': [age_val],
        'Night Sleep Pain': [night_pain_val],
        'Hard Manual Labour Pain': [labour_pain_val]
    })
    
    with st.spinner('Running AI model calculation, please wait...'):
        time.sleep(0.5) # Add a slight delay for better UX
        prob = model.predict_proba(input_df)[0][1]
        risk_percentage = prob * 100
    
    st.markdown("<hr style='border:1px dashed #E5E8E8'>", unsafe_allow_html=True)
    st.markdown("### 📊 Screening Results Assessment")
    
    # 5.1 Plotly Risk Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percentage,
        number = {'suffix': "%", 'font': {'size': 45, 'color': "#2C3E50"}},
        title = {'text': "Predicted Probability", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Hide default progress bar
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "#2ECC71"},   # Green (Safe)
                {'range': [20, 50], 'color': "#F1C40F"},  # Yellow (Warning)
                {'range': [50, 100], 'color': "#E74C3C"}  # Red (High Risk)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 5},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # 5.2 Clinical Intervention Suggestions based on thresholds
    if risk_percentage >= 50:
        st.error(f"**High Risk Warning**: The predicted probability for this patient is {risk_percentage:.1f}%. Immediate further imaging or pathological examination and close follow-up are strongly recommended.")
    elif risk_percentage >= 20:
        st.warning(f"**Moderate Risk**: The patient shows a moderate tendency for the condition ({risk_percentage:.1f}%). Consider combining with other clinical tests and schedule a follow-up assessment.")
    else:
        st.success(f"**Low Risk**: The predicted probability is low ({risk_percentage:.1f}%). Routine health guidance and annual check-ups are advised.")

# ==========================================
# 6. Footer & Disclaimer
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("ℹ️ About the Model & Medical Disclaimer"):
    st.markdown("""
    - **Model Core**: Built on the eXtreme Gradient Boosting (XGBoost) algorithm.
    - **Feature Selection**: Utilizes SHAP values and Recursive Feature Elimination (RFE) to identify the 4 most discriminative clinical indicators from dozens of original variables.
    - **Medical Disclaimer**: This tool is designed to assist healthcare professionals in early screening. The predicted results do not constitute a definitive medical diagnosis. Clinical judgment should be based on comprehensive patient history and other clinical indications.
    """)