import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Spectral Analysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("🔬 Spectral Analysis Platform")
st.markdown("### Advanced tools for spectroscopic data analysis and machine learning")

st.markdown("---")

# Available tools
st.markdown("## 📋 Available Tools")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 📊 Calculate Absorbance
    Process raw spectral data to calculate absorbance spectra with dark spectrum 
    correction, averaging, and smoothing.
    
    #### 🧪 Model Calibration
    Build and optimize machine learning models with multiple algorithms, automated 
    hyperparameter tuning, and cross-validation.
    """)

with col2:
    st.markdown("""
    #### 🎯 Predict Concentration
    Use trained models to predict analyte concentrations from new spectral data 
    with batch processing support.
    
    #### 💬 Provide Feedback
    Submit feedback and suggestions to improve the platform.
    """)

st.markdown("---")

# Quick start
st.markdown("## 🚀 Quick Start")
st.info("""
1. **Calculate Absorbance** - Process your spectral data
2. **Model Calibration** - Build predictive models  
3. **Predict Concentration** - Apply models to new data
4. **Provide Feedback** - Share your experience

Use the **sidebar** to navigate between pages.
""")

