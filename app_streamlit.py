# app_streamlit.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load models
@st.cache_resource
def load_models():
    try:
        models = {
            'knn_clf': joblib.load("knn_classifier.pkl"),
            'log_reg': joblib.load("logistic_regression.pkl"),
            'knn_reg': joblib.load("knn_regressor.pkl"),
            'lin_reg': joblib.load("linear_regression.pkl"),
            'scaler': joblib.load("scaler.pkl")
        }
        return models
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found. Please run main.py first to train the models.\n\nMissing: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

models = load_models()

# App title
st.title("🎓 Student Performance Prediction")
st.markdown("Predict student performance using machine learning models")

# Sidebar for input
st.sidebar.header("📊 Student Information")

features = ['studytime', 'absences', 'failures', 'G1', 'G2', 'Medu', 'Fedu', 'traveltime']
feature_labels = {
    'studytime': 'Study Time (hours/week)',
    'absences': 'Number of Absences',
    'failures': 'Past Class Failures',
    'G1': 'First Period Grade (0-20)',
    'G2': 'Second Period Grade (0-20)',
    'Medu': "Mother's Education (0-4)",
    'Fedu': "Father's Education (0-4)",
    'traveltime': 'Travel Time to School (1-4)'
}

feature_descriptions = {
    'studytime': 'Weekly study time: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)',
    'absences': 'Number of school absences',
    'failures': 'Number of past class failures',
    'G1': 'First period grade (0-20 scale)',
    'G2': 'Second period grade (0-20 scale)',
    'Medu': "Mother's education: 0 (none), 1 (primary), 2 (5th-9th), 3 (secondary), 4 (higher)",
    'Fedu': "Father's education: 0 (none), 1 (primary), 2 (5th-9th), 3 (secondary), 4 (higher)",
    'traveltime': 'Home to school travel time: 1 (<15min), 2 (15-30min), 3 (30min-1h), 4 (>1h)'
}

inputs = {}
for feature in features:
    if feature in ['G1', 'G2']:
        inputs[feature] = st.sidebar.slider(
            feature_labels[feature],
            min_value=0.0,
            max_value=20.0,
            value=10.0,
            step=0.5,
            help=feature_descriptions[feature]
        )
    elif feature == 'studytime':
        inputs[feature] = st.sidebar.slider(
            feature_labels[feature],
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=1.0,
            help=feature_descriptions[feature]
        )
    elif feature == 'traveltime':
        inputs[feature] = st.sidebar.slider(
            feature_labels[feature],
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=1.0,
            help=feature_descriptions[feature]
        )
    elif feature in ['Medu', 'Fedu']:
        inputs[feature] = st.sidebar.slider(
            feature_labels[feature],
            min_value=0.0,
            max_value=4.0,
            value=2.0,
            step=1.0,
            help=feature_descriptions[feature]
        )
    else:
        inputs[feature] = st.sidebar.number_input(
            feature_labels[feature],
            min_value=0.0,
            value=0.0,
            step=1.0,
            help=feature_descriptions[feature]
        )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Current Input Values")
    input_df = pd.DataFrame({
        'Feature': [feature_labels[f] for f in features],
        'Value': [inputs[f] for f in features]
    })
    st.dataframe(input_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("ℹ️ About")
    st.info("""
    This app uses 4 ML models:
    - KNN Classifier
    - Logistic Regression
    - KNN Regressor
    - Linear Regression
    """)

# Predict button
st.markdown("---")
if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
    # Prepare data
    data = np.array([inputs[feat] for feat in features]).reshape(1, -1)
    data_scaled = models['scaler'].transform(data)
    
    # Get predictions
    knn_class = models['knn_clf'].predict(data_scaled)[0]
    log_class = models['log_reg'].predict(data_scaled)[0]
    knn_marks = models['knn_reg'].predict(data_scaled)[0]
    lin_marks = models['lin_reg'].predict(data_scaled)[0]
    
    # Display results
    st.header("📈 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Classification (Pass/Fail)")
        knn_color = "🟢" if knn_class == "Pass" else "🔴"
        log_color = "🟢" if log_class == "Pass" else "🔴"
        st.metric("KNN Prediction", f"{knn_color} {knn_class}")
        st.metric("Logistic Regression", f"{log_color} {log_class}")
    
    with col2:
        st.subheader("📊 Regression (Final Marks)")
        st.metric("KNN Regression", f"{knn_marks:.2f}", delta=f"{knn_marks-10:.2f}")
        st.metric("Linear Regression", f"{lin_marks:.2f}", delta=f"{lin_marks-10:.2f}")
    
    # Summary
    avg_marks = (knn_marks + lin_marks) / 2
    st.markdown("---")
    
    if avg_marks >= 10:
        st.success(f"✅ **Average Predicted Final Marks: {avg_marks:.2f}** - Student is likely to PASS")
        st.balloons()
    else:
        st.warning(f"⚠️ **Average Predicted Final Marks: {avg_marks:.2f}** - Student may need additional support")
    
    # Detailed comparison
    with st.expander("📊 Detailed Model Comparison"):
        comparison_df = pd.DataFrame({
            'Model': ['KNN Classifier', 'Logistic Regression', 'KNN Regressor', 'Linear Regression'],
            'Type': ['Classification', 'Classification', 'Regression', 'Regression'],
            'Prediction': [knn_class, log_class, f"{knn_marks:.2f}", f"{lin_marks:.2f}"]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Adjust Input Values**: Use the sidebar sliders to input student information
    2. **Click Predict**: Press the "Predict Performance" button
    3. **View Results**: 
       - Classification shows Pass/Fail predictions
       - Regression shows predicted final marks (G3)
    4. **Interpret Results**: 
       - Marks ≥ 10 typically indicate passing
       - Compare predictions from different models
    """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Student Performance Prediction using Machine Learning")