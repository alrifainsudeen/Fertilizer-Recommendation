import streamlit as st
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier # Keep import for joblib to load correctly
from sklearn.pipeline import Pipeline # Keep import for joblib to load correctly
from sklearn.preprocessing import LabelEncoder # Keep import for joblib to load correctly
import joblib
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
try:
    warnings.filterwarnings('ignore', category=pd.SettingWithCopyWarning)
except:
    pass 

# Define file names
MODEL_FILE = "fertilizer_model.joblib"
TARGET_ENCODER_FILE = "target_encoder.joblib"
HIGH_CARD_ENCODERS_FILE = "high_card_encoders.joblib"
FEATURE_COLUMNS_FILE = "feature_columns.joblib"
CATEGORY_DATA_FILE = "category_data.joblib"

# --- 1. Model Loading Function ---
@st.cache_resource
def load_model_components():
    """Loads pre-trained model, encoders, and feature list from disk."""
    
    required_files = [MODEL_FILE, TARGET_ENCODER_FILE, HIGH_CARD_ENCODERS_FILE, FEATURE_COLUMNS_FILE, CATEGORY_DATA_FILE]
    
    # Check if all files exist
    if not all(os.path.exists(f) for f in required_files):
        st.error("Model files not found! Please run 'python train_model.py' first.")
        # Return Nones to prevent crash
        return None, None, None, None, None, None 

    try:
        with st.spinner("Loading model components..."):
            model = joblib.load(MODEL_FILE)
            le_target = joblib.load(TARGET_ENCODER_FILE)
            high_card_encoders = joblib.load(HIGH_CARD_ENCODERS_FILE)
            X_cols = joblib.load(FEATURE_COLUMNS_FILE)
            category_data = joblib.load(CATEGORY_DATA_FILE)

            # Extract UI data 
            soil_types = category_data.get('Soil Type', ['N/A'])
            crop_types = category_data.get('Crop Type', ['N/A'])
            
            # The app expects 6 return values, adjust the return order based on how it's unpacked later
            return model, le_target, X_cols, high_card_encoders, soil_types, crop_types
        
    except Exception as e:
        st.error(f"Error loading model files: {e}. Check if Python/library versions match the training environment.")
        return None, None, None, None, None, None

# --- 2. Streamlit UI and Prediction Logic ---

st.set_page_config(page_title="Fertilizer Recommendation App", layout="wide")

st.title("🌱 Smart Fertilizer Recommender")
st.markdown("""
Enter the soil and crop conditions below to get the recommended fertilizer.
---
""")

# Load the model and related objects (INSTANT LOAD)
model, le_target, X_cols, high_card_encoders, soil_types, crop_types = load_model_components()

# Check for successful loading (model is not None)
if model is not None:
    
    # We need to correctly identify high card columns from the encoders dictionary keys
    high_card_cols = list(high_card_encoders.keys()) 

    # --- Input Fields using Streamlit Form ---
    with st.form("input_form"):
        st.header("1. Environmental & Chemical Readings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp = st.number_input("Temperature (°C)", min_value=1.0, max_value=50.0, value=25.0, step=0.1)
            nitro = st.number_input("Nitrogen (N)", min_value=0, max_value=100, value=10)
        
        with col2:
            humi = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0, step=0.1)
            potass = st.number_input("Potassium (K)", min_value=0, max_value=100, value=5)
            
        with col3:
            moist = st.number_input("Moisture (%)", min_value=10.0, max_value=100.0, value=30.0, step=0.1)
            phos = st.number_input("Phosphorous (P)", min_value=0, max_value=100, value=20)
            
        st.header("2. Crop & Soil Information")
        col4, col5 = st.columns(2)
        
        with col4:
            soil_type = st.selectbox("Soil Type", options=soil_types)
            
        with col5:
            crop_type = st.selectbox("Crop Type", options=crop_types)

        submitted = st.form_submit_button("Get Fertilizer Recommendation 🔬")

    # --- Prediction Logic ---
    if submitted:
        # 1. Gather all inputs
        input_data = {
            'Temperature': temp, 'Humidity': humi, 'Moisture': moist, 
            'Nitrogen': nitro, 'Potassium': potass, 'Phosphorous': phos,
            'Soil Type': soil_type, 'Crop Type': crop_type
        }

        # 2. Feature Engineering
        if 'Temp_Moisture' in X_cols:
            input_data['Temp_Moisture'] = input_data['Temperature'] * input_data['Moisture']

        # 3. Create DataFrame and enforce column order
        sample_df = pd.DataFrame([input_data])
        sample_df = sample_df[X_cols] 

        # 4. Apply Label Encoding for high-cardinality columns
        for col in high_card_cols:
            le = high_card_encoders[col]
            sample_value = str(sample_df[col].iloc[0]) 
            
            if sample_value not in le.classes_:
                 st.warning(f"Category '{sample_value}' in {col} was not seen during training. Prediction might be less reliable.")
                 sample_df.loc[:, col] = -1 
            else:
                 sample_df.loc[:, col] = le.transform([sample_value])[0]
        
        # 5. Predict
        try:
            with st.spinner("Calculating recommendation..."):
                prediction = model.predict(sample_df)
                fertilizer_name = le_target.inverse_transform(prediction)
            
            st.success(f"## **✅ Recommended Fertilizer:** {fertilizer_name[0]}")
            
        except Exception as e:
            st.error(f"Prediction Error: Could not generate recommendation. Details: {e}")
