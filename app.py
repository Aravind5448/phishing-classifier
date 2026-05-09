import os
import sys
import pandas as pd
import streamlit as st
import joblib

# Integrity Fix 1: Ensure the src module is always discoverable regardless of where the app is launched from.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.feature_extractor import extract_features

# --- 1. System Configuration ---
st.set_page_config(
    page_title="Phishing Classifier Pipeline",
    page_icon="🛡️",
    layout="centered"
)

# --- 2. Memory & Asset Management ---
@st.cache_resource
def load_artifacts():
    """
    Loads the ML model and feature schema into memory ONLY once.
    Prevents memory leaks and latency spikes on subsequent scans.
    """
    model_path = "models/phishing_model.pkl"
    features_path = "models/feature_names.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return None, None
        
    try:
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        
        # FIX: Strip the RTX 4050 GPU binding for stable web inference
        model.set_params(device='cpu') 
        
        return model, feature_names
    except Exception as e:
        st.error(f"System Error: Corrupted model artifacts. Details: {e}")
        return None, None

model, feature_names = load_artifacts()

# --- 3. User Interface Rendering ---
st.title("🛡️ Zero-Day Phishing URL Classifier")
st.markdown("""
This production-grade pipeline evaluates URLs in real-time using mathematical 
feature extraction (Shannon Entropy) and gradient boosted inference.
""")

# Integrity Fix 2: Hard-stop the UI if the engine isn't ready.
if model is None or feature_names is None:
    st.warning("⚠️ **System Offline:** Model artifacts not found in `/models`. Please execute `train.py` first.")
    st.stop()

url_input = st.text_input("Target URL", placeholder="https://secure-login.verify-account.com")

# --- 4. Execution Pipeline ---
if st.button("Initiate Scan", type="primary"):
    # Integrity Fix 3: Input sanitization
    if not url_input.strip():
        st.error("Validation Error: Please enter a valid URL string.")
    else:
        with st.spinner("Executing lexical extraction and model inference..."):
            try:
                # Step A: Live Extraction
                raw_features = extract_features(url_input)
                
                # Step B: Feature Alignment (Closes the primary loop-hole)
                # We map our extracted features to the EXACT columns the model expects.
                # Missing columns default to 0 to prevent tensor shape mismatches.
                aligned_features = {col: 0 for col in feature_names}
                for key, value in raw_features.items():
                    if key in aligned_features:
                        aligned_features[key] = value
                        
                input_df = pd.DataFrame([aligned_features], columns=feature_names)
                
                # Step C: Inference
                prediction = model.predict(input_df)[0]
                
                # Handle models that support probability scores (like XGBoost/RF)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0]
                    confidence = max(proba) * 100
                else:
                    confidence = 100.0
                
                # Step D: Results Output
                st.divider()
                
                # Note: In most datasets, 1 = Legitimate, 0 = Phishing (Adjusted for PhiUSIIL)
                # PhiUSIIL Mapping: 1 = Phishing, 0 = Legitimate
                if prediction == 1: 
                    st.error(f"## 🚨 HIGH RISK: Phishing Detected")
                    st.metric("Threat Confidence", f"{confidence:.2f}%")
                else:
                    st.success(f"## ✅ LOW RISK: URL Appears Safe")
                    st.metric("Safety Confidence", f"{confidence:.2f}%")
                
                # Step E: Explainability (Crucial for Interviews)
                with st.expander("📊 View Lexical Analysis Breakdown"):
                    st.json(raw_features)
                    st.caption("Note: Entropy scores > 4.0 often indicate Domain Generation Algorithms (DGAs).")
                    
            except Exception as e:
                st.error(f"Runtime Exception during analysis: {e}")