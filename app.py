import streamlit as st
import requests

st.set_page_config(page_title="Phishing Classifier", page_icon="🛡️", layout="centered")

# Note: We will change this to your live Render URL in the next step
API_URL = "https://phishing-classifier-api.onrender.com/predict"

st.title("🛡️ Zero-Day Phishing URL Classifier")
st.markdown("""
This UI is a lightweight client connected to a decoupled FastAPI backend. 
Lexical extraction, Pydantic schema validation, and XGBoost inference are handled by the API.
""")

url_input = st.text_input("Target URL", placeholder="https://secure-login.verify-account.com")

if st.button("Initiate Scan", type="primary"):
    if not url_input.strip():
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("Analyzing URL..."):
            try:
                response = requests.post(API_URL, json={"url": url_input}, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                prediction = data.get("prediction")
                confidence = data.get("confidence")
                features = data.get("features")

                st.divider()
                
                if prediction == 0: 
                    st.error(f"## 🚨 HIGH RISK: Phishing Detected")
                    st.metric("Threat Confidence", f"{confidence:.2f}%")
                else:
                    st.success(f"## ✅ LOW RISK: URL Appears Safe")
                    st.metric("Safety Confidence", f"{confidence:.2f}%")
                
                with st.expander("📊 View Extracted Features"):
                    st.json(features)
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Error: Unable to connect to the backend API.")
            except requests.exceptions.HTTPError as e:
                st.error(f"⚠️ API Error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                st.error(f"Client Error: {e}")