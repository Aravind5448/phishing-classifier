import streamlit as st
import requests

# --- 1. System Configuration ---
st.set_page_config(
    page_title="Phishing Classifier Client",
    page_icon="🛡️",
    layout="centered"
)

# Network Configuration: Pointing to the local FastAPI microservice
API_URL = "http://localhost:8000/predict"

# --- 2. User Interface Rendering ---
st.title("🛡️ Zero-Day Phishing URL Classifier")
st.markdown("""
**Enterprise Frontend Architecture:** This UI acts strictly as a lightweight HTTP client. 
All lexical extraction, strict Pydantic validation, and GPU-accelerated XGBoost inference are executed 
on the decoupled FastAPI microservice.
""")

url_input = st.text_input("Target URL", placeholder="https://secure-login.verify-account.com")

# --- 3. Execution (The API Handshake) ---
if st.button("Initiate Scan", type="primary"):
    if not url_input.strip():
        st.error("Validation Error: Please enter a valid URL string.")
    else:
        with st.spinner("Pinging inference API..."):
            try:
                # The crucial architectural shift: Transmitting a JSON payload to the Backend.
                payload = {"url": url_input}
                response = requests.post(API_URL, json=payload, timeout=10)
                
                # Check for HTTP errors (e.g., 422 Validation errors from Pydantic)
                response.raise_for_status()
                
                # Parse the strict InferenceResponse schema from FastAPI
                data = response.json()
                prediction = data.get("prediction")
                confidence = data.get("confidence")
                features = data.get("features")

                st.divider()
                
                # Results output
                if prediction == 1: 
                    st.error(f"## 🚨 HIGH RISK: Phishing Detected")
                    st.metric("Threat Confidence", f"{confidence:.2f}%")
                else:
                    st.success(f"## ✅ LOW RISK: URL Appears Safe")
                    st.metric("Safety Confidence", f"{confidence:.2f}%")
                
                with st.expander("📊 View Validated Lexical Tensors"):
                    st.json(features)
                    st.caption("Note: These features were extracted and strictly validated by the FastAPI backend.")

            except requests.exceptions.ConnectionError:
                st.error("🚨 Critical Failure: Unable to connect to the Inference Engine. Ensure `uvicorn api:app --reload` is running on port 8000.")
            except requests.exceptions.HTTPError as e:
                # Gracefully display API rejections (e.g., if Pydantic rejects the input)
                st.error(f"⚠️ API Rejection: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                st.error(f"Client Exception: {e}")