# 🛡️ Zero-Day Phishing URL Classifier

> An end-to-end Machine Learning pipeline that detects malicious URLs in real-time using structural mathematics and gradient boosted inference.

## 📌 Architecture Overview
Traditional phishing detectors rely on static blacklists, making them vulnerable to "Zero-Day" attacks (newly registered malicious domains). This project solves that by analyzing the mathematical and lexical structure of the URL itself—no page visit or blacklist lookup required.

### **Core Engineering Features:**
* **Information Theory (Shannon Entropy):** Implements $H(X) = -\sum P(x_i) \log_2 P(x_i)$ to measure string randomness, successfully identifying bot-generated Domain Generation Algorithms (DGAs).
* **Homograph Detection:** Parses URLs for Punycode (`xn--`) to detect advanced brand-impersonation attacks.
* **Gradient Boosted Inference:** Utilizes an **XGBoost** classifier (trained with GPU acceleration) to handle non-linear feature interactions, achieving **99.27% validation accuracy** on the PhiUSIIL dataset.
* **Asynchronous Web UI:** Deployed via Streamlit with a custom caching layer that strips CUDA hardware bindings to ensure thread-safe, CPU-based web inference.

## 🚀 Technical Stack
* **Language:** Python 3.12
* **Machine Learning:** Scikit-Learn, XGBoost
* **Feature Engineering:** tldextract, urllib, math (Entropy)
* **Frontend:** Streamlit
* **Data Persistence:** Joblib

## 🧠 Model Performance
The model was trained on a downsampled, stratified subset of the UCI PhiUSIIL dataset to ensure balanced class representation.
* **Accuracy:** 99.27%
* **Precision/Recall (Phishing):** 0.99 / 0.99
* **Precision/Recall (Legitimate):** 0.99 / 1.00

---
*Developed by Aravind D.*