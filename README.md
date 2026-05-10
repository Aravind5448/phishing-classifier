# Phishing URL Classifier

**Live Demo:** [phishing-classifier-project.streamlit.app](https://phishing-classifier-project.streamlit.app/)  
**Dataset:** [UCI PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

A machine learning pipeline that classifies URLs as legitimate or phishing based on 13 extracted lexical and structural features. The project utilizes a decoupled architecture, separating a Streamlit frontend client from a FastAPI inference backend.

## 🏗️ Architecture & Tech Stack

* **Backend (Inference API):** FastAPI
* **Frontend (Client UI):** Streamlit
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, Joblib (Multiprocessing)
* **Validation & Security:** Pydantic (Data schema enforcement), SlowAPI (Rate limiting)
* **Deployment & CI/CD:** Docker, Render (Backend), Streamlit Cloud (Frontend), GitHub Actions

## 🔍 Feature Engineering

The model does not rely on page content or external database lookups. It evaluates the URL string itself using 13 engineered features designed to identify common obfuscation and Domain Generation Algorithm (DGA) patterns. 

| Feature Category | Features Extracted | Rationale |
| :--- | :--- | :--- |
| **Base Metrics** | URL Length, Domain Length, Subdomain Count | Phishing links often use excessively long URLs or subdomains to hide the root domain. |
| **Punctuation** | Dot Count, Hyphen Count, Special Character Ratio | Attackers frequently use hyphens or special characters to mimic legitimate brands (e.g., `secure-login-apple`). |
| **Structural** | Has IP Address, Is HTTPS | Using an IP instead of a domain name is a strong indicator of malicious intent. |
| **Obfuscation** | Punycode (`xn--`), Shannon Entropy | Detects homograph attacks and measures string randomness to identify algorithmically generated domains. |
| **Linguistic** | Suspicious Keywords, Vowel/Consonant Ratio, Longest Consonant Sequence | Flags common social engineering terms (`login`, `verify`) and detects unnatural character groupings. |

## 📊 Model Performance

The XGBoost classifier was trained on 235,795 rows of the PhiUSIIL dataset. To handle the high volume of string operations during feature extraction, the preprocessing pipeline utilizes CPU multiprocessing via `joblib.Parallel`.

* **Validation Accuracy:** 99.64%
* **F1-Score:** 1.00

*Note: While the model achieves >99% accuracy on the PhiUSIIL dataset, this specific dataset contains overlapping domain structures. In a live production environment facing novel threats, real-world accuracy is expected to be lower due to adversarial drift.*

## 🚀 Local Setup & Installation

To run this project on your local machine, you will need to run the backend and frontend simultaneously in two separate terminal windows.

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/aravind5448/phishing-classifier.git](https://github.com/aravind5448/phishing-classifier.git)
cd phishing-classifier
pip install -r requirements.txt

**2. Start the FastAPI Backend (Terminal 1):**

```bash
uvicorn api:app --reload
The API will be available at http://localhost:8000
```

**3. Start the Streamlit Frontend (Terminal 2):**

```bash
# Open app.py and ensure API_URL is set to "http://localhost:8000/predict" before running
streamlit run app.py
The UI will be available at http://localhost:8501