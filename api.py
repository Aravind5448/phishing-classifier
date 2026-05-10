import joblib
import pandas as pd
import tldextract
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.feature_extractor import extract_features
from src.schemas import URLInferenceRequest, URLFeatures, InferenceResponse

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Zero-Day Phishing Inference API", version="1.0.1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 1. Enterprise Safelist (The False-Positive Shield)
VERIFIED_DOMAINS = {
    "github.com", "google.com", "apple.com", "microsoft.com", 
    "amazon.com", "linkedin.com", "streamlit.io", "render.com"
}

try:
    MODEL = joblib.load("models/phishing_model.pkl")
    MODEL.set_params(device='cpu')
    FEATURE_NAMES = joblib.load("models/feature_names.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load ML artifacts: {e}")

@app.post("/predict", response_model=InferenceResponse)
@limiter.limit("5/minute")
async def predict_url(request: Request, payload: URLInferenceRequest):
    try:
        # A. Pre-Processing & Feature Extraction
        ext = tldextract.extract(payload.url)
        root_domain = f"{ext.domain}.{ext.suffix}".lower()
        
        raw_features = extract_features(payload.url)
        validated_features = URLFeatures(**raw_features)
        
        # B. THE HEURISTIC FIREWALL
        # Rule 1: Auto-Pass Known Verified Domains (Solves the GitHub False Positive)
        if root_domain in VERIFIED_DOMAINS:
            return InferenceResponse(
                prediction=1, # 1 = Safe
                confidence=100.00,
                features=validated_features,
                status="heuristic_safelist_override"
            )
            
        # Rule 2: Auto-Block Raw IP Addresses
        if raw_features["Has_IP"] == 1:
            return InferenceResponse(
                prediction=0, # 0 = Phishing
                confidence=100.00,
                features=validated_features,
                status="heuristic_blocklist_override"
            )

        # C. ML Inference (Only for Unknown Domains)
        input_dict = validated_features.model_dump()
        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES).fillna(0)
        
        prediction = int(MODEL.predict(input_df)[0])
        proba = MODEL.predict_proba(input_df)[0]
        confidence = round(float(max(proba) * 100), 2)
        
        return InferenceResponse(
            prediction=prediction,
            confidence=confidence,
            features=validated_features,
            status="ml_inference_success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")