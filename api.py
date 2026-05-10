import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.feature_extractor import extract_features
from src.schemas import URLInferenceRequest, URLFeatures, InferenceResponse

# 1. Initialize the Rate Limiter (Protection against botnet DDoS)
limiter = Limiter(key_func=get_remote_address)

# 2. System Configuration
app = FastAPI(
    title="Zero-Day Phishing Inference API",
    description="MLOps-ready backend for lexical URL analysis",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 3. Memory & Asset Management
try:
    MODEL = joblib.load("models/phishing_model.pkl")
    MODEL.set_params(device='cpu') # Strip GPU binding for web thread stability
    FEATURE_NAMES = joblib.load("models/feature_names.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load ML artifacts. Have you run train.py? Error: {e}")

# 4. The Inference Endpoint
@app.post("/predict", response_model=InferenceResponse)
@limiter.limit("5/minute")
async def predict_url(request: Request, payload: URLInferenceRequest):
    try:
        # A. Feature Extraction
        raw_features = extract_features(payload.url)
        
        # B. Schema Validation (Pydantic protects the model from garbage data)
        validated_features = URLFeatures(**raw_features)
        
        # C. Tensor Alignment
        input_dict = validated_features.model_dump()
        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES).fillna(0)
        
        # D. Gradient Boosted Inference
        prediction = int(MODEL.predict(input_df)[0])
        proba = MODEL.predict_proba(input_df)[0]
        confidence = round(float(max(proba) * 100), 2)
        
        return InferenceResponse(
            prediction=prediction,
            confidence=confidence,
            features=validated_features,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")