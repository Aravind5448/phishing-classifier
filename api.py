from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize the rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Zero-Day Phishing Inference API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Protect the endpoint
@app.post("/predict")
@limiter.limit("5/minute")
async def predict_url(request: Request, payload: URLInferenceRequest):
    # ... existing inference logic ...

# Load artifacts on startup
try:
    MODEL = joblib.load("models/phishing_model.pkl")
    MODEL.set_params(device='cpu') # Ensure CPU inference for web threads
    FEATURE_NAMES = joblib.load("models/feature_names.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load ML artifacts. Have you run train.py? Error: {e}")

@app.post("/predict", response_model=InferenceResponse)
async def predict_url(request: URLInferenceRequest):
    try:
        # 1. Feature Extraction
        raw_features = extract_features(request.url)
        
        # 2. Schema Validation (Pydantic protects the model here)
        validated_features = URLFeatures(**raw_features)
        
        # 3. Tensor Alignment
        input_dict = validated_features.model_dump()
        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES).fillna(0)
        
        # 4. Inference
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