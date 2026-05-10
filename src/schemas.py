from pydantic import BaseModel, Field

class URLInferenceRequest(BaseModel):
    url: str = Field(..., min_length=4, description="The raw URL to be analyzed")

class URLFeatures(BaseModel):
    """Strict schema for the XGBoost tensor input."""
    URL_Length: int = Field(..., ge=0)
    Domain_Length: int = Field(..., ge=0)
    Is_HTTPS: int = Field(..., ge=0, le=1)
    Num_Dots: int = Field(..., ge=0)
    Num_Hyphens: int = Field(..., ge=0)
    Num_Subdomains: int = Field(..., ge=0)
    Entropy: float = Field(..., ge=0.0)
    Has_IP: int = Field(..., ge=0, le=1)
    Punycode: int = Field(..., ge=0, le=1)
    
    # New Features Validation
    Suspicious_Keywords: int = Field(..., ge=0)
    VC_Ratio: float = Field(..., ge=0.0)
    Longest_Consonant_Seq: int = Field(..., ge=0)
    Special_Char_Ratio: float = Field(..., ge=0.0)

class InferenceResponse(BaseModel):
    prediction: int
    confidence: float
    features: URLFeatures
    status: str