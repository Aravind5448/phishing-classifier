"""
Source module for Phishing URL Classification.
Exposes the core feature extraction pipeline.
"""
from .feature_extractor import extract_features, calculate_entropy

# Defines exactly what is exported when someone runs `from src import *`
__all__ = ["extract_features", "calculate_entropy"]