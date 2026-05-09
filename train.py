import os
import sys
import logging
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# System Configuration
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.feature_extractor import extract_features

# Configure Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath: str, sample_size: int = 50000) -> pd.DataFrame:
    """
    Loads raw data and enforces strict feature alignment by re-extracting 
    features from the raw URLs using the production extractor.
    """
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}. Terminating execution.")
        sys.exit(1)

    logger.info("Loading PhiUSIIL dataset into memory...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        sys.exit(1)

    # Validate essential columns
    # The PhiUSIIL dataset typically uses 'URL' and 'label'
    url_col = 'URL' if 'URL' in df.columns else df.columns[0]
    target_col = 'label' if 'label' in df.columns else 'Result'

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' missing from dataset schema.")
        sys.exit(1)

    # Optimization: Sample data to prevent pipeline bottleneck during development
    if len(df) > sample_size:
        logger.info(f"Downsampling dataset from {len(df)} to {sample_size} for optimal epoch duration.")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info("Executing mathematical feature extraction on raw URLs. This may take a moment...")
    
    # Force alignment: Apply the production extractor to the training data
    extracted_data = df[url_col].apply(lambda x: pd.Series(extract_features(str(x))))
    
    # Construct final matrix
    X = extracted_data
    y = df[target_col]

    return X, y

def build_and_train_pipeline():
    """
    Main orchestration function for model training, evaluation, and serialization.
    """
    DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
    MODEL_DIR = "models"
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Data Ingestion
    X, y = load_and_preprocess_data(DATA_PATH)
    
    # 2. Train/Test Split (Stratified to maintain class distributions)
    logger.info("Partitioning tensors for training and validation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Model Architecture (XGBoost with GPU Acceleration)
    logger.info("Initializing XGBoost classifier with hardware acceleration parameters...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',      # Optimized histogram formulation
        device='cuda',           # Targets the local GPU (RTX 4050)
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    # 4. Execution
    logger.info("Commencing model training...")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logger.warning(f"GPU initialization failed. Falling back to CPU processing. Error: {e}")
        # Fallback to CPU if CUDA is not configured properly in the local environment
        model.set_params(device='cpu')
        model.fit(X_train, y_train)

    # 5. Evaluation
    logger.info("Executing validation inferences...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    logger.info(f"Validation Accuracy: {acc * 100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions))

    # 6. Artifact Serialization
    logger.info("Serializing model artifacts for production deployment...")
    joblib.dump(model, os.path.join(MODEL_DIR, 'phishing_model.pkl'))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    build_and_train_pipeline()