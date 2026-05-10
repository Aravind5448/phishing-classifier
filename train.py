import os
import sys
import logging
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.feature_extractor import extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}. Terminating.")
        sys.exit(1)

    logger.info("Loading FULL PhiUSIIL dataset into memory...")
    df = pd.read_csv(filepath)
    
    url_col = 'URL' if 'URL' in df.columns else df.columns[0]
    target_col = 'label' if 'label' in df.columns else 'Result'

    # The Multiprocessing Upgrade
    logger.info(f"Initiating multiprocessed feature extraction on {len(df)} rows. Utilizing all CPU cores...")
    
    # joblib.Parallel splits the 235,000 URLs across your CPU threads automatically
    features_list = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(extract_features)(str(url)) for url in df[url_col]
    )
    
    X = pd.DataFrame(features_list)
    y = df[target_col]

    return X, y

def build_and_train_pipeline():
    DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_and_preprocess_data(DATA_PATH)
    
    logger.info("Partitioning tensors (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info("Transferring workload to RTX GPU for Gradient Boosting...")
    model = XGBClassifier(
        n_estimators=300,        # Increased trees for complex feature interactions
        max_depth=7,             # Slightly deeper trees to catch edge cases
        learning_rate=0.05,      # Slower learning rate for better generalization
        tree_method='hist',
        device='cuda',           # Explicit RTX utilization
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    logger.info("Executing validation inferences...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    logger.info(f"Validation Accuracy on {len(X_test)} validation samples: {acc * 100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions))

    logger.info("Serializing updated 13-feature model artifacts...")
    joblib.dump(model, os.path.join(MODEL_DIR, 'phishing_model.pkl'))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    logger.info("Enterprise Pipeline execution completed.")

if __name__ == "__main__":
    build_and_train_pipeline()