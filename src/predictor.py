import os
import joblib
import config
from preprocess import preprocess_text

# Global references for loaded model and vectorizer
_vectorizer = None
_classifier = None

def load_models():
    """
    Loads the serialized SVM classifier and TF-IDF vectorizer on startup.
    Raises FileNotFoundError or other Exceptions if files are missing or corrupt.
    """
    global _vectorizer, _classifier
    
    if not os.path.exists(config.VECTORIZER_PKL):
        raise FileNotFoundError(f"TF-IDF Vectorizer file not found at: {config.VECTORIZER_PKL}")
    if not os.path.exists(config.SVM_MODEL_PKL):
        raise FileNotFoundError(f"SVM Model file not found at: {config.SVM_MODEL_PKL}")
        
    _vectorizer = joblib.load(config.VECTORIZER_PKL)
    _classifier = joblib.load(config.SVM_MODEL_PKL)
    print("Inference models loaded successfully.")

# Proactively try to load models on import.
# Web server will catch exceptions on startup if models are not present.
try:
    load_models()
except Exception as e:
    print(f"Warning: Models could not be loaded on import ({str(e)}). Ensure train_evaluate.py has been run.")

def predict_category(text: str) -> dict:
    """
    Preprocesses raw news article text, vectorizes it, and returns predicted category
    and probability confidence score.
    
    Args:
        text (str): Raw article text.
        
    Returns:
        dict: A dictionary containing 'category' (str) and 'confidence' (float 0.0-1.0).
    """
    if not text or not text.strip():
        raise ValueError("Article text cannot be empty.")
        
    global _vectorizer, _classifier
    if _vectorizer is None or _classifier is None:
        # Try reloading once in case they weren't loaded on import
        load_models()
        
    # Preprocess using the identical pipeline
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text.strip():
        raise ValueError("Article contains no valid words for classification.")
        
    # Transform to TF-IDF features (single sample)
    vec_text = _vectorizer.transform([cleaned_text])
    
    # Predict label
    predicted_label = _classifier.predict(vec_text)[0]
    
    # Compute confidence using predict_proba() enabled by CalibratedClassifierCV
    probabilities = _classifier.predict_proba(vec_text)[0]
    class_index = list(_classifier.classes_).index(predicted_label)
    confidence = probabilities[class_index]
    
    return {
        "category": str(predicted_label).title(),
        "confidence": float(confidence)
    }
