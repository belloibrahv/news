from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import config

def get_tfidf_vectorizer():
    """
    Initializes and returns a TfidfVectorizer configured according to specs:
    - max_features = 50,000
    - ngram_range = (1, 2)
    - sublinear_tf = True
    """
    return TfidfVectorizer(
        max_features=config.VOCAB_SIZE,
        ngram_range=config.NGRAM_RANGE,
        sublinear_tf=True
    )

def save_vectorizer(vectorizer, filepath):
    """
    Serializes and saves the fitted vectorizer to disk.
    """
    joblib.dump(vectorizer, filepath)
    print(f"TF-IDF Vectorizer saved to: {filepath}")

def load_vectorizer(filepath):
    """
    Loads and returns a serialized vectorizer from disk.
    """
    return joblib.load(filepath)
