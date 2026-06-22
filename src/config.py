import os

# Configuration settings for the News Article Classification System

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# File Paths
DATASET_CSV = os.path.join(DATA_RAW_DIR, "news_dataset.csv")
VECTORIZER_PKL = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SVM_MODEL_PKL = os.path.join(MODELS_DIR, "svm_model.pkl")
LR_MODEL_PKL = os.path.join(MODELS_DIR, "lr_model.pkl")
NB_MODEL_PKL = os.path.join(MODELS_DIR, "nb_model.pkl")
RF_MODEL_PKL = os.path.join(MODELS_DIR, "rf_model.pkl")

EVAL_RESULTS_CSV = os.path.join(RESULTS_DIR, "evaluation_results.csv")
CONFUSION_MATRIX_CSV = os.path.join(RESULTS_DIR, "confusion_matrix_svm.csv")

# ML Hyperparameters
VOCAB_SIZE = 50000
NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Categories Mapping
CATEGORIES = ["Politics", "Sports", "Technology", "Entertainment", "Business"]

# Category Colors mapping for the presentation UI
CATEGORY_COLORS = {
    "Politics": "#1a56db",      # Blue
    "Sports": "#057a55",        # Green
    "Technology": "#7e3af2",    # Purple
    "Entertainment": "#d97706", # Orange
    "Business": "#e02424"       # Red
}
