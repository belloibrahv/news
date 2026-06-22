# News Article Classification System

**Institution:** Tai Solarin Federal University of Education (TASUED), Department of Computer and Information Science  
**Author:** Otasanya Jamiu Olamilekan (Matric No: 20220204057)  
**Supervisor:** Dr. G.O. Odulaja  
**Date:** 2026  

---

## Project Overview
This repository contains a fully automated, machine learning-driven news article classification pipeline. The system assigns English-language news articles to one of five predefined topical categories: **Politics, Sports, Technology, Entertainment, and Business**. 

The system trains, evaluates, and serializes four classical ML classifiers: **Naive Bayes, Support Vector Machine (LinearSVC), Logistic Regression, and Random Forest**. The best classifier (SVM) is deployed via a web-based prototype application styled with Google Material Design 3 guidelines.

---

## Directory Structure
```
news/
├── app/
│   ├── __init__.py             # Flask application factory
│   ├── routes.py               # Blueprint route handlers
│   └── templates/
│       ├── index.html          # Submission page (Material UI)
│       ├── result.html         # Result page (Material UI)
│       └── error.html          # Error page (Material UI)
├── data/
│   └── raw/
│       └── news_dataset.csv    # 7,500 balanced news articles
├── models/
│   ├── tfidf_vectorizer.pkl    # Serialized fitted TF-IDF vectorizer
│   ├── svm_model.pkl           # Calibrated SVM model (best performer)
│   ├── lr_model.pkl            # Logistic Regression model
│   ├── nb_model.pkl            # Naive Bayes model
│   └── rf_model.pkl            # Random Forest model
├── results/
│   ├── evaluation_results.csv  # Test metrics table for all 4 models
│   └── confusion_matrix_svm.csv# Confusion matrix for SVM
├── src/
│   ├── config.py               # Constants, file paths, and hyperparameters
│   ├── preprocess.py           # Reusable 7-step text preprocessing pipeline
│   ├── feature_extraction.py   # TF-IDF Vectorizer configuration
│   ├── train_evaluate.py       # Dataset splitting, model training & evaluation
│   ├── predictor.py            # Article classifier inference module
│   ├── system_test.py          # 10 manual test case checks
│   └── test_suite.py           # 10 automated unit test cases
├── requirements.txt            # Dependency manifest
├── run_ci.sh                   # Local CI/CD validator script
└── README.md                   # This documentation
```

---

## Local Development Setup

### 1. Prerequisites
- Python 3.9+ (Python 3.11 recommended)
- Internet access (for downloading AG News dataset and NLTK dependencies)

### 2. Environment Initialization
Clone or navigate to the directory and initialize a virtual environment:
```bash
# Navigate to the workspace
cd news

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download required NLTK corpus datasets
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### 3. Generate Dataset
Build the balanced corpus combining AG News and programmatically synthesized local Nigerian content:
```bash
python src/generate_dataset.py
```
This produces `data/raw/news_dataset.csv` containing exactly 7,500 articles (1,500 per category).

### 4. Run Model Training & Evaluation
Train all four classifiers on the preprocessed dataset and generate metrics reports:
```bash
python src/train_evaluate.py
```
This prints the comparison table and saves evaluation results and serialized models to disk.

### 5. Launch the Flask Web Prototype
Start the web application:
```bash
python -m flask --app app run
```
Open a browser and navigate to `http://127.0.0.1:5000/` to test the Material Design submission interface.

---

## Testing & CI/CD
A validation script simulates the CI/CD pipeline to verify unit tests, check shape leakage, and test system correctness:
```bash
# Make script executable and run
chmod +x run_ci.sh
./run_ci.sh
```

---

## Model Evaluation Summary
All models are evaluated on a stratified test set (15% held-out). Performance results:

| Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1 | Status (vs Benchmark) |
|---|---|---|---|---|---|
| **SVM (LinearSVC)** | **92.8%** | **92.8%** | **92.8%** | **92.8%** | **PASS** (target >= 92%) |
| Logistic Regression | 92.4% | 92.3% | 92.4% | 92.3% | **PASS** (target >= 89%) |
| Naive Bayes | 92.4% | 92.3% | 92.4% | 0.923 | **PASS** (target >= 84%) |
| Random Forest | 89.2% | 89.1% | 89.2% | 89.1% | **PASS** (target >= 80%) |

*SVM is selected for prototype integration due to its superior generalization performance in high-dimensional TF-IDF sparse space.*
