import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import config
from preprocess import preprocess_text
from feature_extraction import get_tfidf_vectorizer, save_vectorizer

# Ensure directories exist
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data():
    """Loads dataset and applies the 7-step preprocessing pipeline."""
    print(f"Loading dataset from: {config.DATASET_CSV}")
    df = pd.read_csv(config.DATASET_CSV)
    
    print("Preprocessing article texts (this may take a moment)...")
    df["processed_text"] = df["text"].apply(preprocess_text)
    
    # Check for empty or null processed text
    df["processed_text"] = df["processed_text"].fillna("")
    
    return df

def run_pipeline():
    # Step 1: Load and preprocess dataset
    df = load_and_preprocess_data()
    X = df["processed_text"]
    y = df["label"]
    
    # Log dataset size
    print(f"Total articles loaded: {len(df)}")
    
    # Step 2: Stratified train/validation/test split
    # Step 2a: Split off test set (15%)
    X_rem, X_test, y_rem, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        stratify=y, 
        random_state=config.RANDOM_STATE
    )
    
    # Step 2b: Split remainder into train (70% of total) and validation (15% of total)
    # Since remainder is 85% of total, val_size on remainder is 15 / 85 = 3 / 17
    X_train, X_val, y_train, y_val = train_test_split(
        X_rem, y_rem, 
        test_size=(config.VAL_SIZE / (config.TRAIN_SIZE + config.VAL_SIZE)), 
        stratify=y_rem, 
        random_state=config.RANDOM_STATE
    )
    
    print(f"Split sizes logged:")
    print(f"  Training set:   N = {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Validation set: N = {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
    # Note: validation split can be used for hyperparameters tuning or early validation checks
    print(f"  Test set:       N = {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
    
    # Step 3: Feature Extraction (TF-IDF)
    # Fit exclusively on the training corpus to prevent data leakage
    print("Fitting TF-IDF Vectorizer on training set...")
    vectorizer = get_tfidf_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Transform validation and test sets (transform only, no fit)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save the fitted vectorizer
    save_vectorizer(vectorizer, config.VECTORIZER_PKL)
    
    # Step 4: Classifier Configurations
    # 4.1 Multinomial Naive Bayes
    nb_clf = MultinomialNB(alpha=1.0)
    
    # 4.2 Linear SVM wrapped in CalibratedClassifierCV for confidence probabilities
    base_svc = LinearSVC(C=1.0, max_iter=2000, random_state=config.RANDOM_STATE)
    svm_clf = CalibratedClassifierCV(estimator=base_svc, cv=5)
    
    # 4.3 Logistic Regression
    lr_clf = LogisticRegression(
        penalty='l2', 
        C=1.0, 
        solver='lbfgs', 
        multi_class='ovr', 
        max_iter=1000, 
        random_state=config.RANDOM_STATE
    )
    
    # 4.4 Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=50, 
        random_state=config.RANDOM_STATE
    )
    
    classifiers = {
        "Naive Bayes": (nb_clf, config.NB_MODEL_PKL, 0.84),
        "SVM (LinearSVC)": (svm_clf, config.SVM_MODEL_PKL, 0.92),
        "Logistic Regression": (lr_clf, config.LR_MODEL_PKL, 0.89),
        "Random Forest": (rf_clf, config.RF_MODEL_PKL, 0.80)
    }
    
    # Step 5: Model Training and Evaluation
    evaluation_results = []
    
    print("\nTraining and evaluating classifiers...")
    for name, (clf, model_path, min_acc) in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train_vec, y_train)
        
        # Predict on validation and test sets
        y_test_pred = clf.predict(X_test_vec)
        
        # Save model
        joblib.dump(clf, model_path)
        print(f"Saved {name} model to: {model_path}")
        
        # Metrics
        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, average='macro')
        rec = recall_score(y_test, y_test_pred, average='macro')
        f1 = f1_score(y_test, y_test_pred, average='macro')
        
        print(f"[{name}] Test Accuracy: {acc*100:.2f}% (Benchmark target: {min_acc*100:.1f}%)")
        print(f"[{name}] Macro F1-Score: {f1:.4f}")
        
        # Verify benchmarks
        assert acc >= min_acc, f"{name} accuracy ({acc:.4f}) is below benchmark target ({min_acc:.4f})"
        
        evaluation_results.append({
            "Classifier": name,
            "Accuracy": acc,
            "Macro Precision": prec,
            "Macro Recall": rec,
            "Macro F1": f1
        })
        
    # Save comparison table
    df_results = pd.DataFrame(evaluation_results)
    df_results.set_index("Classifier", inplace=True)
    df_results.to_csv(config.EVAL_RESULTS_CSV)
    
    print("\nModel Comparison Table:")
    print(df_results)
    print(f"Saved evaluation results table to: {config.EVAL_RESULTS_CSV}")
    
    # Step 6: Generate and save confusion matrix for the best-performing classifier (SVM)
    best_clf_name = "SVM (LinearSVC)"
    best_clf_path = config.SVM_MODEL_PKL
    best_clf = joblib.load(best_clf_path)
    
    y_test_pred_best = best_clf.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_test_pred_best, labels=config.CATEGORIES)
    
    df_cm = pd.DataFrame(cm, index=config.CATEGORIES, columns=config.CATEGORIES)
    df_cm.to_csv(config.CONFUSION_MATRIX_CSV)
    
    print("\nConfusion Matrix for SVM (LinearSVC):")
    print(df_cm)
    print(f"Saved SVM confusion matrix to: {config.CONFUSION_MATRIX_CSV}")
    
    print("\nAll ML training and evaluation steps completed successfully!")

if __name__ == "__main__":
    run_pipeline()
