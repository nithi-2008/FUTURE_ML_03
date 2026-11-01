"""
Model Training, Cross-Validation, and Benchmarking Pipeline
Trains Category and Priority classifiers on customer support ticket text.
"""

import os
import json
import time
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix
)

from src.text_preprocessor import clean_text


def load_and_preprocess_data(data_path: str = "data/customer_support_tickets.csv") -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load ticket data, construct full text, and apply preprocessing."""
    df = pd.read_csv(data_path)
    # Combine subject and description
    raw_texts = df["ticket_subject"].fillna("") + " " + df["ticket_description"].fillna("")
    print(f"Applying NLP text cleaning to {len(raw_texts)} tickets...")
    cleaned_texts = raw_texts.apply(clean_text)
    
    categories = df["ticket_category"]
    priorities = df["ticket_priority"]
    return df, cleaned_texts, categories, priorities


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate classification models with tuned hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            C=2.0, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000),
            cv=3
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "SGD Classifier (Log Loss)": SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=30, random_state=42, n_jobs=-1
        )
    }


def extract_top_keywords(model, vectorizer, class_labels: List[str], top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """Extract top most predictive keywords for each class label."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords_per_class = {}

    # Extract coefficients if available
    coef = None
    if hasattr(model, "coef_"):
        coef = model.coef_
    elif hasattr(model, "calibrated_classifiers_"):
        # For CalibratedClassifierCV with LinearSVC
        try:
            coefs = [clf.estimator.coef_ for clf in model.calibrated_classifiers_]
            coef = np.mean(coefs, axis=0)
        except Exception:
            coef = None
    elif hasattr(model, "feature_log_prob_"):
        coef = model.feature_log_prob_

    if coef is not None:
        for idx, label in enumerate(class_labels):
            if idx < len(coef):
                top_indices = np.argsort(coef[idx])[-top_n:][::-1]
                top_words = feature_names[top_indices]
                scores = coef[idx][top_indices]
                top_keywords_per_class[label] = [
                    {"keyword": word, "weight": round(float(score), 4)}
                    for word, score in zip(top_words, scores)
                ]
    else:
        # Fallback to feature importance if available
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            top_words = feature_names[top_indices]
            scores = importances[top_indices]
            generic_list = [
                {"keyword": word, "weight": round(float(score), 4)}
                for word, score in zip(top_words, scores)
            ]
            for label in class_labels:
                top_keywords_per_class[label] = generic_list

    return top_keywords_per_class


def train_and_benchmark(
    X_train_vec, X_test_vec, y_train, y_test, target_name: str
) -> Tuple[Dict[str, Dict[str, Any]], str, Any]:
    """Train all candidates, compute comprehensive metrics, and select winner."""
    models = get_candidate_models()
    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model = None

    class_names = sorted(list(set(y_train)))

    print(f"\n--- Benchmarking Models for Target: [{target_name}] ---")
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        fit_time = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(X_test_vec)
        inf_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        cm_normalized = confusion_matrix(y_test, y_pred, labels=class_names, normalize="true")

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "train_time_sec": round(float(fit_time), 3),
            "inference_time_sec": round(float(inf_time), 4),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_norm": [[round(float(v), 3) for v in row] for row in cm_normalized],
            "class_names": class_names,
            "classification_report": report
        }

        print(f" > {name:<26} | Acc: {acc:.4f} | F1 (Macro): {f1_macro:.4f} | Train: {fit_time:.2f}s")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_model = clf

    print(f"Winner for {target_name}: {best_model_name} with F1-Macro = {best_f1:.4f}")
    return results, best_model_name, best_model


def run_training_pipeline(
    data_path: str = "data/customer_support_tickets.csv",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """Execute end-to-end model training, evaluation, and serialization."""
    os.makedirs(models_dir, exist_ok=True)

    df, cleaned_texts, categories, priorities = load_and_preprocess_data(data_path)

    # Chronological/Stratified split
    X_train_text, X_test_text, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
        cleaned_texts, categories, priorities,
        test_size=0.20,
        random_state=42,
        stratify=categories
    )

    print(f"\nVectorizing text with TF-IDF (1-2 N-grams, max 5,000 features)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    print(f"TF-IDF Matrix Shape: {X_train_vec.shape}")

    # Benchmark Category Models
    cat_benchmarks, best_cat_name, best_cat_model = train_and_benchmark(
        X_train_vec, X_test_vec, y_cat_train, y_cat_test, target_name="Ticket Category"
    )

    # Benchmark Priority Models
    prio_benchmarks, best_prio_name, best_prio_model = train_and_benchmark(
        X_train_vec, X_test_vec, y_prio_train, y_prio_test, target_name="Ticket Priority"
    )

    # Extract Top Keywords
    cat_classes = sorted(list(set(y_cat_train)))
    prio_classes = sorted(list(set(y_prio_train)))

    cat_top_words = extract_top_keywords(best_cat_model, vectorizer, cat_classes, top_n=10)
    prio_top_words = extract_top_keywords(best_prio_model, vectorizer, prio_classes, top_n=10)

    # Serialization paths
    cat_model_path = os.path.join(models_dir, "category_model.joblib")
    prio_model_path = os.path.join(models_dir, "priority_model.joblib")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_cat_model, cat_model_path)
    joblib.dump(best_prio_model, prio_model_path)
    joblib.dump(vectorizer, vec_path)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_dataset_samples": len(df),
        "train_samples": len(X_train_text),
        "test_samples": len(X_test_text),
        "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
        "category_classes": cat_classes,
        "priority_classes": prio_classes,
        "best_category_model": best_cat_name,
        "best_priority_model": best_prio_name,
        "category_benchmarks": cat_benchmarks,
        "priority_benchmarks": prio_benchmarks,
        "top_keywords_category": cat_top_words,
        "top_keywords_priority": prio_top_words
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[OK] Model training completed successfully!")
    print(f"[OK] Category model saved to: {cat_model_path}")
    print(f"[OK] Priority model saved to: {prio_model_path}")
    print(f"[OK] Vectorizer saved to: {vec_path}")
    print(f"[OK] Metadata saved to: {meta_path}")

    return metadata


if __name__ == "__main__":
    run_training_pipeline()
