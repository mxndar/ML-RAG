#!/usr/bin/env python3
"""
Baseline Query-Intent Classifier: TF-IDF + LogReg / SVM.

Features:
  - TF-IDF word n-grams (1-2) + character n-grams (2-4)
  - Hand-crafted features: keyword flags, digit presence, query length
  - Trains both LogReg and LinearSVM, picks better on validation macro-F1

Usage:
  python -m app.router.baseline_model          # train + evaluate
  python -m app.router.baseline_model --eval   # evaluate only (load saved model)
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix

DATA_DIR = Path("data/query_intent")
MODEL_DIR = Path("app/router/saved_models")
LABELS = ["Page", "Section", "Table", "General"]
SEED = 42


# ════════════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════════════

def load_split(name: str) -> tuple[list[str], list[str]]:
    """Load a CSV split, return (queries, labels)."""
    path = DATA_DIR / f"{name}.csv"
    queries, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            queries.append(row["query"])
            labels.append(row["label"])
    return queries, labels


# ════════════════════════════════════════════════════════════════
# Hand-crafted Features
# ════════════════════════════════════════════════════════════════

_PAGE_RE = re.compile(
    r"\bpage\s+\d+|\bp\.\s*\d+|\bpages?\s+\d+|\bpage\s+(one|two|three|four|five|"
    r"six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|"
    r"thirty|forty|fifty|hundred)\b",
    re.IGNORECASE,
)
_SECTION_KEYWORDS = re.compile(
    r"\bsection\b|\bchapter\b|\bpart\b|\bappendix\b|\bheading\b|\bsummar(y|ize)\b|"
    r"\bintroduction\b|\bconclusion\b|\bmethodology\b|\bmethod(s)?\b|\bresult(s)?\b|"
    r"\bdiscussion\b|\bappendix\b|\breference(s)?\b|\bfaq\b|\boverview\b|\bglossary\b|"
    r"\bdisclaimer\b|\bwarranty\b|\beligibility\b",
    re.IGNORECASE,
)
_TABLE_KEYWORDS = re.compile(
    r"\btable\b|\btabul(ar|ated)\b|\bextract\b|\bdata\s+table\b|\bin\s+table\s+form\b|"
    r"\btable\s+form(at)?\b",
    re.IGNORECASE,
)
_HAS_DIGITS = re.compile(r"\d")


def extract_handcrafted(queries: list[str]) -> np.ndarray:
    """Extract hand-crafted features for a list of queries."""
    features = []
    for q in queries:
        ql = q.lower()
        feats = [
            1.0 if _PAGE_RE.search(ql) else 0.0,            # has_page_reference
            1.0 if _SECTION_KEYWORDS.search(ql) else 0.0,   # has_section_keyword
            1.0 if _TABLE_KEYWORDS.search(ql) else 0.0,     # has_table_keyword
            1.0 if _HAS_DIGITS.search(q) else 0.0,          # has_digits
            len(q.split()) / 20.0,                           # normalized_word_count
            1.0 if q.endswith("?") else 0.0,                 # is_question
            1.0 if "please" in ql else 0.0,                  # has_please
        ]
        features.append(feats)
    return np.array(features)


# ════════════════════════════════════════════════════════════════
# Model Building
# ════════════════════════════════════════════════════════════════

def build_features(queries: list[str], word_tfidf=None, char_tfidf=None, fit=False):
    """Build combined TF-IDF + handcrafted feature matrix."""
    if fit:
        word_feats = word_tfidf.fit_transform(queries)
        char_feats = char_tfidf.fit_transform(queries)
    else:
        word_feats = word_tfidf.transform(queries)
        char_feats = char_tfidf.transform(queries)

    hand_feats = csr_matrix(extract_handcrafted(queries))
    return hstack([word_feats, char_feats, hand_feats])


def train_and_evaluate():
    """Train LogReg + SVM, evaluate on val, pick the best, evaluate on test."""
    print("=" * 60)
    print("Baseline Model Training: TF-IDF + LogReg / SVM")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    train_q, train_y = load_split("train")
    val_q, val_y = load_split("val")
    test_q, test_y = load_split("test")
    print(f"   Train: {len(train_q)}, Val: {len(val_q)}, Test: {len(test_q)}")

    # Build TF-IDF vectorizers
    print("\n2. Building features...")
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=5000,
        sublinear_tf=True, min_df=2,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=5000,
        sublinear_tf=True, min_df=2,
    )

    X_train = build_features(train_q, word_tfidf, char_tfidf, fit=True)
    X_val = build_features(val_q, word_tfidf, char_tfidf)
    X_test = build_features(test_q, word_tfidf, char_tfidf)
    print(f"   Feature dim: {X_train.shape[1]}")

    # Train LogReg
    print("\n3. Training LogisticRegression...")
    lr = LogisticRegression(
        max_iter=1000, C=1.0, random_state=SEED,
        multi_class="multinomial", solver="lbfgs",
    )
    lr.fit(X_train, train_y)
    lr_val_pred = lr.predict(X_val)
    lr_val_f1 = f1_score(val_y, lr_val_pred, average="macro")
    lr_val_acc = accuracy_score(val_y, lr_val_pred)
    print(f"   LogReg Val: accuracy={lr_val_acc:.4f}, macro-F1={lr_val_f1:.4f}")

    # Train LinearSVC
    print("\n4. Training LinearSVC...")
    svm = LinearSVC(
        C=1.0, max_iter=2000, random_state=SEED,
        dual="auto",
    )
    svm.fit(X_train, train_y)
    svm_val_pred = svm.predict(X_val)
    svm_val_f1 = f1_score(val_y, svm_val_pred, average="macro")
    svm_val_acc = accuracy_score(val_y, svm_val_pred)
    print(f"   SVM Val:    accuracy={svm_val_acc:.4f}, macro-F1={svm_val_f1:.4f}")

    # Pick best model
    if lr_val_f1 >= svm_val_f1:
        best_model = lr
        best_name = "LogisticRegression"
        best_val_f1 = lr_val_f1
    else:
        best_model = svm
        best_name = "LinearSVC"
        best_val_f1 = svm_val_f1

    print(f"\n5. Best model: {best_name} (val macro-F1={best_val_f1:.4f})")

    # Evaluate on test
    print("\n6. Test set evaluation...")
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred, average="macro")
    test_report = classification_report(test_y, test_pred, target_names=LABELS, digits=4)
    test_cm = confusion_matrix(test_y, test_pred, labels=LABELS)

    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Test macro-F1: {test_f1:.4f}")
    print(f"\n   Classification Report:\n{test_report}")
    print(f"   Confusion Matrix (rows=true, cols=pred):")
    print(f"   Labels: {LABELS}")
    for i, row in enumerate(test_cm):
        print(f"   {LABELS[i]:>10s}: {row}")

    # Latency benchmark
    print("\n7. CPU inference latency...")
    latencies = []
    for q in test_q[:100]:
        t0 = time.perf_counter()
        x = build_features([q], word_tfidf, char_tfidf)
        best_model.predict(x)
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    print(f"   Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

    # Also evaluate on hard_test.csv if available
    hard_test_path = DATA_DIR / "hard_test.csv"
    hard_metrics = {}
    if hard_test_path.exists():
        print("\n8. Hard test set evaluation...")
        hard_q, hard_y = [], []
        with open(hard_test_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hard_q.append(row["query"])
                hard_y.append(row["label"])
        X_hard = build_features(hard_q, word_tfidf, char_tfidf)
        hard_pred = best_model.predict(X_hard)
        hard_acc = accuracy_score(hard_y, hard_pred)
        hard_f1 = f1_score(hard_y, hard_pred, average="macro")
        print(f"   Hard test accuracy: {hard_acc:.4f}")
        print(f"   Hard test macro-F1: {hard_f1:.4f}")
        hard_metrics = {"accuracy": round(hard_acc, 4), "macro_f1": round(hard_f1, 4)}

    # Save model + vectorizers
    print("\n9. Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "baseline_model.joblib")
    joblib.dump(word_tfidf, MODEL_DIR / "word_tfidf.joblib")
    joblib.dump(char_tfidf, MODEL_DIR / "char_tfidf.joblib")
    print(f"   Saved to {MODEL_DIR}/")

    # Save metrics
    metrics = {
        "model": best_name,
        "val_accuracy": round(lr_val_acc if best_name == "LogisticRegression" else svm_val_acc, 4),
        "val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_classification_report": test_report,
        "test_confusion_matrix": test_cm.tolist(),
        "labels": LABELS,
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "feature_dim": X_train.shape[1],
        "comparison": {
            "LogisticRegression": {"val_accuracy": round(lr_val_acc, 4), "val_macro_f1": round(lr_val_f1, 4)},
            "LinearSVC": {"val_accuracy": round(svm_val_acc, 4), "val_macro_f1": round(svm_val_f1, 4)},
        },
    }
    if hard_metrics:
        metrics["hard_test"] = hard_metrics

    with open(MODEL_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved to {MODEL_DIR}/baseline_metrics.json")

    print("\n" + "=" * 60)
    print("✅ Baseline training complete!")
    print("=" * 60)

    return metrics


def predict_single(query: str) -> tuple[str, str, float]:
    """
    Load saved model and predict a single query.
    Returns: (label, tool, confidence)
    """
    model = joblib.load(MODEL_DIR / "baseline_model.joblib")
    word_tfidf = joblib.load(MODEL_DIR / "word_tfidf.joblib")
    char_tfidf = joblib.load(MODEL_DIR / "char_tfidf.joblib")

    X = build_features([query], word_tfidf, char_tfidf)

    label = model.predict(X)[0]

    # Get confidence (probability for LogReg, decision function distance for SVM)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(max(proba))
    else:
        # SVM: use decision function and softmax-ish approximation
        decision = model.decision_function(X)[0]
        exp_d = np.exp(decision - np.max(decision))
        proba = exp_d / exp_d.sum()
        confidence = float(max(proba))

    tool_map = {
        "Page": "PageRetriever",
        "Section": "SectionRetriever",
        "Table": "TableExtractor",
        "General": "BM25Search",
    }
    return label, tool_map[label], confidence


if __name__ == "__main__":
    if "--eval" in sys.argv:
        # Quick eval of saved model
        q = sys.argv[-1] if len(sys.argv) > 2 else "What's on page 5?"
        label, tool, conf = predict_single(q)
        print(f"Query: {q}")
        print(f"Label: {label}, Tool: {tool}, Confidence: {conf:.4f}")
    else:
        train_and_evaluate()
