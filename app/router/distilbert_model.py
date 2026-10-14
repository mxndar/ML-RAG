#!/usr/bin/env python3
"""
DistilBERT Query-Intent Classifier.

Fine-tunes distilbert-base-uncased with a 4-way classification head.
Max sequence length 64 (queries are short). Early-stops on validation macro-F1.

Usage:
  python -m app.router.distilbert_model          # train + evaluate
  python -m app.router.distilbert_model --eval   # evaluate saved model
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

DATA_DIR = Path("data/query_intent")
MODEL_DIR = Path("app/router/saved_models")
LABELS = ["Page", "Section", "Table", "General"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
SEED = 42
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
PATIENCE = 3  # early stopping patience


# ════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════

def load_split(name: str) -> tuple[list[str], list[int]]:
    """Load a CSV split, return (queries, label_ids)."""
    path = DATA_DIR / f"{name}.csv"
    queries, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            queries.append(row["query"])
            labels.append(LABEL2ID[row["label"]])
    return queries, labels


class QueryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════

def train_and_evaluate():
    """Fine-tune DistilBERT, evaluate, save."""
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    print("=" * 60)
    print("DistilBERT Classifier Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    print("\n1. Loading data...")
    train_q, train_y = load_split("train")
    val_q, val_y = load_split("val")
    test_q, test_y = load_split("test")
    print(f"   Train: {len(train_q)}, Val: {len(val_q)}, Test: {len(test_q)}")

    # Tokenize
    print("\n2. Tokenizing...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_enc = tokenizer(train_q, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    val_enc = tokenizer(val_q, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    test_enc = tokenizer(test_q, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

    train_ds = QueryDataset(train_enc, train_y)
    val_ds = QueryDataset(val_enc, val_y)
    test_ds = QueryDataset(test_enc, test_y)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    print("\n3. Loading DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Training loop with early stopping
    print("\n4. Training...")
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_input = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                outputs = model(**batch_input)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch["labels"].numpy())

        val_f1 = f1_score(val_true, val_preds, average="macro")
        val_acc = accuracy_score(val_true, val_preds)

        print(f"   Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Test evaluation
    print("\n5. Test set evaluation...")
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_input = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**batch_input)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(batch["labels"].numpy())

    test_pred_labels = [ID2LABEL[p] for p in test_preds]
    test_true_labels = [ID2LABEL[t] for t in test_true]

    test_acc = accuracy_score(test_true_labels, test_pred_labels)
    test_f1 = f1_score(test_true_labels, test_pred_labels, average="macro")
    test_report = classification_report(test_true_labels, test_pred_labels, target_names=LABELS, digits=4)
    test_cm = confusion_matrix(test_true_labels, test_pred_labels, labels=LABELS)

    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Test macro-F1: {test_f1:.4f}")
    print(f"\n   Classification Report:\n{test_report}")
    print(f"   Confusion Matrix:")
    print(f"   Labels: {LABELS}")
    for i, row in enumerate(test_cm):
        print(f"   {LABELS[i]:>10s}: {row}")

    # Latency benchmark
    print("\n6. CPU inference latency...")
    model.to("cpu")
    model.eval()
    latencies = []
    for q in test_q[:100]:
        t0 = time.perf_counter()
        inputs = tokenizer(q, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        _ = torch.argmax(outputs.logits, dim=-1).item()
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    print(f"   Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

    # Hard test
    hard_test_path = DATA_DIR / "hard_test.csv"
    hard_metrics = {}
    if hard_test_path.exists():
        print("\n7. Hard test set evaluation...")
        hard_q, hard_y = [], []
        with open(hard_test_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hard_q.append(row["query"])
                hard_y.append(row["label"])
        model.to(device)
        hard_enc = tokenizer(hard_q, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        hard_ds = QueryDataset(hard_enc, [LABEL2ID[l] for l in hard_y])
        hard_loader = DataLoader(hard_ds, batch_size=BATCH_SIZE)
        hard_preds = []
        model.eval()
        with torch.no_grad():
            for batch in hard_loader:
                batch_input = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                outputs = model(**batch_input)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                hard_preds.extend(preds)
        hard_pred_labels = [ID2LABEL[p] for p in hard_preds]
        hard_acc = accuracy_score(hard_y, hard_pred_labels)
        hard_f1 = f1_score(hard_y, hard_pred_labels, average="macro")
        print(f"   Hard test accuracy: {hard_acc:.4f}")
        print(f"   Hard test macro-F1: {hard_f1:.4f}")
        hard_metrics = {"accuracy": round(hard_acc, 4), "macro_f1": round(hard_f1, 4)}

    # Save
    print("\n8. Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    model.save_pretrained(MODEL_DIR / "distilbert")
    tokenizer.save_pretrained(MODEL_DIR / "distilbert")

    metrics = {
        "model": "DistilBERT",
        "val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_classification_report": test_report,
        "test_confusion_matrix": test_cm.tolist(),
        "labels": LABELS,
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "epochs_trained": epoch + 1,
        "max_seq_len": MAX_LEN,
    }
    if hard_metrics:
        metrics["hard_test"] = hard_metrics

    with open(MODEL_DIR / "distilbert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved to {MODEL_DIR}/distilbert/")

    print("\n" + "=" * 60)
    print("✅ DistilBERT training complete!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    if "--eval" in sys.argv:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        q = sys.argv[-1] if len(sys.argv) > 2 else "What's on page 5?"
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR / "distilbert")
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR / "distilbert")
        model.eval()
        inputs = tokenizer(q, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        print(f"Query: {q}")
        print(f"Label: {ID2LABEL[pred_id]}, Confidence: {probs[pred_id]:.4f}")
        print(f"All probs: {dict(zip(LABELS, [f'{p:.4f}' for p in probs.tolist()]))}")
    else:
        train_and_evaluate()
