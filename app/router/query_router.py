"""
Query Router — ML-based first-stage intent classifier.

Predicts (label, tool, confidence) for a query. If confidence >= threshold,
the router dispatches directly to the mapped tool (skipping the LLM planner).
Otherwise, falls back to the existing LLM planner.

Supports two model backends:
  - tfidf_logreg: TF-IDF + LogisticRegression (fast, ~0.5ms)
  - distilbert: Fine-tuned DistilBERT (slower, ~15-30ms, potentially more accurate)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

LABELS = ["Page", "Section", "Table", "General"]
TOOL_MAP = {
    "Page": "page_retriever",
    "Section": "section_retriever",
    "Table": "table_extractor",
    "General": "bm25",
}

# Module-level cached model
_model = None
_model_type = None


def _get_model_dir() -> Path:
    """Get the saved models directory."""
    return Path(settings.router_model_path) if settings.router_model_path else Path("app/router/saved_models")


def load_model(model_type: str | None = None) -> None:
    """Load the router model into memory."""
    global _model, _model_type

    mtype = model_type or settings.router_model_type
    model_dir = _get_model_dir()

    if mtype == "tfidf_logreg":
        _load_tfidf_model(model_dir)
    elif mtype == "distilbert":
        _load_distilbert_model(model_dir)
    else:
        raise ValueError(f"Unknown router model type: {mtype}")

    _model_type = mtype
    logger.info(f"Router model loaded: {mtype}")


def _load_tfidf_model(model_dir: Path) -> None:
    """Load TF-IDF + sklearn model."""
    global _model
    import joblib

    _model = {
        "classifier": joblib.load(model_dir / "baseline_model.joblib"),
        "word_tfidf": joblib.load(model_dir / "word_tfidf.joblib"),
        "char_tfidf": joblib.load(model_dir / "char_tfidf.joblib"),
    }


def _load_distilbert_model(model_dir: Path) -> None:
    """Load fine-tuned DistilBERT."""
    global _model
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    distilbert_dir = model_dir / "distilbert"
    tokenizer = DistilBertTokenizer.from_pretrained(distilbert_dir)
    model = DistilBertForSequenceClassification.from_pretrained(distilbert_dir)
    model.eval()

    _model = {"tokenizer": tokenizer, "model": model}


def is_loaded() -> bool:
    """Check if the router model is loaded."""
    return _model is not None


def predict(query: str) -> tuple[str, str, float]:
    """
    Predict query intent.

    Args:
        query: User's query string.

    Returns:
        (label, tool_name, confidence)
        - label: "Page", "Section", "Table", or "General"
        - tool_name: the tool key used in execute_tools ("page_retriever", etc.)
        - confidence: float 0-1, max class probability
    """
    if not is_loaded():
        load_model()

    if _model_type == "tfidf_logreg":
        return _predict_tfidf(query)
    elif _model_type == "distilbert":
        return _predict_distilbert(query)
    else:
        raise RuntimeError(f"Unknown model type: {_model_type}")


def _predict_tfidf(query: str) -> tuple[str, str, float]:
    """Predict using TF-IDF + sklearn model."""
    from app.router.baseline_model import build_features

    classifier = _model["classifier"]
    word_tfidf = _model["word_tfidf"]
    char_tfidf = _model["char_tfidf"]

    X = build_features([query], word_tfidf, char_tfidf)
    label = classifier.predict(X)[0]

    # Get confidence
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X)[0]
        confidence = float(max(proba))
    else:
        decision = classifier.decision_function(X)[0]
        exp_d = np.exp(decision - np.max(decision))
        proba = exp_d / exp_d.sum()
        confidence = float(max(proba))

    return label, TOOL_MAP[label], confidence


def _predict_distilbert(query: str) -> tuple[str, str, float]:
    """Predict using fine-tuned DistilBERT."""
    import torch

    tokenizer = _model["tokenizer"]
    model = _model["model"]

    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = torch.argmax(probs).item()
    confidence = float(probs[pred_id])
    label = LABELS[pred_id]

    return label, TOOL_MAP[label], confidence


def extract_tool_params(query: str, label: str) -> dict:
    """
    Extract tool-specific parameters from the query text.

    For the ML router to bypass the planner, we need to parse
    page numbers, section names, and search terms from the query.
    """
    params: dict = {"query": query}

    if label == "Page":
        # Extract page numbers
        pages = _extract_page_numbers(query)
        if pages:
            params["pages"] = pages

    elif label == "Section":
        # Extract section name
        section = _extract_section_name(query)
        if section:
            params["section"] = section

    elif label == "Table":
        # Strip stopwords/verbs to get meaningful content words for table keyword search
        table_stopwords = {
            "table", "tables", "show", "give", "find", "get", "extract",
            "display", "provide", "tell", "list", "fetch", "retrieve",
            "return", "print", "what", "which", "where", "how", "the",
            "a", "an", "is", "are", "was", "were", "be", "in", "of",
            "for", "with", "on", "at", "from", "by", "about", "to",
            "me", "my", "i", "and", "or", "but", "not", "all", "any",
            "this", "that", "these", "those", "it", "its",
            "can", "could", "do", "does", "did", "has", "have", "had",
        }
        words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        content_words = [w for w in words if w not in table_stopwords and len(w) > 1]
        params["query"] = " ".join(content_words) if content_words else query

    elif label == "General":
        # BM25 uses the query directly
        params["query"] = query

    return params


def _extract_page_numbers(query: str) -> list[int]:
    """Extract page numbers from a query string."""
    pages: list[int] = []

    # Match "page X", "page X to Y", "pages X through Y"
    # Digit patterns
    for m in re.finditer(r"page\s+(\d+)", query, re.IGNORECASE):
        pages.append(int(m.group(1)))

    # Range: "pages X to/through/- Y"
    range_match = re.search(
        r"pages?\s+(\d+)\s*(?:to|through|–|-)\s*(\d+)", query, re.IGNORECASE
    )
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        pages = list(range(start, end + 1))

    # Word-form numbers
    word_nums = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50,
    }
    for word, num in word_nums.items():
        if re.search(rf"\bpage\s+{word}\b", query, re.IGNORECASE):
            pages.append(num)

    # Ordinals
    ordinal_map = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        "last": -1,
    }
    for ordinal, num in ordinal_map.items():
        if re.search(rf"\b{ordinal}\s+page\b", query, re.IGNORECASE):
            if num > 0:
                pages.append(num)

    # "1st", "2nd", etc.
    for m in re.finditer(r"(\d+)(?:st|nd|rd|th)\s+page", query, re.IGNORECASE):
        pages.append(int(m.group(1)))

    return sorted(set(pages)) if pages else [1]  # Default to page 1 if no number found


def _extract_section_name(query: str) -> str:
    """Extract section/heading name from a query string."""
    # Try to match "the X section", "X section", "section X"
    patterns = [
        r"(?:the\s+)?(.+?)\s+(?:section|chapter|part)\b",
        r"(?:section|chapter|part)\s+(?:on\s+)?(.+?)(?:\?|$|\.)",
        r"(?:in|under|from)\s+(?:the\s+)?(.+?)(?:\?|$|\.)",
        r"(?:summarize|explain|recap|break\s+down)\s+(?:the\s+)?(.+?)(?:\s+for\s+me|\?|$|\.)",
    ]

    for pattern in patterns:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            name = m.group(1).strip().strip("'\"")
            # Clean up common prefixes/suffixes
            name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.IGNORECASE)
            name = re.sub(r"\s+(?:section|chapter|part|heading)$", "", name, flags=re.IGNORECASE)
            if len(name) > 2:
                return name

    # Fallback: return the whole query as the search term
    return query
