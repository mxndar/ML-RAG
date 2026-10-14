"""
Tests for the ML Query-Intent Router.

Covers:
  1. Dataset validation (schema, class balance, no leakage)
  2. Baseline model loading and prediction
  3. Router threshold branching (confident → tool, uncertain → planner)
  4. Parameter extraction (page numbers, section names)
  5. Config defaults
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure app is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data/query_intent")
MODEL_DIR = Path("app/router/saved_models")


# ════════════════════════════════════════════════════════════════
# 1. Dataset Validation
# ════════════════════════════════════════════════════════════════

class TestDataset:
    """Test the generated dataset quality."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.train = self._load(DATA_DIR / "train.csv")
        self.val = self._load(DATA_DIR / "val.csv")
        self.test = self._load(DATA_DIR / "test.csv")
        self.raw = self._load(DATA_DIR / "raw.csv")

    def _load(self, path: Path) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_raw_has_1600_rows(self):
        assert len(self.raw) == 1600

    def test_class_balance_raw(self):
        counts = Counter(r["label"] for r in self.raw)
        for label in ["Page", "Section", "Table", "General"]:
            assert counts[label] == 400, f"{label}: {counts[label]} != 400"

    def test_required_columns(self):
        expected = {"id", "base_instance_id", "query", "label", "tool",
                    "template_id", "generation_method", "is_hard_negative"}
        for row in self.raw[:5]:
            assert expected.issubset(row.keys())

    def test_no_cross_split_duplicates(self):
        train_qs = {r["query"].lower() for r in self.train}
        val_qs = {r["query"].lower() for r in self.val}
        test_qs = {r["query"].lower() for r in self.test}
        assert len(train_qs & val_qs) == 0, "Train/val overlap"
        assert len(train_qs & test_qs) == 0, "Train/test overlap"
        assert len(val_qs & test_qs) == 0, "Val/test overlap"

    def test_no_group_leakage(self):
        train_gids = {r["base_instance_id"] for r in self.train}
        val_gids = {r["base_instance_id"] for r in self.val}
        test_gids = {r["base_instance_id"] for r in self.test}
        assert len(train_gids & val_gids) == 0
        assert len(train_gids & test_gids) == 0
        assert len(val_gids & test_gids) == 0

    def test_valid_labels(self):
        valid = {"Page", "Section", "Table", "General"}
        for row in self.raw:
            assert row["label"] in valid, f"Invalid label: {row['label']}"

    def test_hard_test_exists(self):
        hard = DATA_DIR / "hard_test.csv"
        assert hard.exists()
        with open(hard) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 100


# ════════════════════════════════════════════════════════════════
# 2. Baseline Model
# ════════════════════════════════════════════════════════════════

class TestBaselineModel:
    """Test baseline model prediction."""

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        if not (MODEL_DIR / "baseline_model.joblib").exists():
            pytest.skip("Baseline model not trained yet")

    def test_predict_page_query(self):
        from app.router.baseline_model import predict_single
        label, tool, conf = predict_single("Show me page 5")
        assert label == "Page"
        assert tool == "PageRetriever"
        assert 0.0 <= conf <= 1.0

    def test_predict_section_query(self):
        from app.router.baseline_model import predict_single
        label, tool, conf = predict_single("Summarize the Methodology section")
        assert label == "Section"
        assert tool == "SectionRetriever"

    def test_predict_table_query(self):
        from app.router.baseline_model import predict_single
        label, tool, conf = predict_single("Extract the revenue table")
        assert label == "Table"
        assert tool == "TableExtractor"

    def test_predict_general_query(self):
        from app.router.baseline_model import predict_single
        label, tool, conf = predict_single("What is the refund policy?")
        assert label == "General"
        assert tool == "BM25Search"

    def test_confidence_range(self):
        from app.router.baseline_model import predict_single
        for q in ["Show page 10", "What is compliance?", "Extract the table on pricing"]:
            _, _, conf = predict_single(q)
            assert 0.0 <= conf <= 1.0


# ════════════════════════════════════════════════════════════════
# 3. Router Threshold Branching
# ════════════════════════════════════════════════════════════════

class TestRouterBranching:
    """Test that route_ml_router correctly branches on confidence."""

    def test_confident_routes_to_tools(self):
        from app.orchestration.graph import route_ml_router
        state = {"router_used": True, "router_confidence": 0.95, "intent_label": "Page"}
        assert route_ml_router(state) == "confident"

    def test_uncertain_routes_to_planner(self):
        from app.orchestration.graph import route_ml_router
        state = {"router_used": False, "router_confidence": 0.4, "intent_label": "Page"}
        assert route_ml_router(state) == "uncertain"

    def test_missing_router_used_routes_to_planner(self):
        from app.orchestration.graph import route_ml_router
        state = {}
        assert route_ml_router(state) == "uncertain"


# ════════════════════════════════════════════════════════════════
# 4. Parameter Extraction
# ════════════════════════════════════════════════════════════════

class TestParameterExtraction:
    """Test extract_tool_params."""

    def test_extract_page_number(self):
        from app.router.query_router import extract_tool_params
        params = extract_tool_params("Show me page 42", "Page")
        assert 42 in params["pages"]

    def test_extract_page_range(self):
        from app.router.query_router import extract_tool_params
        params = extract_tool_params("Show me pages 5 to 8", "Page")
        assert params["pages"] == [5, 6, 7, 8]

    def test_extract_page_word_form(self):
        from app.router.query_router import extract_tool_params
        params = extract_tool_params("Show me page seven", "Page")
        assert 7 in params["pages"]

    def test_extract_section_name(self):
        from app.router.query_router import extract_tool_params
        params = extract_tool_params("Summarize the Methodology section", "Section")
        assert "section" in params
        assert len(params["section"]) > 0

    def test_general_passthrough(self):
        from app.router.query_router import extract_tool_params
        params = extract_tool_params("What is the refund policy?", "General")
        assert params["query"] == "What is the refund policy?"


# ════════════════════════════════════════════════════════════════
# 5. Config Defaults
# ════════════════════════════════════════════════════════════════

class TestConfig:
    """Test router config defaults."""

    def test_router_enabled_default(self):
        from app.config import settings
        assert settings.router_enabled is True

    def test_router_model_type_default(self):
        from app.config import settings
        assert settings.router_model_type == "tfidf_logreg"

    def test_router_confidence_threshold_default(self):
        from app.config import settings
        assert settings.router_confidence_threshold == 0.75

    def test_router_model_path_default(self):
        from app.config import settings
        assert settings.router_model_path == "app/router/saved_models"
