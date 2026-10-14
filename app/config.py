"""
Centralized configuration — loads from .env with defaults.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for the PDF Agentic RAG system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Cloud LLM (Groq) ───────────────────────────
    groq_api_key: str = ""
    planner_model: str = "qwen/qwen3.8-27b"
    generator_model: str = "qwen/qwen3.8-27b"

    # ── Embedding (semantic cache ONLY) ─────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ── Reranker ────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Cache ───────────────────────────────────────────────
    cache_ttl_hours: int = 24
    semantic_threshold: float = 0.95

    # ── Retrieval ───────────────────────────────────────────
    max_evidence_tokens: int = 600
    bm25_top_k: int = 15
    reranker_top_k: int = 4
    max_critic_retries: int = 2

    # ── ML Router ──────────────────────────────────────────────
    router_enabled: bool = True
    router_model_type: str = "tfidf_logreg"  # "tfidf_logreg" | "distilbert"
    router_model_path: str = "app/router/saved_models"
    router_confidence_threshold: float = 0.75

    # ── Chunking ────────────────────────────────────────────
    chunk_similarity_threshold: float = 0.5
    min_chunk_sentences: int = 2
    max_chunk_sentences: int = 15

    # ── Paths ───────────────────────────────────────────────
    data_dir: Path = Path("data")
    indices_dir: Path = Path("indices")

    def ensure_dirs(self) -> None:
        """Create data and indices directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)


# Singleton
settings = Settings()
settings.ensure_dirs()
