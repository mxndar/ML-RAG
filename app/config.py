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
    planner_model: str = "llama3-8b-8192"
    generator_model: str = "llama-3.3-70b-versatile"

    # ── Embedding (semantic cache ONLY) ─────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ── Reranker ────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Cache ───────────────────────────────────────────────
    cache_ttl_hours: int = 24
    semantic_threshold: float = 0.95

    # ── Retrieval ───────────────────────────────────────────
    max_evidence_tokens: int = 300
    bm25_top_k: int = 10
    reranker_top_k: int = 2
    max_critic_retries: int = 2

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
