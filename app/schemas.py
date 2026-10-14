"""
Pydantic schemas — data contracts for the entire pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Chunk Metadata ──────────────────────────────────────────

class ChunkMeta(BaseModel):
    """Metadata attached to every document chunk."""
    chunk_id: str
    page_number: int
    section_name: str = ""
    document_name: str = ""
    source: str = ""


class Chunk(BaseModel):
    """A single document chunk with text, metadata, and optional score."""
    text: str
    metadata: ChunkMeta
    score: float = 0.0


# ── Table ───────────────────────────────────────────────────

class ExtractedTable(BaseModel):
    """A table extracted from the PDF."""
    page_number: int
    table_index: int = 0
    document_name: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    markdown: str = ""


# ── Planner ─────────────────────────────────────────────────

class PlannerOutput(BaseModel):
    """Planner's decision: which tools to use and with what parameters."""
    tools: list[str] = Field(
        description="Tool names: bm25, page_retriever, section_retriever, table_extractor"
    )
    params: dict = Field(
        default_factory=dict,
        description="Tool-specific parameters: query, pages, section, etc.",
    )


# ── Evidence ────────────────────────────────────────────────

class Evidence(BaseModel):
    """Merged evidence from all tools, ready for the generator."""
    chunks: list[Chunk] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    total_tokens: int = 0
    confidence: float = 0.0

    def __bool__(self) -> bool:
        """True if evidence contains any retrieved chunks or tables."""
        return bool(self.chunks or self.tables)


# ── Critic ──────────────────────────────────────────────────

class CriticResult(BaseModel):
    """Output of the rule-based critic."""
    is_complete: bool
    issues: list[str] = Field(default_factory=list)


# ── Trace ───────────────────────────────────────────────────

class TraceStep(BaseModel):
    """One step in the execution trace — for observability."""
    node: str
    duration_ms: float = 0.0
    result_summary: str = ""
    cache_hit: bool = False
    details: dict = Field(default_factory=dict)


# ── Cache ───────────────────────────────────────────────────

class CacheEntry(BaseModel):
    """Stored in both exact and semantic caches."""
    answer: str
    evidence: Evidence
    timestamp: float
    query: str = ""


# ── Query Request / Response ────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming query from the user."""
    query: str
    top_k: int = 10


class QueryResponse(BaseModel):
    """Full response returned to the user."""
    answer: str
    evidence: Evidence
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    cache_type: str = "none"  # "exact" | "semantic" | "none"
    llm_calls: int = 0
    # ML Router metadata
    router_used: bool = False
    intent_label: str = ""
    router_confidence: float = 0.0
    trace: list[TraceStep] = Field(default_factory=list)
