"""
Semantic Chunker — splits document text by sentence similarity,
paragraph boundaries, and headings. NOT fixed-size chunking.

Also builds the auxiliary indices:
  - section_index: {section_name → [chunk_ids]}
  - page_index:    {page_number  → [chunk_ids]}
  - BM25 index:    BM25Okapi over all chunk texts
"""

from __future__ import annotations

import logging
import pickle
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.schemas import Chunk, ChunkMeta

logger = logging.getLogger(__name__)

# ── Heading detection patterns ──────────────────────────────

HEADING_PATTERNS = [
    # "1. Title" or "1.2.3 Title"
    re.compile(r"^(\d+\.)+\s*.+$"),
    # "Chapter 5" or "CHAPTER 5"
    re.compile(r"^[Cc][Hh][Aa][Pp][Tt][Ee][Rr]\s+\d+", re.IGNORECASE),
    # "Section 3.2"
    re.compile(r"^[Ss]ection\s+[\d.]+", re.IGNORECASE),
    # "APPENDIX A"
    re.compile(r"^[Aa]ppendix\s+[A-Z\d]", re.IGNORECASE),
    # ALL CAPS lines (likely headings), at least 3 chars, at most 100
    re.compile(r"^[A-Z][A-Z\s\-:]{2,99}$"),
]


def is_heading(line: str) -> bool:
    """Check if a line looks like a section heading."""
    line = line.strip()
    if not line or len(line) > 150:
        return False
    return any(p.match(line) for p in HEADING_PATTERNS)


# ── Sentence splitting ──────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


# ── Semantic chunker ────────────────────────────────────────

@dataclass
class ChunkingResult:
    """Everything produced by the chunker."""
    chunks: list[Chunk] = field(default_factory=list)
    section_index: dict[str, list[str]] = field(default_factory=dict)
    page_index: dict[int, list[str]] = field(default_factory=dict)
    bm25_index: BM25Okapi | None = None
    bm25_corpus: list[str] = field(default_factory=list)
    bm25_chunk_ids: list[str] = field(default_factory=list)


_cached_embedding_model = None


def get_embedding_model():
    """Get or initialize the cached singleton embedding model."""
    global _cached_embedding_model
    if _cached_embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        try:
            _cached_embedding_model = SentenceTransformer(settings.embedding_model, local_files_only=True)
        except Exception:
            _cached_embedding_model = SentenceTransformer(settings.embedding_model)
    return _cached_embedding_model


def _compute_sentence_embeddings(sentences: list[str], model) -> np.ndarray:
    """Compute embeddings for a list of sentences."""
    if not sentences:
        return np.array([])
    embeddings = model.encode(sentences, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


def _find_split_points(
    similarities: list[float],
    threshold: float,
) -> list[int]:
    """
    Find indices where consecutive sentence similarity drops below threshold.
    These are paragraph/topic boundaries.
    """
    split_points = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_points.append(i + 1)  # Split AFTER sentence i
    return split_points


def chunk_document(
    pages_text: list[str],
    document_name: str,
    embedding_model=None,
) -> ChunkingResult:
    """
    Semantically chunk a document.

    Process:
      1. Concatenate all pages, tracking page boundaries
      2. Split into sentences
      3. Detect headings → force split at headings
      4. Compute sentence embeddings
      5. Find low-similarity boundaries → split there
      6. Merge tiny chunks, split oversized chunks
      7. Build section_index, page_index, BM25 index

    Args:
        pages_text: Cleaned text per page (1-indexed by list position).
        document_name: Name of the source document.
        embedding_model: Preloaded SentenceTransformer (optional, loaded if None).
    """
    # ── Load embedding model if needed ──
    if embedding_model is None:
        embedding_model = get_embedding_model()

    # ── Step 1: Build sentence list with page tracking ──
    @dataclass
    class SentenceInfo:
        text: str
        page_number: int
        is_heading: bool = False
        section_name: str = ""

    all_sentences: list[SentenceInfo] = []
    current_section = ""

    for page_idx, page_text in enumerate(pages_text):
        page_num = page_idx + 1
        if not page_text.strip():
            continue

        lines = page_text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if is_heading(line):
                current_section = line
                all_sentences.append(
                    SentenceInfo(
                        text=line, page_number=page_num,
                        is_heading=True, section_name=line,
                    )
                )
            else:
                sentences = split_sentences(line)
                for sent in sentences:
                    all_sentences.append(
                        SentenceInfo(
                            text=sent, page_number=page_num,
                            is_heading=False, section_name=current_section,
                        )
                    )

    if not all_sentences:
        logger.warning("No sentences extracted from document")
        return ChunkingResult()

    logger.info(f"Extracted {len(all_sentences)} sentences from {len(pages_text)} pages")

    # ── Step 2: Compute embeddings ──
    sentence_texts = [s.text for s in all_sentences]
    embeddings = _compute_sentence_embeddings(sentence_texts, embedding_model)

    # ── Step 3: Compute consecutive similarities ──
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        similarities.append(sim)

    # ── Step 4: Find split points ──
    # Natural similarity drops
    sim_splits = set(_find_split_points(similarities, settings.chunk_similarity_threshold))

    # Force splits at headings
    heading_splits = {i for i, s in enumerate(all_sentences) if s.is_heading and i > 0}

    all_splits = sorted(sim_splits | heading_splits)

    # ── Step 5: Create chunks from split points ──
    chunk_boundaries = [0] + all_splits + [len(all_sentences)]
    raw_chunks: list[list[SentenceInfo]] = []

    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        group = all_sentences[start:end]
        if group:
            raw_chunks.append(group)

    # ── Step 6: Merge tiny chunks, split oversized ──
    final_chunks: list[list[SentenceInfo]] = []
    buffer: list[SentenceInfo] = []

    for group in raw_chunks:
        if len(buffer) + len(group) <= settings.max_chunk_sentences:
            buffer.extend(group)
        else:
            if buffer:
                final_chunks.append(buffer)
            if len(group) > settings.max_chunk_sentences:
                # Split oversized group
                for i in range(0, len(group), settings.max_chunk_sentences):
                    sub = group[i : i + settings.max_chunk_sentences]
                    final_chunks.append(sub)
            else:
                buffer = group
                continue
            buffer = []

    if len(buffer) >= settings.min_chunk_sentences:
        final_chunks.append(buffer)
    elif buffer and final_chunks:
        final_chunks[-1].extend(buffer)
    elif buffer:
        final_chunks.append(buffer)

    # ── Step 7: Build Chunk objects with metadata ──
    result = ChunkingResult()

    for group in final_chunks:
        chunk_id = str(uuid.uuid4())[:12]
        text = " ".join(s.text for s in group)
        page_numbers = sorted(set(s.page_number for s in group))
        section_names = [s.section_name for s in group if s.section_name]
        section_name = section_names[0] if section_names else ""

        chunk = Chunk(
            text=text,
            metadata=ChunkMeta(
                chunk_id=chunk_id,
                page_number=page_numbers[0],
                section_name=section_name,
                document_name=document_name,
                source=f"{document_name}, page {page_numbers[0]}"
                + (f"-{page_numbers[-1]}" if len(page_numbers) > 1 else ""),
            ),
        )
        result.chunks.append(chunk)

        # Section index
        if section_name:
            result.section_index.setdefault(section_name, []).append(chunk_id)

        # Page index
        for pn in page_numbers:
            result.page_index.setdefault(pn, []).append(chunk_id)

    # ── Step 8: Build BM25 index ──
    import string
    result.bm25_corpus = [c.text for c in result.chunks]
    result.bm25_chunk_ids = [c.metadata.chunk_id for c in result.chunks]
    
    tokenized_corpus = []
    for doc in result.bm25_corpus:
        words = doc.lower().split()
        tokenized_corpus.append([w.strip(string.punctuation) for w in words if w.strip(string.punctuation)])
        
    result.bm25_index = BM25Okapi(tokenized_corpus)

    logger.info(
        f"Chunking complete: {len(result.chunks)} chunks, "
        f"{len(result.section_index)} sections, "
        f"{len(result.page_index)} pages indexed"
    )

    return result


def save_chunking_result(result: ChunkingResult, indices_dir: Path | None = None) -> None:
    """Persist the chunking result to disk."""
    out_dir = indices_dir or settings.indices_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "chunks.pkl", "wb") as f:
        pickle.dump(result.chunks, f)

    with open(out_dir / "section_index.pkl", "wb") as f:
        pickle.dump(result.section_index, f)

    with open(out_dir / "page_index.pkl", "wb") as f:
        pickle.dump(result.page_index, f)

    with open(out_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(result.bm25_index, f)

    with open(out_dir / "bm25_corpus.pkl", "wb") as f:
        pickle.dump(result.bm25_corpus, f)

    with open(out_dir / "bm25_chunk_ids.pkl", "wb") as f:
        pickle.dump(result.bm25_chunk_ids, f)

    logger.info(f"Saved indices to {out_dir}")


def load_chunking_result(indices_dir: Path | None = None) -> ChunkingResult:
    """Load a previously saved chunking result from disk."""
    in_dir = indices_dir or settings.indices_dir

    with open(in_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(in_dir / "section_index.pkl", "rb") as f:
        section_index = pickle.load(f)

    with open(in_dir / "page_index.pkl", "rb") as f:
        page_index = pickle.load(f)

    with open(in_dir / "bm25_index.pkl", "rb") as f:
        bm25_index = pickle.load(f)

    with open(in_dir / "bm25_corpus.pkl", "rb") as f:
        bm25_corpus = pickle.load(f)

    with open(in_dir / "bm25_chunk_ids.pkl", "rb") as f:
        bm25_chunk_ids = pickle.load(f)

    return ChunkingResult(
        chunks=chunks,
        section_index=section_index,
        page_index=page_index,
        bm25_index=bm25_index,
        bm25_corpus=bm25_corpus,
        bm25_chunk_ids=bm25_chunk_ids,
    )
