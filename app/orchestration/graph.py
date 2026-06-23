"""
LangGraph Orchestration — the complete agentic RAG workflow.

Flow:
  Query → Exact Cache → Semantic Cache → Planner → Tool Execution
  → Reranker → Evidence Merger → Critic → (retry?) → Generator → Cache Update
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.cache.exact_cache import ExactCache
from app.cache.semantic_cache import SemanticCache
from app.config import settings
from app.retrieval.critic import validate_evidence
from app.retrieval.evidence_merger import merge_evidence
from app.retrieval.planner import run_planner
from app.retrieval.reranker import rerank
from app.schemas import (
    Chunk,
    Evidence,
    ExtractedTable,
    QueryResponse,
    TraceStep,
)
from app.tools import bm25_tool, page_tool, section_tool, table_tool

logger = logging.getLogger(__name__)


# ── State Definition ────────────────────────────────────────

class RAGState(TypedDict, total=False):
    """State passed through the LangGraph workflow."""
    query: str
    # Cache
    cache_hit: bool
    cache_type: str
    cached_answer: str
    cached_evidence: Evidence
    semantic_similarity: float
    semantic_is_fresh: bool
    # Planner
    planned_tools: list[str]
    tool_params: dict
    # Tool results
    bm25_chunks: list[Chunk]
    page_chunks: list[Chunk]
    section_chunks: list[Chunk]
    tables: list[ExtractedTable]
    # Evidence
    evidence: Evidence
    # Critic
    critic_issues: list[str]
    retry_count: int
    # Generator
    answer: str
    citations: list[str]
    confidence: float
    # Observability
    llm_calls: int
    trace: list[TraceStep]


# ── Shared instances ────────────────────────────────────────

_exact_cache: ExactCache | None = None
_semantic_cache: SemanticCache | None = None


def get_caches() -> tuple[ExactCache, SemanticCache]:
    """Get or create the shared cache instances."""
    global _exact_cache, _semantic_cache
    if _exact_cache is None:
        _exact_cache = ExactCache()
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
        _semantic_cache.load()  # Load from disk if available
    return _exact_cache, _semantic_cache


# ── Node Functions ──────────────────────────────────────────

async def check_exact_cache(state: RAGState) -> dict[str, Any]:
    """Check the exact (SHA256) cache."""
    start = time.time()
    exact_cache, _ = get_caches()

    entry = await exact_cache.get(state["query"])

    trace_step = TraceStep(
        node="exact_cache",
        duration_ms=(time.time() - start) * 1000,
    )

    if entry is not None:
        trace_step.cache_hit = True
        trace_step.result_summary = "Exact cache HIT"
        return {
            "cache_hit": True,
            "cache_type": "exact",
            "cached_answer": entry.answer,
            "cached_evidence": entry.evidence,
            "confidence": 1.0,
            "trace": state.get("trace", []) + [trace_step],
        }

    trace_step.result_summary = "Exact cache MISS"
    return {
        "cache_hit": False,
        "cache_type": "none",
        "trace": state.get("trace", []) + [trace_step],
    }


async def check_semantic_cache(state: RAGState) -> dict[str, Any]:
    """Check the semantic (FAISS) cache."""
    start = time.time()
    _, semantic_cache = get_caches()

    entry, similarity, is_fresh = semantic_cache.search(state["query"])

    trace_step = TraceStep(
        node="semantic_cache",
        duration_ms=(time.time() - start) * 1000,
        details={"similarity": round(similarity, 4)},
    )

    if entry is not None:
        trace_step.cache_hit = True
        trace_step.result_summary = f"Semantic cache HIT (sim={similarity:.4f}, fresh={is_fresh})"
        result: dict[str, Any] = {
            "cache_hit": True,
            "cache_type": "semantic",
            "cached_evidence": entry.evidence,
            "semantic_similarity": similarity,
            "semantic_is_fresh": is_fresh,
            "trace": state.get("trace", []) + [trace_step],
        }
        if is_fresh:
            result["cached_answer"] = entry.answer
            result["confidence"] = 1.0
        return result

    trace_step.result_summary = f"Semantic cache MISS (sim={similarity:.4f})"
    return {
        "cache_hit": False,
        "semantic_similarity": similarity,
        "trace": state.get("trace", []) + [trace_step],
    }


async def run_planner_node(state: RAGState) -> dict[str, Any]:
    """Run the planner LLM to decide tools."""
    start = time.time()

    retry_issues = state.get("critic_issues")
    prev_tools = state.get("planned_tools")

    planner_output, metrics = await run_planner(
        query=state["query"],
        retry_issues=retry_issues if retry_issues else None,
        prev_tools=prev_tools if retry_issues else None,
    )

    trace_step = TraceStep(
        node="planner",
        duration_ms=metrics["duration_ms"],
        result_summary=f"Tools: {planner_output.tools}",
        details={
            "tools": planner_output.tools,
            "params": planner_output.params,
            "tokens": metrics["tokens"],
        },
    )

    return {
        "planned_tools": planner_output.tools,
        "tool_params": planner_output.params,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "trace": state.get("trace", []) + [trace_step],
    }


async def execute_tools(state: RAGState) -> dict[str, Any]:
    """Execute the tools selected by the planner — in parallel where possible."""
    start = time.time()
    tools = state.get("planned_tools", ["bm25"])
    params = state.get("tool_params", {})
    query = params.get("query", state["query"])

    # Prepare coroutines
    tasks: dict[str, Any] = {}

    if "bm25" in tools:
        tasks["bm25"] = bm25_tool.bm25_search(query)

    if "page_retriever" in tools and "pages" in params:
        tasks["page"] = page_tool.page_retrieve(params["pages"])

    if "section_retriever" in tools and "section" in params:
        tasks["section"] = section_tool.section_retrieve(params["section"])

    if "table_extractor" in tools:
        table_pages = params.get("pages")
        table_keyword = params.get("query")
        tasks["table"] = table_tool.table_extract(
            pages=table_pages, keyword=table_keyword,
        )

    # Execute in parallel
    results: dict[str, Any] = {}
    if tasks:
        task_keys = list(tasks.keys())
        task_coros = list(tasks.values())
        completed = await asyncio.gather(*task_coros, return_exceptions=True)

        for key, result in zip(task_keys, completed):
            if isinstance(result, Exception):
                logger.error(f"Tool '{key}' failed: {result}")
                results[key] = []
            else:
                results[key] = result

    trace_step = TraceStep(
        node="tool_execution",
        duration_ms=(time.time() - start) * 1000,
        result_summary=(
            f"BM25: {len(results.get('bm25', []))} | "
            f"Page: {len(results.get('page', []))} | "
            f"Section: {len(results.get('section', []))} | "
            f"Table: {len(results.get('table', []))}"
        ),
        details={"tools_executed": list(results.keys())},
    )

    return {
        "bm25_chunks": results.get("bm25", []),
        "page_chunks": results.get("page", []),
        "section_chunks": results.get("section", []),
        "tables": results.get("table", []),
        "trace": state.get("trace", []) + [trace_step],
    }


async def rerank_node(state: RAGState) -> dict[str, Any]:
    """Rerank BM25 results using cross-encoder."""
    start = time.time()
    bm25_chunks = state.get("bm25_chunks", [])

    if bm25_chunks:
        reranked = await rerank(state["query"], bm25_chunks)
    else:
        reranked = []

    trace_step = TraceStep(
        node="reranker",
        duration_ms=(time.time() - start) * 1000,
        result_summary=f"Reranked to top {len(reranked)} chunks",
        details={
            "input_count": len(bm25_chunks),
            "output_count": len(reranked),
            "scores": [round(c.score, 3) for c in reranked],
        },
    )

    return {
        "bm25_chunks": reranked,
        "trace": state.get("trace", []) + [trace_step],
    }


async def merge_evidence_node(state: RAGState) -> dict[str, Any]:
    """Merge all tool outputs into unified evidence."""
    start = time.time()

    evidence = merge_evidence(
        bm25_chunks=state.get("bm25_chunks"),
        page_chunks=state.get("page_chunks"),
        section_chunks=state.get("section_chunks"),
        tables=state.get("tables"),
    )

    trace_step = TraceStep(
        node="evidence_merger",
        duration_ms=(time.time() - start) * 1000,
        result_summary=(
            f"{len(evidence.chunks)} chunks + {len(evidence.tables)} tables "
            f"= {evidence.total_tokens} tokens"
        ),
    )

    return {
        "evidence": evidence,
        "trace": state.get("trace", []) + [trace_step],
    }


async def critic_node(state: RAGState) -> dict[str, Any]:
    """Rule-based evidence validation."""
    start = time.time()
    evidence = state.get("evidence", Evidence())

    result = validate_evidence(state["query"], evidence)

    trace_step = TraceStep(
        node="critic",
        duration_ms=(time.time() - start) * 1000,
        result_summary="COMPLETE ✓" if result.is_complete else f"INCOMPLETE: {result.issues}",
        details={"issues": result.issues, "is_complete": result.is_complete},
    )

    return {
        "critic_issues": result.issues if not result.is_complete else [],
        "retry_count": state.get("retry_count", 0) + (0 if result.is_complete else 1),
        "trace": state.get("trace", []) + [trace_step],
    }


async def generate_node(state: RAGState) -> dict[str, Any]:
    """Generate the final answer using the LLM."""
    start = time.time()
    from app.generation.generator import generate_answer

    # Use cached evidence if available (stale semantic cache hit)
    evidence = state.get("evidence") or state.get("cached_evidence", Evidence())

    answer, metrics = await generate_answer(state["query"], evidence)

    trace_step = TraceStep(
        node="generator",
        duration_ms=metrics["duration_ms"],
        result_summary=f"{metrics['tokens']} tokens, confidence={metrics['confidence']}",
        details=metrics,
    )

    return {
        "answer": answer,
        "citations": metrics.get("citations", []),
        "confidence": metrics["confidence"],
        "evidence": evidence,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "trace": state.get("trace", []) + [trace_step],
    }


async def cache_update_node(state: RAGState) -> dict[str, Any]:
    """Store the result in both caches."""
    start = time.time()
    exact_cache, semantic_cache = get_caches()

    answer = state.get("answer", "")
    evidence = state.get("evidence", Evidence())
    query = state["query"]

    # Update exact cache
    await exact_cache.set(query, answer, evidence)

    # Update semantic cache
    semantic_cache.add(query, answer, evidence)
    semantic_cache.save()

    trace_step = TraceStep(
        node="cache_update",
        duration_ms=(time.time() - start) * 1000,
        result_summary="Updated exact + semantic cache",
    )

    return {
        "trace": state.get("trace", []) + [trace_step],
    }


# ── Routing Functions ───────────────────────────────────────

def route_exact_cache(state: RAGState) -> str:
    """Route after exact cache check."""
    if state.get("cache_hit") and state.get("cache_type") == "exact":
        return "hit"
    return "miss"


def route_semantic_cache(state: RAGState) -> str:
    """Route after semantic cache check."""
    if state.get("cache_hit") and state.get("cache_type") == "semantic":
        if state.get("semantic_is_fresh"):
            return "hit_fresh"
        return "hit_stale"
    return "miss"


def route_critic(state: RAGState) -> str:
    """Route after critic: complete → generate, incomplete → retry planner."""
    issues = state.get("critic_issues", [])
    retry_count = state.get("retry_count", 0)

    if not issues:
        return "complete"

    if retry_count >= settings.max_critic_retries:
        logger.info(f"Max retries ({settings.max_critic_retries}) reached, proceeding with best evidence")
        return "complete"

    return "retry"


# ── Build the Graph ─────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """
    Build the complete LangGraph workflow.

    Returns a compiled StateGraph ready for invocation.
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("check_exact_cache", check_exact_cache)
    graph.add_node("check_semantic_cache", check_semantic_cache)
    graph.add_node("run_planner", run_planner_node)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("rerank", rerank_node)
    graph.add_node("merge_evidence", merge_evidence_node)
    graph.add_node("critic", critic_node)
    graph.add_node("generate", generate_node)
    graph.add_node("cache_update", cache_update_node)

    # ── Edges ──

    # Start → Exact Cache
    graph.add_edge(START, "check_exact_cache")

    # Exact Cache → hit: END, miss: Semantic Cache
    graph.add_conditional_edges(
        "check_exact_cache",
        route_exact_cache,
        {"hit": END, "miss": "check_semantic_cache"},
    )

    # Semantic Cache → hit_fresh: END, hit_stale: Generate, miss: Planner
    graph.add_conditional_edges(
        "check_semantic_cache",
        route_semantic_cache,
        {"hit_fresh": END, "hit_stale": "generate", "miss": "run_planner"},
    )

    # Planner → Tool Execution
    graph.add_edge("run_planner", "execute_tools")

    # Tool Execution → Rerank
    graph.add_edge("execute_tools", "rerank")

    # Rerank → Evidence Merger
    graph.add_edge("rerank", "merge_evidence")

    # Evidence Merger → Critic
    graph.add_edge("merge_evidence", "critic")

    # Critic → complete: Generate, retry: Planner (loop)
    graph.add_conditional_edges(
        "critic",
        route_critic,
        {"complete": "generate", "retry": "run_planner"},
    )

    # Generate → Cache Update
    graph.add_edge("generate", "cache_update")

    # Cache Update → END
    graph.add_edge("cache_update", END)

    return graph.compile()


# ── Public API ──────────────────────────────────────────────

# Compiled graph singleton
_graph = None


def get_graph():
    """Get or build the compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph


async def run_query(query: str) -> QueryResponse:
    """
    Run a query through the full agentic RAG pipeline.

    Returns a QueryResponse with answer, evidence, citations, trace, etc.
    """
    graph = get_graph()

    initial_state: RAGState = {
        "query": query,
        "cache_hit": False,
        "cache_type": "none",
        "cached_answer": "",
        "cached_evidence": Evidence(),
        "semantic_similarity": 0.0,
        "semantic_is_fresh": False,
        "planned_tools": [],
        "tool_params": {},
        "bm25_chunks": [],
        "page_chunks": [],
        "section_chunks": [],
        "tables": [],
        "evidence": Evidence(),
        "critic_issues": [],
        "retry_count": 0,
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "llm_calls": 0,
        "trace": [],
    }

    # Run the graph
    final_state = await graph.ainvoke(initial_state)

    # Build response
    answer = final_state.get("cached_answer") or final_state.get("answer", "")
    evidence = final_state.get("evidence") or final_state.get("cached_evidence", Evidence())

    return QueryResponse(
        answer=answer,
        evidence=evidence,
        citations=final_state.get("citations", []),
        confidence=final_state.get("confidence", 0.0),
        cache_type=final_state.get("cache_type", "none"),
        llm_calls=final_state.get("llm_calls", 0),
        trace=final_state.get("trace", []),
    )
