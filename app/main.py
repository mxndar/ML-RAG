"""
Gradio UI — Main entry point for the PDF Agentic RAG system.

Two tabs:
  1. Ingest: Upload PDF → process → index
  2. Search: Query → streaming answer + evidence panel + execution trace
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import gradio as gr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from app.config import settings
from app.schemas import Chunk, ChunkMeta, QueryResponse

# ── Global state ────────────────────────────────────────────

_is_indexed = False
_index_status = "No document indexed. Upload a PDF in the Ingest tab."

# Router analytics tracking
_router_stats = {
    "total_queries": 0,
    "router_handled": 0,
    "planner_fallback": 0,
    "cache_hits": 0,
    "confidences": [],
    "latencies_with_router": [],
    "latencies_without_router": [],
}


def _check_indexed() -> bool:
    """Check if indices exist on disk."""
    global _is_indexed
    required = ["chunks.pkl", "bm25_index.pkl", "page_index.pkl", "section_index.pkl"]
    _is_indexed = all((settings.indices_dir / f).exists() for f in required)
    return _is_indexed


# ── Ingestion ───────────────────────────────────────────────

def ingest_pdf(files, progress=gr.Progress()):
    """Process and index multiple PDF files with live progress streaming."""
    global _is_indexed, _index_status

    if not files:
        yield "❌ No files uploaded."
        return

    # If Gradio returns a single file instead of a list, wrap it
    if not isinstance(files, list):
        files = [files]

    logger.info(f"Ingesting {len(files)} files...")

    try:
        from app.ingestion.loader import load_pdf
        from app.ingestion.cleaner import clean_document
        from app.ingestion.chunker import (
            chunk_document,
            get_embedding_model,
            save_chunking_result,
            ChunkingResult,
        )
        from app.ingestion.table_parser import extract_tables, save_table_store, TableStore
        import string
        from rank_bm25 import BM25Okapi

        status = f"📄 Starting batch ingestion of {len(files)} PDFs...\n"
        yield status

        progress(0, desc="Preparing embedding model...")
        embedding_model = get_embedding_model()

        # Global accumulators
        global_chunks = []
        global_section_index = {}
        global_page_index = {}
        global_bm25_corpus = []
        global_bm25_chunk_ids = []
        
        global_tables = []
        global_table_page_index = {}
        
        total_pages = 0
        doc_names = []

        for i, file in enumerate(files):
            file_path = file.name if hasattr(file, "name") else str(file)
            filename = file_path.split("/")[-1]
            status += f"\n[{i+1}/{len(files)}] Processing {filename}..."
            yield status
            logger.info(f"Processing: {file_path}")

            # 1. Load PDF
            loader_result = load_pdf(file_path)
            total_pages += loader_result.total_pages
            doc_names.append(loader_result.document_name)

            # 2. Clean text
            pages_text = [p.text for p in loader_result.pages]
            cleaned_pages = clean_document(pages_text)

            # 3. Semantic chunking (re-using preloaded embedding model)
            chunking_result = chunk_document(
                pages_text=cleaned_pages,
                document_name=loader_result.document_name,
                embedding_model=embedding_model,
            )
            
            # Aggregate Chunks
            global_chunks.extend(chunking_result.chunks)
            global_bm25_corpus.extend(chunking_result.bm25_corpus)
            global_bm25_chunk_ids.extend(chunking_result.bm25_chunk_ids)
            
            for section, ids in chunking_result.section_index.items():
                global_section_index.setdefault(section, []).extend(ids)
                
            for page, ids in chunking_result.page_index.items():
                global_page_index.setdefault(page, []).extend(ids)

            # 4. Extract tables
            table_store = extract_tables(file_path)
            
            # Aggregate Tables
            for tbl in table_store.tables:
                idx = len(global_tables)
                global_tables.append(tbl)
                global_table_page_index.setdefault(tbl.page_number, []).append(idx)

            # 4b. Create synthetic BM25 chunks from table content
            #     This makes table data searchable via keyword (BM25) queries
            for tbl in table_store.tables:
                import uuid
                table_chunk_id = f"tbl-{uuid.uuid4().hex[:8]}"
                table_text = tbl.markdown
                table_chunk = Chunk(
                    text=table_text,
                    metadata=ChunkMeta(
                        chunk_id=table_chunk_id,
                        page_number=tbl.page_number,
                        section_name=f"Table (page {tbl.page_number})",
                        document_name=tbl.document_name,
                        source="table",
                    ),
                )
                global_chunks.append(table_chunk)
                global_bm25_corpus.append(table_text)
                global_bm25_chunk_ids.append(table_chunk_id)
                global_page_index.setdefault(tbl.page_number, []).append(table_chunk_id)

            status += f" ✅ ({len(chunking_result.chunks)} chunks, {len(table_store.tables)} tables)"
            progress((i + 1) / (len(files) + 1), desc=f"Processed {filename}")
            yield status

        # 5. Build Unified BM25 Index over the global corpus
        status += f"\n\n⚙️ Building unified BM25 search index over {len(global_chunks)} chunks..."
        yield status
        logger.info("Building unified BM25 index...")
        tokenized_corpus = []
        for doc in global_bm25_corpus:
            words = doc.lower().split()
            tokenized_corpus.append([w.strip(string.punctuation) for w in words if w.strip(string.punctuation)])
            
        global_bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        # 6. Save Everything
        unified_chunking_result = ChunkingResult(
            chunks=global_chunks,
            section_index=global_section_index,
            page_index=global_page_index,
            bm25_index=global_bm25_index,
            bm25_corpus=global_bm25_corpus,
            bm25_chunk_ids=global_bm25_chunk_ids,
        )
        save_chunking_result(unified_chunking_result)
        
        unified_table_store = TableStore(
            tables=global_tables,
            page_index=global_table_page_index,
        )
        save_table_store(unified_table_store)

        # 7. Reload tools into memory
        status += "\n🔄 Reloading tools with new data..."
        yield status
        _load_tools()

        _is_indexed = True
        _index_status = f"✅ Indexed: {len(files)} documents ({total_pages} total pages)"

        status += "\n\n✅ Batch Indexing complete! Go to the Search tab to query."
        status += f"\n\n📈 Global Summary:"
        status += f"\n  • Documents: {len(files)}"
        status += f"\n  • Total Pages: {total_pages}"
        status += f"\n  • Total Chunks: {len(global_chunks)}"
        status += f"\n  • Total Sections: {len(global_section_index)}"
        status += f"\n  • Total Tables: {len(global_tables)}"
        status += f"\n  • Unified BM25 index: ready"
        progress(1.0, desc="Batch Indexing complete!")
        yield status

    except Exception as e:
        logger.exception("Batch ingestion failed")
        yield f"❌ Batch Ingestion failed: {e}"


def _load_tools():
    """Load all tools from persisted indices."""
    from app.tools import bm25_tool, page_tool, section_tool, table_tool
    from app.ingestion.chunker import get_embedding_model

    bm25_tool.load_bm25()
    page_tool.load_page_index()
    section_tool.load_section_index()
    table_tool.load_tables()
    # Preload embedding model into memory
    get_embedding_model()
    logger.info("All tools loaded successfully")


# ── Search ──────────────────────────────────────────────────

def search_query(query: str) -> tuple[str, str, str, str]:
    """
    Run a query through the agentic RAG pipeline.

    Returns: (answer, evidence_text, trace_text, metrics_text)
    """
    if not query.strip():
        return "Please enter a query.", "", "", ""

    # Check if indexed
    if not _check_indexed():
        _try_load_existing()
    if not _is_indexed:
        return "⚠️ No document indexed. Please upload a PDF in the Ingest tab first.", "", "", ""

    # Ensure tools are loaded
    from app.tools import bm25_tool
    if not bm25_tool.is_loaded():
        _load_tools()

    try:
        from app.orchestration.graph import run_query
        response: QueryResponse = asyncio.run(run_query(query))

        # Track router stats
        _router_stats["total_queries"] += 1
        total_ms = sum(s.duration_ms for s in response.trace)
        for step in response.trace:
            if step.node == "ml_router":
                conf = step.details.get("confidence", 0)
                _router_stats["confidences"].append(conf)
                if "ROUTE" in step.result_summary:
                    _router_stats["router_handled"] += 1
                    _router_stats["latencies_with_router"].append(total_ms)
                elif "FALLBACK" in step.result_summary or "DISABLED" in step.result_summary:
                    _router_stats["planner_fallback"] += 1
                    _router_stats["latencies_without_router"].append(total_ms)
            if step.node in ("exact_cache", "semantic_cache") and step.cache_hit:
                _router_stats["cache_hits"] += 1

        return _format_response(response)

    except Exception as e:
        logger.exception("Query failed")
        return f"❌ Error: {e}", "", "", ""


def _try_load_existing():
    """Try to load existing indices from disk."""
    global _is_indexed, _index_status
    if _check_indexed():
        try:
            _load_tools()
            _is_indexed = True
            _index_status = "✅ Loaded existing indices from disk"
        except Exception:
            pass


def _format_response(response: QueryResponse) -> tuple[str, str, str, str]:
    """Format the QueryResponse into display strings."""

    # ── Answer ──
    answer = response.answer
    if response.cache_type != "none":
        answer = f"⚡ **Cache Hit ({response.cache_type})**\n\n{answer}"

    # ── Evidence Panel ──
    evidence_parts = []
    for chunk in response.evidence.chunks:
        score_str = f" (score: {chunk.score:.3f})" if chunk.score > 0 else ""
        section = f"Section: {chunk.metadata.section_name}\n" if chunk.metadata.section_name else ""
        evidence_parts.append(
            f"### 📎 {chunk.metadata.source}{score_str}\n"
            f"{section}"
            f"```\n{chunk.text[:500]}\n```"
        )
    for table in response.evidence.tables:
        evidence_parts.append(
            f"### 📊 Table from Page {table.page_number}\n"
            f"{table.markdown}"
        )
    evidence_text = "\n\n---\n\n".join(evidence_parts) if evidence_parts else "*No evidence retrieved*"

    # ── Execution Trace ──
    trace_parts = []
    total_ms = 0.0
    for step in response.trace:
        icon = "✅" if step.cache_hit else "⚙️"
        trace_parts.append(
            f"{icon} **{step.node}** — {step.duration_ms:.0f}ms — {step.result_summary}"
        )
        total_ms += step.duration_ms

    trace_text = "\n\n".join(trace_parts) if trace_parts else "*No trace data*"

    # ── Metrics ──
    metrics_parts = [
        f"**LLM Calls:** {response.llm_calls}",
        f"**Cache:** {response.cache_type}",
        f"**Confidence:** {response.confidence:.1%}",
        f"**Total Latency:** {total_ms:.0f}ms",
    ]
    if response.citations:
        metrics_parts.append(f"**Citations:** {', '.join(response.citations)}")
    # Router info from trace
    for step in response.trace:
        if step.node == "ml_router" and step.details:
            label = step.details.get('label', '?')
            conf = step.details.get('confidence', 0)
            routed = '✅ Routed' if 'ROUTE' in step.result_summary else '🔄 Planner'
            metrics_parts.append(f"**Router:** {routed} ({label}, {conf:.2f})")
        if step.node == "semantic_cache" and step.details.get("similarity"):
            metrics_parts.append(f"**Semantic Similarity:** {step.details['similarity']:.4f}")

    metrics_text = " | ".join(metrics_parts)

    return answer, evidence_text, trace_text, metrics_text


def get_system_stats() -> str:
    """Get cache + router statistics."""
    parts = []

    # Cache stats
    try:
        from app.orchestration.graph import get_caches
        exact, semantic = get_caches()
        exact_stats = exact.stats()
        semantic_stats = semantic.stats()

        parts.append(
            f"## 💾 Cache Statistics\n\n"
            f"### Exact Cache\n"
            f"- Size: {exact_stats['size']} entries\n"
            f"- Hits: {exact_stats['hits']}\n"
            f"- Misses: {exact_stats['misses']}\n"
            f"- Hit Rate: {exact_stats['hit_rate']:.1%}\n\n"
            f"### Semantic Cache\n"
            f"- Size: {semantic_stats['size']} entries\n"
            f"- Hits: {semantic_stats['hits']}\n"
            f"- Misses: {semantic_stats['misses']}\n"
            f"- Hit Rate: {semantic_stats['hit_rate']:.1%}\n"
            f"- Threshold: {semantic_stats['threshold']}"
        )
    except Exception:
        parts.append("## 💾 Cache\n\nCache not initialized yet.")

    # Router stats
    from app.config import settings
    total = _router_stats["total_queries"]
    routed = _router_stats["router_handled"]
    fallback = _router_stats["planner_fallback"]
    cache_h = _router_stats["cache_hits"]
    confs = _router_stats["confidences"]

    parts.append(f"\n---\n\n## 🤖 ML Router Statistics\n")
    parts.append(
        f"- **Status:** {'✅ Enabled' if settings.router_enabled else '❌ Disabled'}\n"
        f"- **Model:** {settings.router_model_type}\n"
        f"- **Confidence Threshold:** {settings.router_confidence_threshold}\n"
    )

    if total > 0:
        non_cache = routed + fallback
        router_pct = 100 * routed / non_cache if non_cache > 0 else 0
        planner_pct = 100 * fallback / non_cache if non_cache > 0 else 0

        parts.append(
            f"### Query Routing (non-cache queries)\n"
            f"- Total queries: **{total}**\n"
            f"- Cache hits: **{cache_h}**\n"
            f"- ML Router handled: **{routed}** ({router_pct:.1f}%)\n"
            f"- Planner fallback: **{fallback}** ({planner_pct:.1f}%)\n"
            f"- **LLM calls saved:** {routed} (queries that skipped the planner)\n"
        )

        if confs:
            import statistics
            avg_conf = statistics.mean(confs)
            min_conf = min(confs)
            max_conf = max(confs)
            parts.append(
                f"### Confidence Distribution\n"
                f"- Average: **{avg_conf:.3f}**\n"
                f"- Min: {min_conf:.3f} | Max: {max_conf:.3f}\n"
            )

        lat_with = _router_stats["latencies_with_router"]
        lat_without = _router_stats["latencies_without_router"]
        if lat_with:
            avg_with = statistics.mean(lat_with)
            parts.append(f"### Latency\n- Avg with ML router: **{avg_with:.0f}ms**\n")
        if lat_without:
            avg_without = statistics.mean(lat_without)
            parts.append(f"- Avg with LLM planner: **{avg_without:.0f}ms**\n")
        if lat_with and lat_without:
            reduction = statistics.mean(lat_without) - statistics.mean(lat_with)
            parts.append(f"- **Latency reduction:** {reduction:.0f}ms\n")
    else:
        parts.append("\n*No queries processed yet. Submit queries in the Search tab.*")

    return "\n".join(parts)


# ── Build Gradio App ────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Build the Gradio interface."""

    with gr.Blocks(
        title="ML-RAG | PDF Agentic RAG",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        css="""
        .main-header { text-align: center; margin-bottom: 1rem; }
        .status-bar { padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        """,
    ) as app:

        gr.Markdown(
            "# 🤖 ML-RAG\n"
            "### PDF Agentic RAG with ML Query-Intent Router\n"
            "*ML Router • BM25 + Cross-Encoder Reranking • Semantic Caching • LangGraph*",
            elem_classes="main-header",
        )

        with gr.Tabs():
            # ── Tab 1: Ingest ──
            with gr.Tab("📥 Ingest", id="ingest"):
                gr.Markdown("### Upload a PDF document to index")
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Upload PDFs",
                            file_types=[".pdf"],
                            type="filepath",
                            file_count="multiple",
                        )
                        ingest_btn = gr.Button("🚀 Process & Index", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        ingest_output = gr.Textbox(
                            label="Ingestion Status",
                            lines=15,
                            interactive=False,
                        )

                ingest_btn.click(
                    fn=ingest_pdf,
                    inputs=[file_input],
                    outputs=[ingest_output],
                )

            # ── Tab 2: Search ──
            with gr.Tab("🔍 Search", id="search"):
                # Metrics bar at top
                metrics_bar = gr.Markdown("*Submit a query to see metrics*")

                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What are the key findings in section 3?",
                        lines=2,
                        scale=4,
                    )
                    search_btn = gr.Button("🔍 Search", variant="primary", scale=1)

                # Answer
                answer_output = gr.Markdown(label="Answer")

                with gr.Row():
                    # Evidence panel
                    with gr.Column():
                        with gr.Accordion("📎 Evidence", open=False):
                            evidence_output = gr.Markdown()

                    # Trace panel
                    with gr.Column():
                        with gr.Accordion("🔗 Execution Trace", open=False):
                            trace_output = gr.Markdown()

                search_btn.click(
                    fn=search_query,
                    inputs=[query_input],
                    outputs=[answer_output, evidence_output, trace_output, metrics_bar],
                )
                query_input.submit(
                    fn=search_query,
                    inputs=[query_input],
                    outputs=[answer_output, evidence_output, trace_output, metrics_bar],
                )

            # ── Tab 3: Analytics ──
            with gr.Tab("📊 Analytics", id="analytics"):
                gr.Markdown("### Cache & Router Statistics")
                stats_output = gr.Markdown("Click refresh to see stats")
                refresh_btn = gr.Button("🔄 Refresh Stats")
                refresh_btn.click(fn=get_system_stats, outputs=[stats_output])

    return app


# ── Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    # Copy .env.example to .env if not exists
    env_path = Path(".env")
    env_example = Path(".env.example")
    if not env_path.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_path)
        logger.info("Created .env from .env.example")

    # Try to load existing indices
    _try_load_existing()

    # Launch
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
