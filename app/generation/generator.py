"""
Generator — Groq SDK.
Generates answers from evidence with streaming support.
Returns citations and confidence score.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator

from groq import AsyncGroq

from app.config import settings
from app.schemas import Evidence

logger = logging.getLogger(__name__)

GENERATOR_PROMPT = """\
You are a precise document analysis assistant. Based ONLY on the evidence provided below, \
answer the user's question. Follow these rules strictly:

1. ONLY use information from the provided evidence. Do NOT make up information.
2. Cite sources using [Page X] or [Section: name] format.
3. If the evidence is insufficient to fully answer, say what you CAN answer and note what's missing.
4. Be concise and direct.
5. At the end of your answer, add a confidence line: "Confidence: high/medium/low"

EVIDENCE:
{evidence_text}

QUESTION: {query}

ANSWER:"""


def _format_evidence(evidence: Evidence) -> str:
    """Format evidence into a text block for the generator prompt."""
    parts: list[str] = []

    for chunk in evidence.chunks:
        source = chunk.metadata.source or f"Page {chunk.metadata.page_number}"
        section = f", Section: {chunk.metadata.section_name}" if chunk.metadata.section_name else ""
        parts.append(f"[{source}{section}]\n{chunk.text}")

    for table in evidence.tables:
        parts.append(f"[Table from {table.document_name} - Page {table.page_number}]\n{table.markdown}")

    return "\n\n---\n\n".join(parts) if parts else "(No evidence available)"


async def generate_answer(
    query: str,
    evidence: Evidence,
) -> tuple[str, dict]:
    """
    Generate a complete answer (non-streaming).

    Returns:
        (answer_text, metrics_dict)
    """
    start = time.time()
    
    if not settings.groq_api_key:
        return "Error: Groq API key not configured.", {"duration_ms": 0, "tokens": 0, "confidence": 0.0, "citations": []}

    client = AsyncGroq(api_key=settings.groq_api_key)

    evidence_text = _format_evidence(evidence)
    prompt = GENERATOR_PROMPT.format(evidence_text=evidence_text, query=query)

    try:
        response = await client.chat.completions.create(
            model=settings.generator_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512,
        )

        answer = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0

    except Exception as e:
        logger.error(f"Generator LLM call failed: {e}")
        answer = f"Error generating answer: {e}"
        tokens_used = 0

    duration_ms = (time.time() - start) * 1000

    # Extract confidence from answer
    confidence = _extract_confidence(answer)

    # Extract citations
    citations = _extract_citations(answer)

    metrics = {
        "duration_ms": duration_ms,
        "tokens": tokens_used,
        "confidence": confidence,
        "citations": citations,
    }

    logger.info(f"Generated answer ({duration_ms:.0f}ms, {tokens_used} tokens)")
    return answer, metrics


async def generate_answer_stream(
    query: str,
    evidence: Evidence,
) -> AsyncGenerator[str, None]:
    """
    Generate an answer with token-by-token streaming.
    Yields individual tokens/chunks as they arrive from Groq.
    """
    if not settings.groq_api_key:
        yield "Error: Groq API key not configured."
        return

    client = AsyncGroq(api_key=settings.groq_api_key)

    evidence_text = _format_evidence(evidence)
    prompt = GENERATOR_PROMPT.format(evidence_text=evidence_text, query=query)

    try:
        response_stream = await client.chat.completions.create(
            model=settings.generator_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512,
            stream=True,
        )
        
        async for chunk in response_stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        yield f"\n\n[Error: {e}]"


def _extract_confidence(answer: str) -> float:
    """Extract confidence level from the answer text."""
    lower = answer.lower()
    if "confidence: high" in lower:
        return 0.9
    elif "confidence: medium" in lower:
        return 0.6
    elif "confidence: low" in lower:
        return 0.3
    return 0.5  # Default


def _extract_citations(answer: str) -> list[str]:
    """Extract [Page X] and [Section: Y] citations from the answer."""
    import re
    citations = re.findall(r"\[(?:Page\s+\d+|Section:\s*[^\]]+)\]", answer)
    return list(set(citations))
