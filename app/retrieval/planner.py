"""
Planner Agent — Groq SDK.
Routes queries to the right tools. NEVER answers questions.
"""

from __future__ import annotations

import json
import logging
import re
import time

from groq import AsyncGroq

from app.config import settings
from app.schemas import PlannerOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a query planner for a document retrieval system. Given a user question about a PDF document, \
you decide which retrieval tools to use. You MUST NOT answer the question. You ONLY select tools and parameters.

Available tools:
- bm25: Keyword search across all document chunks. Use for general questions, factual lookups, \
and when you need to find specific information across the entire document.
- page_retriever: Fetch full content from specific page numbers. Use when the user mentions \
a page number or you need content from a known page.
- section_retriever: Fetch content from a named section or chapter. Use when the user mentions \
a section, chapter, heading, or specific part of the document.
- table_extractor: Extract tables from the document. Use when the question involves numerical data, \
comparisons, dosages, statistics, tabular information, or when you are looking for specific \
reference items, functions, or commands that might be listed in a table.

You can select MULTIPLE tools when needed. For example, if the question asks about data in a \
specific section, use both section_retriever and table_extractor.

Respond ONLY with valid JSON in this exact format:
{"tools": ["tool1", "tool2"], "params": {"query": "search terms", "pages": [1, 2], "section": "section name"}}

Rules:
- "query" param is required when using "bm25"
- "pages" param is a list of integers, required when using "page_retriever" or optionally with "table_extractor"
- "section" param is a string, required when using "section_retriever"
- Always include a "query" param with relevant search terms, even if not using bm25
- Do NOT include any explanation, only the JSON object
"""

RETRY_TEMPLATE = """\
The previous retrieval attempt was incomplete. Issues found: {issues}

The original question was: {query}

Previous tools used: {prev_tools}

Select DIFFERENT or ADDITIONAL tools to find the missing information. \
Try different search terms or look in different sections/pages.

Respond ONLY with valid JSON:
{{"tools": ["tool1"], "params": {{"query": "different search terms"}}}}
"""

VALID_TOOLS = {"bm25", "page_retriever", "section_retriever", "table_extractor"}


async def run_planner(
    query: str,
    retry_issues: list[str] | None = None,
    prev_tools: list[str] | None = None,
) -> tuple[PlannerOutput, dict]:
    """
    Run the planner LLM to decide which tools to use.

    Args:
        query: User's question.
        retry_issues: Issues from the critic (if retrying).
        prev_tools: Tools used in previous attempt (if retrying).

    Returns:
        (PlannerOutput, metrics_dict)
    """
    start = time.time()

    if not settings.groq_api_key:
        logger.warning("No Groq API key found, falling back to BM25")
        return (
            PlannerOutput(tools=["bm25"], params={"query": query}),
            {"duration_ms": (time.time() - start) * 1000, "tokens": 0, "fallback": True},
        )

    client = AsyncGroq(api_key=settings.groq_api_key)

    # Build prompt
    if retry_issues:
        user_prompt = RETRY_TEMPLATE.format(
            issues=", ".join(retry_issues),
            query=query,
            prev_tools=prev_tools or [],
        )
    else:
        user_prompt = f"Question: {query}"

    try:
        response = await client.chat.completions.create(
            model=settings.planner_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        
        raw_response = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0

    except Exception as e:
        logger.error(f"Planner LLM call failed: {e}")
        # Fallback: just use BM25
        return (
            PlannerOutput(tools=["bm25"], params={"query": query}),
            {"duration_ms": (time.time() - start) * 1000, "tokens": 0, "fallback": True},
        )

    # Parse JSON from response
    planner_output = _parse_planner_response(raw_response, query)

    duration_ms = (time.time() - start) * 1000
    metrics = {
        "duration_ms": duration_ms,
        "tokens": tokens_used,
        "raw_response": raw_response[:500],
        "fallback": False,
    }

    logger.info(
        f"Planner decided: tools={planner_output.tools}, "
        f"params={planner_output.params} ({duration_ms:.0f}ms)"
    )
    return planner_output, metrics


def _parse_planner_response(raw: str, original_query: str) -> PlannerOutput:
    """
    Parse the planner's JSON response with fallback regex extraction.
    """
    raw = raw.strip()

    # Try direct JSON parse
    try:
        data = json.loads(raw)
        return _validate_planner_output(data, original_query)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return _validate_planner_output(data, original_query)
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return _validate_planner_output(data, original_query)
        except json.JSONDecodeError:
            pass

    # Complete fallback
    logger.warning(f"Could not parse planner response, falling back to BM25: {raw[:200]}")
    return PlannerOutput(tools=["bm25"], params={"query": original_query})


def _validate_planner_output(data: dict, original_query: str) -> PlannerOutput:
    """Validate and sanitize planner output."""
    tools = data.get("tools", ["bm25"])

    # Filter to valid tools only
    tools = [t for t in tools if t in VALID_TOOLS]
    if not tools:
        tools = ["bm25"]

    params = data.get("params", {})

    # Ensure query param exists
    if "query" not in params:
        params["query"] = original_query

    # Ensure pages is a list of ints
    if "pages" in params:
        try:
            params["pages"] = [int(p) for p in params["pages"]]
        except (TypeError, ValueError):
            del params["pages"]

    return PlannerOutput(tools=tools, params=params)
