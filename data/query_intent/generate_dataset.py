#!/usr/bin/env python3
"""
Query-Intent Dataset Generator for ML-RAG.

Generates a balanced, labelled query-intent dataset with 4 classes:
  Page, Section, Table, General
Each mapping to a retrieval tool:
  PageRetriever, SectionRetriever, TableExtractor, BM25Search

Strategy:
  1. 20 templates per class with {slot} placeholders
  2. Programmatic slot filling with variation (casing, punctuation, phrasing)
  3. Hard negatives (~15-20% per class) for known confusion pairs
  4. Grouped stratified 70/15/15 split on base_instance_id
  5. Fixed random seed (42) for full reproducibility

Usage:
  python data/query_intent/generate_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SEED = 42
TARGET_PER_CLASS = 400
HARD_NEGATIVE_RATIO = 0.18  # ~18% hard negatives per class
OUTPUT_DIR = Path(__file__).parent

random.seed(SEED)


# ════════════════════════════════════════════════════════════════
# Schema
# ════════════════════════════════════════════════════════════════

COLUMNS = [
    "id", "base_instance_id", "query", "label", "tool",
    "template_id", "generation_method", "is_hard_negative",
]

LABEL_TOOL_MAP = {
    "Page": "PageRetriever",
    "Section": "SectionRetriever",
    "Table": "TableExtractor",
    "General": "BM25Search",
}


@dataclass
class QueryInstance:
    query: str
    label: str
    template_id: str
    is_hard_negative: bool = False
    generation_method: str = "template"
    base_instance_id: str = ""

    def __post_init__(self):
        self.tool = LABEL_TOOL_MAP[self.label]
        if not self.base_instance_id:
            # Hash the template_id + first slot value combo as base ID
            self.base_instance_id = self.template_id


# ════════════════════════════════════════════════════════════════
# Slot Value Banks
# ════════════════════════════════════════════════════════════════

PAGE_NUMBERS = list(range(1, 51)) + [55, 60, 75, 100, 120, 150, 200, 250, 300]

PAGE_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 20: "twenty", 25: "twenty-five", 30: "thirty",
    50: "fifty", 100: "one hundred",
}

PAGE_RANGES = [
    "10–12", "10 to 12", "pages 3 through 5", "45–47", "45 to 47",
    "1–3", "pages 1 through 3", "20 to 22", "pages 7 through 9",
    "100–102", "pages 15 through 17", "5–7", "pages 30 through 32",
    "8 to 10", "pages 50 to 52",
]

ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "twelfth",
    "1st", "2nd", "3rd", "4th", "5th", "10th", "15th", "20th",
    "last", "penultimate",
]

DOC_WORDS = [
    "manual", "report", "document", "PDF", "contract", "paper",
    "handbook", "policy document", "spec sheet", "guide",
    "whitepaper", "specification", "brochure", "proposal",
]

SECTION_NAMES = [
    "Introduction", "Abstract", "Executive Summary", "Background",
    "Literature Review", "Related Work", "Methodology", "Methods",
    "Results", "Discussion", "Conclusion", "Recommendations",
    "Appendix", "References", "Acknowledgments", "Terms and Conditions",
    "Scope of Work", "Definitions", "Limitations", "Financial Highlights",
    "Risk Factors", "Governing Law", "Confidentiality Clause",
    "Termination Clause", "Eligibility Criteria", "Warranty",
    "Disclaimer", "Safety Instructions", "Installation Guide",
    "Troubleshooting", "FAQ", "Overview", "Objectives",
    "Assumptions", "Glossary",
]

TABLE_TOPICS = [
    "quarterly revenue", "annual budget", "pricing plan",
    "product specifications", "model comparison", "employee headcount",
    "sales figures", "expense breakdown", "performance metrics",
    "test results", "survey responses", "shipment schedule",
    "inventory list", "exam timetable", "nutritional information",
    "technical specifications", "risk matrix", "project milestones",
    "financial statements", "tax rates", "discount tiers",
    "warranty comparison", "feature comparison", "benchmark results",
    "growth projections", "salary bands", "attendance record",
    "grading rubric", "ingredient list", "error codes",
]

GENERAL_TOPICS = [
    "refund policy", "data privacy", "warranty period",
    "cancellation process", "eligibility criteria", "late fees",
    "security measures", "encryption standard", "onboarding process",
    "termination rights", "dispute resolution", "force majeure",
    "intellectual property", "non-disclosure obligations",
    "employee benefits", "leave policy", "code of conduct",
    "quality assurance", "escalation process", "customer support",
    "compliance requirements", "data retention",
    "service level agreement", "liability limits", "renewal terms",
]


# ════════════════════════════════════════════════════════════════
# Template Banks (20 per class, with variation helpers)
# ════════════════════════════════════════════════════════════════

def _vary_prefix(query: str) -> list[str]:
    """Generate minor phrasing variants of a query."""
    variants = [query]
    # Add/remove trailing punctuation
    if query.endswith("."):
        variants.append(query[:-1])
        variants.append(query[:-1] + "?")
    elif query.endswith("?"):
        variants.append(query[:-1] + ".")
    else:
        variants.append(query + ".")
        variants.append(query + "?")
    # Add please
    if "please" not in query.lower():
        variants.append(query.rstrip(".?!") + ", please.")
    return variants


def _choose_page_str(page: int) -> str:
    """Return either digit or word form of a page number."""
    if page in PAGE_WORDS and random.random() < 0.3:
        return PAGE_WORDS[page]
    return str(page)


# ── Page Templates ──────────────────────────────────────────

def generate_page_queries() -> list[QueryInstance]:
    """Generate ~400 Page-class queries from 20 templates."""
    instances: list[QueryInstance] = []

    templates = [
        ("page_01", lambda p, d: f"Show me page {p} of the {d}"),
        ("page_02", lambda p, d: f"What's on page {p}"),
        ("page_03", lambda p, d: f"Can you pull up page {p}"),
        ("page_04", lambda p, d: f"Go to page {p}, please"),
        ("page_05", lambda p, d: f"I need the content from page {p}"),
        ("page_06", lambda p, d: f"Open page {p} of the {d}"),
        ("page_07", lambda p, d: f"What does it say on page {p}"),
        ("page_08", lambda p, d: f"Display page {p}"),
        ("page_09", lambda p, d: f"Fetch page number {p}"),
        ("page_10", lambda p, d: f"Take me to page {p}"),
        ("page_11", lambda p, d: f"Read page {p} of this {d}"),
        ("page_12", lambda p, d: f"What information is on page {p}"),
        ("page_13", lambda p, d: f"Jump to page {p}"),
        ("page_14", lambda p, d: f"I want to see page {p}"),
        ("page_15", lambda p, d: f"Give me everything on page {p}"),
        ("page_16", lambda p, d: f"Pull the text from page {p}"),
        ("page_17", lambda p, d: f"Could you show me page {p}"),
        ("page_18", lambda p, d: f"Let me see page {p} of the {d}"),
        ("page_19", lambda p, d: f"Navigate to page {p}"),
        ("page_20", lambda p, d: f"What content appears on page {p}"),
    ]

    # Range-specific templates
    range_templates = [
        ("page_r1", lambda r: f"Retrieve pages {r}"),
        ("page_r2", lambda r: f"Can you show {r}"),
        ("page_r3", lambda r: f"Show me {r}"),
        ("page_r4", lambda r: f"Get me the content from {r}"),
    ]

    # Ordinal templates
    ordinal_templates = [
        ("page_o1", lambda o, d: f"What's written on the {o} page"),
        ("page_o2", lambda o, d: f"Show me the {o} page of the {d}"),
        ("page_o3", lambda o, d: f"What does the {o} page say"),
    ]

    # Special templates
    special_templates = [
        ("page_s1", lambda: "What's the content on the last page"),
        ("page_s2", lambda: "Show me the first page"),
        ("page_s3", lambda: "What's on the very last page of the document"),
    ]

    # Fill main templates
    page_sample = random.sample(PAGE_NUMBERS, min(len(PAGE_NUMBERS), 45))
    for tid, tmpl in templates:
        # Pick a subset of pages and docs for this template
        pages_for_tmpl = random.sample(page_sample, min(18, len(page_sample)))
        docs_for_tmpl = random.sample(DOC_WORDS, min(6, len(DOC_WORDS)))
        for page in pages_for_tmpl:
            doc = random.choice(docs_for_tmpl)
            p_str = _choose_page_str(page)
            q = tmpl(p_str, doc)
            base_id = f"{tid}_{page}"
            instances.append(QueryInstance(
                query=q, label="Page", template_id=tid,
                base_instance_id=base_id,
            ))

    # Fill range templates
    for tid, tmpl in range_templates:
        for rng in random.sample(PAGE_RANGES, min(12, len(PAGE_RANGES))):
            q = tmpl(rng)
            base_id = f"{tid}_{hashlib.md5(rng.encode()).hexdigest()[:8]}"
            instances.append(QueryInstance(
                query=q, label="Page", template_id=tid,
                base_instance_id=base_id,
            ))

    # Fill ordinal templates
    for tid, tmpl in ordinal_templates:
        for ordinal in random.sample(ORDINALS, min(12, len(ORDINALS))):
            doc = random.choice(DOC_WORDS)
            q = tmpl(ordinal, doc)
            base_id = f"{tid}_{ordinal}"
            instances.append(QueryInstance(
                query=q, label="Page", template_id=tid,
                base_instance_id=base_id,
            ))

    # Add special queries
    for tid, tmpl in special_templates:
        q = tmpl()
        instances.append(QueryInstance(
            query=q, label="Page", template_id=tid,
            base_instance_id=tid,
        ))

    # Add phrasing variants for diversity
    extra: list[QueryInstance] = []
    for inst in random.sample(instances, min(60, len(instances))):
        q = inst.query
        variants = []
        if not q.endswith("?"):
            variants.append(q.rstrip(".!,") + "?")
        if not q.endswith("."):
            variants.append(q.rstrip("?!,") + ".")
        if "please" not in q.lower():
            variants.append(q.rstrip(".?!") + ", please.")
        if q.startswith("Show") and "Could" not in q:
            variants.append("Could you s" + q[1:])
        if q.startswith("Can you"):
            variants.append("Could you" + q[7:])
        for v in variants[:1]:  # take just 1 variant per source
            extra.append(QueryInstance(
                query=v, label=inst.label, template_id=inst.template_id + "_var",
                base_instance_id=inst.base_instance_id,
            ))
    instances.extend(extra)

    return instances


# ── Section Templates ───────────────────────────────────────

def generate_section_queries() -> list[QueryInstance]:
    """Generate ~400 Section-class queries from 20 templates."""
    instances: list[QueryInstance] = []

    templates = [
        ("sec_01", lambda s, d: f"What does the {s} section say"),
        ("sec_02", lambda s, d: f"Summarize the {s}"),
        ("sec_03", lambda s, d: f"Can you explain the {s} part"),
        ("sec_04", lambda s, d: f"Tell me about the {s} section"),
        ("sec_05", lambda s, d: f"What's covered under {s}"),
        ("sec_06", lambda s, d: f"Give me a summary of the {s}"),
        ("sec_07", lambda s, d: f"What does the {d} say in its {s}"),
        ("sec_08", lambda s, d: f"Walk me through the {s} section"),
        ("sec_09", lambda s, d: f"What are the key points in {s}"),
        ("sec_10", lambda s, d: f"Explain what's written in the {s}"),
        ("sec_11", lambda s, d: f"What is discussed in the {s} chapter"),
        ("sec_12", lambda s, d: f"Show me the {s} section"),
        ("sec_13", lambda s, d: f"What details are in the {s}"),
        ("sec_14", lambda s, d: f"How is {s} described in the document"),
        ("sec_15", lambda s, d: f"Break down the {s} for me"),
        ("sec_16", lambda s, d: f"What's the gist of the {s}"),
        ("sec_17", lambda s, d: f"Recap the {s} section"),
        ("sec_18", lambda s, d: f"Pull up the {s} part of the {d}"),
        ("sec_19", lambda s, d: f"What information is under {s}"),
        ("sec_20", lambda s, d: f"Does the {s} mention anything specific"),
    ]

    for tid, tmpl in templates:
        sections_for_tmpl = random.sample(SECTION_NAMES, min(18, len(SECTION_NAMES)))
        docs_for_tmpl = random.sample(DOC_WORDS, min(4, len(DOC_WORDS)))
        for section in sections_for_tmpl:
            doc = random.choice(docs_for_tmpl)
            q = tmpl(section, doc)
            base_id = f"{tid}_{hashlib.md5(section.encode()).hexdigest()[:8]}"
            instances.append(QueryInstance(
                query=q, label="Section", template_id=tid,
                base_instance_id=base_id,
            ))

    # Add phrasing variants
    extra: list[QueryInstance] = []
    for inst in random.sample(instances, min(50, len(instances))):
        q = inst.query
        variants = []
        if not q.endswith("?"):
            variants.append(q.rstrip(".!,") + "?")
        if not q.endswith("."):
            variants.append(q.rstrip("?!,") + ".")
        if "please" not in q.lower():
            variants.append(q.rstrip(".?!") + ", please.")
        for v in variants[:1]:
            extra.append(QueryInstance(
                query=v, label=inst.label, template_id=inst.template_id + "_var",
                base_instance_id=inst.base_instance_id,
            ))
    instances.extend(extra)

    return instances


# ── Table Templates ─────────────────────────────────────────

def generate_table_queries() -> list[QueryInstance]:
    """Generate ~400 Table-class queries from 20 templates."""
    instances: list[QueryInstance] = []

    templates = [
        ("tbl_01", lambda t: f"Extract the table on {t}"),
        ("tbl_02", lambda t: f"Show me the {t} table"),
        ("tbl_03", lambda t: f"Can you pull the data table about {t}"),
        ("tbl_04", lambda t: f"Get me the numbers from the {t} table"),
        ("tbl_05", lambda t: f"What does the table on {t} show"),
        ("tbl_06", lambda t: f"I need the {t} data in table form"),
        ("tbl_07", lambda t: f"Extract the {t} figures from the table"),
        ("tbl_08", lambda t: f"Give me the table comparing {t}"),
        ("tbl_09", lambda t: f"Pull the {t} statistics table"),
        ("tbl_10", lambda t: f"What are the values in the {t} table"),
        ("tbl_11", lambda t: f"Retrieve the table for {t}"),
        ("tbl_12", lambda t: f"Fetch the {t} table from the document"),
        ("tbl_13", lambda t: f"Show the tabulated {t} data"),
        ("tbl_14", lambda t: f"List the rows in the {t} table"),
        ("tbl_15", lambda t: f"Convert the {t} table into something readable"),
        ("tbl_16", lambda t: f"Give me the breakdown table for {t}"),
        ("tbl_17", lambda t: f"What numbers are shown in the {t} table"),
        ("tbl_18", lambda t: f"Extract all data from the {t} table"),
        ("tbl_19", lambda t: f"Summarize the {t} table"),
        ("tbl_20", lambda t: f"Is there a table showing {t}"),
    ]

    for tid, tmpl in templates:
        topics_for_tmpl = random.sample(TABLE_TOPICS, min(18, len(TABLE_TOPICS)))
        for topic in topics_for_tmpl:
            q = tmpl(topic)
            base_id = f"{tid}_{hashlib.md5(topic.encode()).hexdigest()[:8]}"
            instances.append(QueryInstance(
                query=q, label="Table", template_id=tid,
                base_instance_id=base_id,
            ))

    # Add phrasing variants
    extra: list[QueryInstance] = []
    for inst in random.sample(instances, min(50, len(instances))):
        q = inst.query
        variants = []
        if not q.endswith("?"):
            variants.append(q.rstrip(".!,") + "?")
        if not q.endswith("."):
            variants.append(q.rstrip("?!,") + ".")
        if "please" not in q.lower():
            variants.append(q.rstrip(".?!") + ", please.")
        for v in variants[:1]:
            extra.append(QueryInstance(
                query=v, label=inst.label, template_id=inst.template_id + "_var",
                base_instance_id=inst.base_instance_id,
            ))
    instances.extend(extra)

    return instances


# ── General Templates ───────────────────────────────────────

def generate_general_queries() -> list[QueryInstance]:
    """Generate ~400 General-class queries from 20 templates."""
    instances: list[QueryInstance] = []

    templates = [
        ("gen_01", lambda t: f"What is the {t}"),
        ("gen_02", lambda t: f"Does this document mention {t}"),
        ("gen_03", lambda t: f"How does {t} work according to this document"),
        ("gen_04", lambda t: f"Explain {t} as described in the document"),
        ("gen_05", lambda t: f"Is there anything about {t} in here"),
        ("gen_06", lambda t: f"What are the requirements for {t}"),
        ("gen_07", lambda t: f"What does the document say about {t}"),
        ("gen_08", lambda t: f"Find information about {t}"),
        ("gen_09", lambda t: f"What's the policy on {t}"),
        ("gen_10", lambda t: f"Are there any mentions of {t}"),
        ("gen_11", lambda t: f"Does the document cover {t}"),
        ("gen_12", lambda t: f"What's the definition of {t}"),
        ("gen_13", lambda t: f"Search for {t} in this document"),
        ("gen_14", lambda t: f"What can you tell me about {t}"),
        ("gen_15", lambda t: f"Is {t} addressed anywhere in the document"),
        ("gen_16", lambda t: f"What are the key takeaways about {t}"),
        ("gen_17", lambda t: f"How is {t} defined here"),
        ("gen_18", lambda t: f"What steps are involved in {t}"),
        ("gen_19", lambda t: f"Who is responsible for {t}"),
        ("gen_20", lambda t: f"What's the procedure for {t}"),
    ]

    for tid, tmpl in templates:
        topics_for_tmpl = random.sample(GENERAL_TOPICS, min(18, len(GENERAL_TOPICS)))
        for topic in topics_for_tmpl:
            q = tmpl(topic)
            base_id = f"{tid}_{hashlib.md5(topic.encode()).hexdigest()[:8]}"
            instances.append(QueryInstance(
                query=q, label="General", template_id=tid,
                base_instance_id=base_id,
            ))

    # Add phrasing variants
    extra: list[QueryInstance] = []
    for inst in random.sample(instances, min(50, len(instances))):
        q = inst.query
        variants = []
        if not q.endswith("?"):
            variants.append(q.rstrip(".!,") + "?")
        if not q.endswith("."):
            variants.append(q.rstrip("?!,") + ".")
        if "please" not in q.lower():
            variants.append(q.rstrip(".?!") + ", please.")
        for v in variants[:1]:
            extra.append(QueryInstance(
                query=v, label=inst.label, template_id=inst.template_id + "_var",
                base_instance_id=inst.base_instance_id,
            ))
    instances.extend(extra)

    return instances


# ════════════════════════════════════════════════════════════════
# Hard Negatives — deliberately ambiguous / confusable examples
# ════════════════════════════════════════════════════════════════

def generate_hard_negatives() -> list[QueryInstance]:
    """
    Generate hard negatives for known confusion pairs:
      - Section vs. General (biggest confusion pair)
      - Page vs. Section
      - Table vs. General

    Labelling rules:
      1. If a query names a specific section header → Section
         If it asks about a topic without naming a section → General
      2. If a query has a concrete page number AND section → Page
      3. Must have explicit table/extract cue → Table; otherwise → General
      4. Words "page"/"section"/"table" in non-lookup context → use judgment
    """
    instances: list[QueryInstance] = []
    hn_id = 0

    def _hn(query: str, label: str, confusion_pair: str) -> QueryInstance:
        nonlocal hn_id
        hn_id += 1
        return QueryInstance(
            query=query,
            label=label,
            template_id=f"hn_{confusion_pair}_{hn_id:03d}",
            is_hard_negative=True,
            base_instance_id=f"hn_{confusion_pair}_{hn_id:03d}",
        )

    # ── Section vs. General (biggest confusion pair) ──────────

    # These name a specific section header → Section
    section_queries_labeled_section = [
        "What does the Refunds section say?",
        "Tell me about the Warranty section.",
        "What's covered in the Termination Clause?",
        "Summarize the Confidentiality Clause section.",
        "What does the Governing Law part cover?",
        "Explain the Eligibility Criteria section.",
        "What information is in the Disclaimer?",
        "Walk me through the Safety Instructions section.",
        "What is in the Executive Summary?",
        "Can you explain the Scope of Work?",
        "Break down the Risk Factors section.",
        "What's written in the Financial Highlights?",
        "Give me the key points from the Methodology.",
        "What's discussed in the Literature Review?",
        "Recap the Terms and Conditions.",
        "What does the Non-Disclosure section state?",
        "Tell me what's in the Assumptions section.",
        "What does the Glossary define?",
        "Explain the Troubleshooting section.",
        "Show me what's under Recommendations.",
        "What are the key points in the Results section?",
        "Does the Discussion section mention any limitations?",
        "What's the gist of the Installation Guide?",
        "Pull up the FAQ section for me.",
        "What details are in the Acknowledgments?",
    ]
    for q in section_queries_labeled_section:
        instances.append(_hn(q, "Section", "sec_gen"))

    # These ask about a topic without naming a section → General
    general_queries_labeled_general = [
        "What does the document say about refunds?",
        "Is there anything about warranties in here?",
        "What's the policy on termination?",
        "Does this document mention confidentiality?",
        "How does the governing law apply?",
        "What are the eligibility requirements?",
        "Is there a disclaimer in this document?",
        "What safety instructions are provided?",
        "Can you find the executive summary details?",
        "What is the scope of work described here?",
        "Are there any risk factors mentioned?",
        "What financial highlights are discussed?",
        "How is the methodology described?",
        "What does the literature review cover?",
        "What terms and conditions apply?",
        "Are there any non-disclosure requirements?",
        "What assumptions were made?",
        "Is there a glossary in this document?",
        "What troubleshooting advice is given?",
        "What recommendations are made?",
        "What results were found?",
        "Are there any limitations discussed?",
        "How do I install the software?",
        "What frequently asked questions are addressed?",
        "Who is acknowledged in this document?",
    ]
    for q in general_queries_labeled_general:
        instances.append(_hn(q, "General", "sec_gen"))

    # ── Page vs. Section ──────────────────────────────────────

    # Has page number + section name → Page (page is more actionable)
    page_section_labeled_page = [
        "What's on page 5 about the methodology?",
        "Show me page 12 where it discusses the results.",
        "Can I see page 3 of the introduction?",
        "What does page 8 say about risk factors?",
        "Pull up page 15, the part about recommendations.",
        "Go to page 22 where the conclusion is.",
        "Fetch page 7 with the executive summary.",
        "Take me to page 45 about the financial highlights.",
        "What's on page 10 regarding the scope of work?",
        "Display page 30, the warranty section.",
        "Read page 1 of the abstract.",
        "Show me page 50 about the appendix.",
        "What does page 18 in the methodology cover?",
        "I need page 25 where limitations are discussed.",
        "Open page 40 with the terms and conditions.",
        "What's on page 6 about safety instructions?",
        "Go to page 33 where it talks about compliance.",
        "Can I see page 14 of the troubleshooting guide?",
        "Pull up page 9 with the installation steps.",
        "What does page 27 say about data privacy?",
        "Show page 2 of the introduction section.",
        "Read page 19 where it covers eligibility.",
        "Display page 35, the references part.",
        "Take me to page 44 about the glossary.",
        "What's on page 11 about termination?",
        "I need page 55 with the appendix tables.",
        "Fetch page 4 about the project objectives.",
        "Open page 28, the scope section.",
        "What does page 16 cover about warranties?",
        "Go to page 38 in the FAQ.",
        "Pull up page 21 about dispute resolution.",
        "Show me page 42 with the budget overview.",
        "Can I see page 13 about the background?",
        "What's written on page 37 about assumptions?",
        "Read page 48 of the closing remarks.",
    ]
    for q in page_section_labeled_page:
        instances.append(_hn(q, "Page", "page_sec"))

    # Has section name but no page → Section
    section_no_page = [
        "What does the methodology part say about sampling?",
        "Tell me about the results in the discussion chapter.",
        "What's in the risk factors section about credit risk?",
        "Explain the recommendations section's key findings.",
        "What conclusions are drawn in the conclusion section?",
        "Summarize the executive summary's main points.",
        "How are limitations addressed in the limitations section?",
        "What does the scope of work section specify about deliverables?",
        "Walk me through the warranty section's coverage.",
        "What financial data is in the financial highlights section?",
    ]
    for q in section_no_page:
        instances.append(_hn(q, "Section", "page_sec"))

    # ── Table vs. General ─────────────────────────────────────

    # Has explicit table/extract cue → Table
    table_explicit = [
        "Extract the quarterly revenue table.",
        "Show me the data table for employee headcount.",
        "Pull the pricing table from the document.",
        "Get the table comparing product specifications.",
        "Fetch the risk matrix table.",
        "Extract the salary bands table.",
        "Show the tabulated performance metrics.",
        "Give me the benchmark results table.",
        "Pull the tax rates table.",
        "Is there a table showing growth projections?",
        "Extract the table with discount tiers.",
        "Show me the feature comparison table.",
        "Retrieve the financial statements table.",
        "Give me the attendance record in table form.",
        "Can you extract the grading rubric table?",
        "Extract the shipment schedule table.",
        "Show me the error codes table.",
        "Pull the ingredient list table.",
        "Get the table listing exam timetable data.",
        "Fetch the nutritional information table.",
        "Extract the table with project milestones.",
        "Show me the tabulated survey responses.",
        "Pull the warranty comparison table.",
        "Give me the test results in table form.",
        "Can you extract the expense breakdown table?",
        "Retrieve the table comparing model performance.",
        "Show the annual budget table.",
        "Extract the inventory list table.",
        "Get the data table for sales figures.",
        "Fetch the table of technical specifications.",
        "Pull the tabulated data on discount tiers.",
        "Give me the growth projections in table format.",
        "Show me the table with benchmark scores.",
        "Extract all tables about financial data.",
        "Can you pull up the pricing plan table?",
    ]
    for q in table_explicit:
        instances.append(_hn(q, "Table", "tbl_gen"))

    # Mentions numbers/data but no explicit table cue → General
    general_no_table_cue = [
        "What are the quarterly revenue figures?",
        "How many employees does the company have?",
        "What's the pricing structure?",
        "What are the product specifications?",
        "What risks are identified in the risk matrix?",
        "What are the salary ranges?",
        "How did the product perform?",
        "What were the benchmark results?",
        "What tax rates apply?",
        "What are the growth projections?",
        "What discount tiers are available?",
        "How do the features compare?",
        "What are the financial figures?",
        "What's the attendance record?",
        "How is the grading rubric structured?",
    ]
    for q in general_no_table_cue:
        instances.append(_hn(q, "General", "tbl_gen"))

    # ── Tricky: words "page"/"section"/"table" in non-lookup context ──

    tricky = [
        # Contains "section" but IS actually asking about a named section
        ("Does this contract have a section on liability?", "Section"),
        ("Is there a section about data privacy?", "Section"),
        ("Which section covers employee benefits?", "Section"),
        ("Where is the section on compliance?", "Section"),
        # Contains "page" but not a page lookup
        ("How many pages is this document?", "General"),
        ("What's on the title page?", "Page"),
        ("Is this a single-page summary?", "General"),
        ("The page numbers seem wrong.", "General"),
        # Contains "table" but not a table lookup
        ("The table of contents lists five chapters.", "General"),
        ("Is there a table of contents?", "General"),
        ("Where is the table of figures?", "General"),
    ]
    for q, label in tricky:
        instances.append(_hn(q, label, "tricky"))

    return instances


# ════════════════════════════════════════════════════════════════
# Dataset Assembly
# ════════════════════════════════════════════════════════════════

def _deduplicate(instances: list[QueryInstance]) -> list[QueryInstance]:
    """Remove exact duplicate queries, keeping the first occurrence."""
    seen: set[str] = set()
    deduped: list[QueryInstance] = []
    for inst in instances:
        key = inst.query.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(inst)
    return deduped


def _balance_classes(
    instances: list[QueryInstance], target: int = TARGET_PER_CLASS,
) -> list[QueryInstance]:
    """
    Balance classes to hit the target count per class.
    Downsamples overrepresented classes, warns if underrepresented.
    """
    by_class: dict[str, list[QueryInstance]] = defaultdict(list)
    for inst in instances:
        by_class[inst.label].append(inst)

    balanced: list[QueryInstance] = []
    for label in LABEL_TOOL_MAP:
        class_instances = by_class[label]
        if len(class_instances) > target:
            # Keep all hard negatives, downsample the rest
            hard = [i for i in class_instances if i.is_hard_negative]
            normal = [i for i in class_instances if not i.is_hard_negative]
            needed_normal = target - len(hard)
            if needed_normal > 0:
                normal = random.sample(normal, min(needed_normal, len(normal)))
            balanced.extend(hard)
            balanced.extend(normal)
        elif len(class_instances) < target:
            print(f"  ⚠ {label}: only {len(class_instances)} instances (target: {target})")
            balanced.extend(class_instances)
        else:
            balanced.extend(class_instances)

    return balanced


def _grouped_stratified_split(
    instances: list[QueryInstance],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    # test_ratio is implicit: 1 - train - val
) -> tuple[list[QueryInstance], list[QueryInstance], list[QueryInstance]]:
    """
    Grouped stratified split: all instances sharing a base_instance_id
    go into the same split. Stratified by label.
    """
    # Group by base_instance_id
    groups: dict[str, list[QueryInstance]] = defaultdict(list)
    for inst in instances:
        groups[inst.base_instance_id].append(inst)

    # For each group, determine its label (majority vote, usually unanimous)
    group_labels: dict[str, str] = {}
    for gid, ginsts in groups.items():
        labels = Counter(i.label for i in ginsts)
        group_labels[gid] = labels.most_common(1)[0][0]

    # Stratify: group the group-IDs by label
    label_groups: dict[str, list[str]] = defaultdict(list)
    for gid, label in group_labels.items():
        label_groups[label].append(gid)

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()

    for label in LABEL_TOOL_MAP:
        gids = label_groups.get(label, [])
        random.shuffle(gids)

        n = len(gids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # remaining goes to test

        train_ids.update(gids[:n_train])
        val_ids.update(gids[n_train:n_train + n_val])
        test_ids.update(gids[n_train + n_val:])

    train: list[QueryInstance] = []
    val: list[QueryInstance] = []
    test: list[QueryInstance] = []

    for inst in instances:
        if inst.base_instance_id in train_ids:
            train.append(inst)
        elif inst.base_instance_id in val_ids:
            val.append(inst)
        else:
            test.append(inst)

    return train, val, test


# ════════════════════════════════════════════════════════════════
# Output
# ════════════════════════════════════════════════════════════════

def _to_rows(instances: list[QueryInstance]) -> list[dict]:
    """Convert QueryInstance list to list of dicts for CSV writing."""
    rows = []
    for i, inst in enumerate(instances):
        rows.append({
            "id": i,
            "base_instance_id": inst.base_instance_id,
            "query": inst.query,
            "label": inst.label,
            "tool": inst.tool,
            "template_id": inst.template_id,
            "generation_method": inst.generation_method,
            "is_hard_negative": inst.is_hard_negative,
        })
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ Wrote {len(rows)} rows to {path}")


def _renumber_ids(rows: list[dict]) -> list[dict]:
    """Renumber the 'id' field sequentially."""
    for i, row in enumerate(rows):
        row["id"] = i
    return rows


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Query-Intent Dataset Generator")
    print("=" * 60)

    # 1. Generate template-based queries per class
    print("\n1. Generating template-based queries...")
    page_qs = generate_page_queries()
    section_qs = generate_section_queries()
    table_qs = generate_table_queries()
    general_qs = generate_general_queries()
    print(f"   Page: {len(page_qs)}, Section: {len(section_qs)}, "
          f"Table: {len(table_qs)}, General: {len(general_qs)}")

    # 2. Generate hard negatives
    print("\n2. Generating hard negatives...")
    hard_negs = generate_hard_negatives()
    hn_counts = Counter(i.label for i in hard_negs)
    print(f"   Hard negatives: {dict(hn_counts)} (total: {len(hard_negs)})")

    # 3. Combine all
    all_instances = page_qs + section_qs + table_qs + general_qs + hard_negs

    # 4. Deduplicate
    print("\n3. Deduplicating...")
    before = len(all_instances)
    all_instances = _deduplicate(all_instances)
    print(f"   {before} → {len(all_instances)} (removed {before - len(all_instances)} dupes)")

    # 5. Balance to target
    print("\n4. Balancing classes...")
    all_instances = _balance_classes(all_instances)
    final_counts = Counter(i.label for i in all_instances)
    print(f"   Final counts: {dict(final_counts)} (total: {len(all_instances)})")

    # 6. Write raw.csv
    print("\n5. Writing raw.csv...")
    raw_rows = _renumber_ids(_to_rows(all_instances))
    _write_csv(raw_rows, OUTPUT_DIR / "raw.csv")

    # 7. Grouped stratified split
    print("\n6. Performing grouped stratified split (70/15/15)...")
    train, val, test = _grouped_stratified_split(all_instances)

    train_counts = Counter(i.label for i in train)
    val_counts = Counter(i.label for i in val)
    test_counts = Counter(i.label for i in test)
    print(f"   Train: {len(train)} {dict(train_counts)}")
    print(f"   Val:   {len(val)} {dict(val_counts)}")
    print(f"   Test:  {len(test)} {dict(test_counts)}")

    # Check for group leakage
    train_gids = {i.base_instance_id for i in train}
    val_gids = {i.base_instance_id for i in val}
    test_gids = {i.base_instance_id for i in test}
    assert train_gids.isdisjoint(val_gids), "Leakage: train ∩ val"
    assert train_gids.isdisjoint(test_gids), "Leakage: train ∩ test"
    assert val_gids.isdisjoint(test_gids), "Leakage: val ∩ test"
    print("   ✓ No group leakage across splits")

    # 8. Write split CSVs
    print("\n7. Writing split CSVs...")
    _write_csv(_renumber_ids(_to_rows(train)), OUTPUT_DIR / "train.csv")
    _write_csv(_renumber_ids(_to_rows(val)), OUTPUT_DIR / "val.csv")
    _write_csv(_renumber_ids(_to_rows(test)), OUTPUT_DIR / "test.csv")

    # 9. Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label in LABEL_TOOL_MAP:
        total = sum(1 for i in all_instances if i.label == label)
        hn = sum(1 for i in all_instances if i.label == label and i.is_hard_negative)
        print(f"  {label:10s}: {total:4d} total, {hn:3d} hard negatives ({100*hn/max(total,1):.1f}%)")
    print(f"\n  Total: {len(all_instances)}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"  Seed:  {SEED}")
    print("=" * 60)


if __name__ == "__main__":
    main()
