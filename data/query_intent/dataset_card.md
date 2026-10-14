# Query-Intent Dataset Card

## Overview

A synthetic, labelled dataset for training a **query-intent classifier** to route PDF document queries to the correct retrieval tool. Part of the ML-RAG extension to the PDF Agentic RAG system.

| Property | Value |
|----------|-------|
| **Total queries** | 1,600 |
| **Classes** | 4 (Page, Section, Table, General) |
| **Per-class count** | 400 |
| **Train / Val / Test** | ~70% / ~15% / ~15% |
| **Splitting method** | Grouped stratified on `base_instance_id` |
| **Random seed** | 42 |
| **Generation method** | Template × slot fill + phrasing variants |
| **Hard test set** | 100 hand-written queries (separate file) |

---

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique row ID (per split) |
| `base_instance_id` | str | Groups a template fill + its variants — used for grouped splitting to prevent leakage |
| `query` | str | The query text |
| `label` | str | Intent class: `Page`, `Section`, `Table`, `General` |
| `tool` | str | Target tool: `PageRetriever`, `SectionRetriever`, `TableExtractor`, `BM25Search` |
| `template_id` | str | Which template generated this query (e.g., `page_01`, `sec_15_var`) |
| `generation_method` | str | `template` (all in this version) |
| `is_hard_negative` | bool | Whether this is a deliberately ambiguous/confusable example |

---

## Class Definitions & Tool Mapping

| Class | Tool | Description |
|-------|------|-------------|
| **Page** | `PageRetriever` | Query requests content from a specific page number or page range |
| **Section** | `SectionRetriever` | Query references a named section, heading, or chapter of the document |
| **Table** | `TableExtractor` | Query asks for tabular/structured data with an explicit table cue |
| **General** | `BM25Search` | Keyword/factual search — no specific page, section header, or table request |

---

## Labelling Rules (Hard Negatives)

### Rule 1: Page vs. Section
If a query contains **both a concrete page number AND a section name**, it is labelled **Page**. The page number is the more actionable, precise retrieval instruction.

> Example: _"What's on page 5 about the methodology?"_ → **Page**

### Rule 2: Section vs. General
If the query **names a specific document section/heading** (e.g., "the Refunds section", "the Methodology"), label = **Section**. If it asks about a **topic** without naming a section header, label = **General**.

> _"What does the Refunds section say?"_ → **Section**
> _"What does the document say about refunds?"_ → **General**

### Rule 3: Table vs. General
The query must have an **explicit table/extract cue** to be labelled **Table**. Words like "table", "extract", "tabulated", "in table form" qualify. Simply mentioning numbers, data, or a topic that might be in a table → **General**.

> _"Extract the quarterly revenue table."_ → **Table**
> _"What are the quarterly revenue figures?"_ → **General**

### Rule 4: Context Words
Words "page", "section", or "table" appearing in a non-lookup context are judged individually:
- _"Does this contract have a section on liability?"_ → **Section** (IS asking about a named section)
- _"How many pages is this document?"_ → **General** (meta-question, not a page lookup)
- _"The table of contents lists five chapters."_ → **General** (table of contents ≠ data table)

---

## Split Statistics

### raw.csv (1,600 total)

| Class | Count | Hard Negatives | HN Rate |
|-------|------:|---------------:|--------:|
| Page | 400 | 36 | 9.0% |
| Section | 400 | 39 | 9.8% |
| Table | 400 | 35 | 8.8% |
| General | 400 | 46 | 11.5% |

### Split Counts

| Split | Total | Page | Section | Table | General |
|-------|------:|-----:|--------:|------:|--------:|
| Train | 1,119 | 281 | 279 | 279 | 280 |
| Val | 234 | 57 | 60 | 58 | 59 |
| Test | 247 | 62 | 61 | 63 | 61 |

---

## Example Queries Per Class

### Page
- _"Show me page 42 of the report"_
- _"What does it say on page 3?"_
- _"Retrieve pages 10 to 12"_
- _"What's written on the twelfth page?"_

### Section
- _"Summarize the Conclusion"_
- _"What are the key points in Methodology?"_
- _"Tell me about the Risk Factors section"_
- _"Pull up the FAQ section for me"_

### Table
- _"Extract the quarterly revenue table"_
- _"Show me the feature comparison table"_
- _"Give me the benchmark results table"_
- _"List the rows in the pricing plan table"_

### General
- _"What is the refund policy?"_
- _"How does data privacy work according to this document?"_
- _"What are the requirements for compliance?"_
- _"Who is responsible for quality assurance?"_

---

## Generation Method

1. **20 templates per class** with `{slot}` placeholders (e.g., `{page}`, `{section}`, `{topic}`)
2. **Slot filling**: 18 slot values sampled per template from curated banks (page numbers 1-300, 35 section names, 30 table topics, 25 general topics)
3. **Phrasing variants**: 50 random instances per class receive punctuation/politeness variations (question marks, "please" suffix, "Could you" rewrites)
4. **Hard negatives**: 116-156 hand-crafted deliberately ambiguous examples targeting known confusion pairs
5. **Deduplication**: exact string dedup (case-insensitive)
6. **Balancing**: downsample to 400/class, preserving all hard negatives

---

## Splitting: Preventing Leakage

All queries sharing the same `base_instance_id` (a template × slot-value key) are placed in the same split. This prevents near-duplicate queries (same template, same slot value, different punctuation) from leaking across train/val/test and inflating metrics.

Split method: **manual grouped stratified split** — group by `base_instance_id`, stratify by `label`, 70/15/15 ratio.

---

## Known Limitations

1. **Synthetic phrasing**: All queries are template-generated, which may not fully capture real user query patterns (typos, multi-intent queries, code-switching, domain-specific jargon).
2. **Limited vocabulary**: Slot value banks are finite (~25-35 values per class), so the model may overfit to these specific topics/section names.
3. **English only**: No multilingual coverage.
4. **Single-intent**: Each query maps to exactly one tool; real users may ask multi-tool questions.
5. **Hard-negative rate (~9-12%)** is below the 15-20% target, but covers the key confusion pairs (Section/General, Page/Section, Table/General) adequately.
6. **Mitigation**: The confidence-threshold fallback to the LLM planner ensures routing mistakes don't break end-to-end correctness.

---

## Files

| File | Description |
|------|-------------|
| `raw.csv` | Full dataset (1,600 rows, pre-split) |
| `train.csv` | Training split (~70%) |
| `val.csv` | Validation split (~15%) |
| `test.csv` | Test split (~15%) |
| `hard_test.csv` | 100 hand-written held-out queries (not template-derived) |
| `generate_dataset.py` | Reproducible generation script |
| `validate_dataset.py` | Quality check script |
| `dataset_card.md` | This file |
