#!/usr/bin/env python3
"""
Dataset Validation Script for Query-Intent Dataset.

Quality checks:
  1. Class balance within ±5% of target (400)
  2. No exact duplicate query strings across splits
  3. No exact duplicate query strings within splits
  4. Query length distribution (word count histogram)
  5. Hard-negative rate per class
  6. Group leakage check (base_instance_id appears in exactly one split)
  7. Schema validation (all columns present, no nulls)
  8. Sample queries per class per split

Usage:
  python data/query_intent/validate_dataset.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent
TARGET_PER_CLASS = 400
TOLERANCE = 0.05  # ±5%
EXPECTED_COLUMNS = [
    "id", "base_instance_id", "query", "label", "tool",
    "template_id", "generation_method", "is_hard_negative",
]
VALID_LABELS = {"Page", "Section", "Table", "General"}
VALID_TOOLS = {"PageRetriever", "SectionRetriever", "TableExtractor", "BM25Search"}


def load_csv(path: Path) -> list[dict]:
    """Load a CSV file into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_schema(rows: list[dict], name: str) -> bool:
    """Validate that all required columns are present and non-null."""
    ok = True
    if not rows:
        print(f"  ✗ {name}: empty file!")
        return False

    cols = set(rows[0].keys())
    missing = set(EXPECTED_COLUMNS) - cols
    if missing:
        print(f"  ✗ {name}: missing columns: {missing}")
        ok = False

    for i, row in enumerate(rows):
        for col in EXPECTED_COLUMNS:
            if col not in row or row[col] is None or row[col].strip() == "":
                if col == "is_hard_negative" and row.get(col) in ("True", "False", "true", "false", "0", "1"):
                    continue
                print(f"  ✗ {name}: row {i} has null/empty '{col}'")
                ok = False
                break
        # Validate label values
        if row.get("label") not in VALID_LABELS:
            print(f"  ✗ {name}: row {i} has invalid label: {row.get('label')}")
            ok = False
        if row.get("tool") not in VALID_TOOLS:
            print(f"  ✗ {name}: row {i} has invalid tool: {row.get('tool')}")
            ok = False

    if ok:
        print(f"  ✓ {name}: schema valid ({len(rows)} rows)")
    return ok


def check_class_balance(rows: list[dict], name: str) -> bool:
    """Check class balance within ±5% of target."""
    counts = Counter(r["label"] for r in rows)
    ok = True
    total = len(rows)

    # For splits, check proportional balance
    if name == "raw":
        for label in VALID_LABELS:
            count = counts.get(label, 0)
            lo = int(TARGET_PER_CLASS * (1 - TOLERANCE))
            hi = int(TARGET_PER_CLASS * (1 + TOLERANCE))
            status = "✓" if lo <= count <= hi else "✗"
            if status == "✗":
                ok = False
            print(f"  {status} {name}/{label}: {count} (target: {lo}–{hi})")
    else:
        expected_per = total / len(VALID_LABELS)
        for label in VALID_LABELS:
            count = counts.get(label, 0)
            lo = int(expected_per * (1 - TOLERANCE * 2))
            hi = int(expected_per * (1 + TOLERANCE * 2))
            status = "✓" if lo <= count <= hi else "⚠"
            print(f"  {status} {name}/{label}: {count} (expected ~{expected_per:.0f})")

    return ok


def check_duplicates_across_splits(
    train: list[dict], val: list[dict], test: list[dict],
) -> bool:
    """Check for duplicate queries across splits."""
    ok = True
    train_qs = {r["query"].strip().lower() for r in train}
    val_qs = {r["query"].strip().lower() for r in val}
    test_qs = {r["query"].strip().lower() for r in test}

    tv = train_qs & val_qs
    tt = train_qs & test_qs
    vt = val_qs & test_qs

    if tv:
        print(f"  ✗ {len(tv)} duplicate queries in train ∩ val")
        for q in list(tv)[:3]:
            print(f"    e.g.: \"{q[:60]}\"")
        ok = False
    else:
        print("  ✓ No duplicates across train ∩ val")

    if tt:
        print(f"  ✗ {len(tt)} duplicate queries in train ∩ test")
        for q in list(tt)[:3]:
            print(f"    e.g.: \"{q[:60]}\"")
        ok = False
    else:
        print("  ✓ No duplicates across train ∩ test")

    if vt:
        print(f"  ✗ {len(vt)} duplicate queries in val ∩ test")
        for q in list(vt)[:3]:
            print(f"    e.g.: \"{q[:60]}\"")
        ok = False
    else:
        print("  ✓ No duplicates across val ∩ test")

    return ok


def check_duplicates_within(rows: list[dict], name: str) -> bool:
    """Check for duplicate queries within a single split."""
    queries = [r["query"].strip().lower() for r in rows]
    counts = Counter(queries)
    dupes = {q: c for q, c in counts.items() if c > 1}
    if dupes:
        print(f"  ⚠ {name}: {len(dupes)} duplicate queries within split")
        for q in list(dupes.keys())[:3]:
            print(f"    \"{q[:60]}\" × {dupes[q]}")
        return False
    else:
        print(f"  ✓ {name}: no within-split duplicates")
        return True


def check_query_lengths(rows: list[dict], name: str) -> None:
    """Report query length distribution (word count)."""
    lengths = [len(r["query"].split()) for r in rows]
    if not lengths:
        return

    min_l = min(lengths)
    max_l = max(lengths)
    avg_l = sum(lengths) / len(lengths)

    # Bucket into 5-word bins
    buckets: dict[str, int] = defaultdict(int)
    for l in lengths:
        bucket = f"{(l // 5) * 5}-{(l // 5) * 5 + 4}"
        buckets[bucket] += 1

    print(f"  {name}: min={min_l}, max={max_l}, avg={avg_l:.1f} words")

    # Check if >80% in one bucket
    max_bucket_count = max(buckets.values())
    if max_bucket_count / len(lengths) > 0.80:
        print(f"  ⚠ {name}: >80% of queries fall in a single 5-word bucket!")
    else:
        print(f"  ✓ {name}: query lengths well distributed across buckets")

    # Print top buckets
    for bucket in sorted(buckets.keys()):
        pct = 100 * buckets[bucket] / len(lengths)
        bar = "█" * int(pct / 2)
        print(f"    {bucket:>7s} words: {buckets[bucket]:4d} ({pct:5.1f}%) {bar}")


def check_hard_negative_rate(rows: list[dict], name: str) -> None:
    """Report hard-negative rate per class."""
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hard": 0})
    for r in rows:
        label = r["label"]
        by_class[label]["total"] += 1
        if r["is_hard_negative"] in ("True", "true", "1"):
            by_class[label]["hard"] += 1

    for label in sorted(by_class.keys()):
        total = by_class[label]["total"]
        hard = by_class[label]["hard"]
        pct = 100 * hard / max(total, 1)
        status = "✓" if 5 <= pct <= 25 else "⚠"
        print(f"  {status} {name}/{label}: {hard}/{total} ({pct:.1f}%) hard negatives")


def check_group_leakage(
    train: list[dict], val: list[dict], test: list[dict],
) -> bool:
    """Verify base_instance_ids don't leak across splits."""
    train_gids = {r["base_instance_id"] for r in train}
    val_gids = {r["base_instance_id"] for r in val}
    test_gids = {r["base_instance_id"] for r in test}

    tv = train_gids & val_gids
    tt = train_gids & test_gids
    vt = val_gids & test_gids

    ok = True
    if tv:
        print(f"  ✗ GROUP LEAKAGE: {len(tv)} base_instance_ids in train ∩ val")
        ok = False
    if tt:
        print(f"  ✗ GROUP LEAKAGE: {len(tt)} base_instance_ids in train ∩ test")
        ok = False
    if vt:
        print(f"  ✗ GROUP LEAKAGE: {len(vt)} base_instance_ids in val ∩ test")
        ok = False

    if ok:
        print("  ✓ No group leakage (all base_instance_ids in exactly one split)")
    return ok


def show_samples(rows: list[dict], name: str, n: int = 3) -> None:
    """Show n random examples per class."""
    import random
    random.seed(42)

    by_class: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r["query"])

    print(f"\n  Samples from {name}:")
    for label in sorted(by_class.keys()):
        queries = by_class[label]
        samples = random.sample(queries, min(n, len(queries)))
        print(f"    {label}:")
        for q in samples:
            print(f"      → \"{q}\"")


def main():
    print("=" * 60)
    print("Query-Intent Dataset Validation")
    print("=" * 60)

    # Load data
    raw = load_csv(DATA_DIR / "raw.csv")
    train = load_csv(DATA_DIR / "train.csv")
    val = load_csv(DATA_DIR / "val.csv")
    test = load_csv(DATA_DIR / "test.csv")

    all_pass = True

    # 1. Schema validation
    print("\n── 1. Schema Validation ──")
    all_pass &= check_schema(raw, "raw")
    all_pass &= check_schema(train, "train")
    all_pass &= check_schema(val, "val")
    all_pass &= check_schema(test, "test")

    # 2. Class balance
    print("\n── 2. Class Balance ──")
    all_pass &= check_class_balance(raw, "raw")
    check_class_balance(train, "train")
    check_class_balance(val, "val")
    check_class_balance(test, "test")

    # 3. Cross-split duplicates
    print("\n── 3. Cross-Split Duplicates ──")
    all_pass &= check_duplicates_across_splits(train, val, test)

    # 4. Within-split duplicates
    print("\n── 4. Within-Split Duplicates ──")
    check_duplicates_within(train, "train")
    check_duplicates_within(val, "val")
    check_duplicates_within(test, "test")

    # 5. Query length distribution
    print("\n── 5. Query Length Distribution ──")
    check_query_lengths(raw, "raw")

    # 6. Hard-negative rate
    print("\n── 6. Hard-Negative Rate ──")
    check_hard_negative_rate(raw, "raw")
    check_hard_negative_rate(train, "train")

    # 7. Group leakage
    print("\n── 7. Group Leakage Check ──")
    all_pass &= check_group_leakage(train, val, test)

    # 8. Samples
    print("\n── 8. Sample Queries ──")
    show_samples(train, "train", n=3)
    show_samples(test, "test", n=3)

    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ SOME CHECKS FAILED — review above")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
