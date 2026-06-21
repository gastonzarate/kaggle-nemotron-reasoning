"""End-to-end integration test on the actual produced dataset.

Regenerates `data/synthetic/structured_cot_v2/train.csv` via rewriter_v2 and
asserts invariants on a sample of rows. If this test passes, the dataset is
SAFE to ship to training; if it fails, we MUST NOT push.
"""

from __future__ import annotations

import csv
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from src.data.rewriter_v2 import rewrite_v2

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "raw" / "cot-tong" / "problem_ids_matched.csv"
OUT_CSV = ROOT / "data" / "synthetic" / "structured_cot_v2" / "train.csv"


@pytest.fixture(scope="module")
def regenerated_dataset():
    """Regenerate the dataset once for the whole test module."""
    if not IN_CSV.exists():
        pytest.skip(f"CoT-tong dataset not available at {IN_CSV}")
    stats = rewrite_v2(IN_CSV, OUT_CSV)
    return stats


@pytest.fixture(scope="module")
def all_rows(regenerated_dataset):
    rows: list[dict] = []
    with open(OUT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _extract_boxed(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not m:
        return ""
    return m[-1].strip()


def _verify_official(stored: str, predicted: str) -> bool:
    """Mirror the Kaggle official metric exactly."""
    stored, predicted = stored.strip(), predicted.strip()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Envelope: every single row must conform
# ─────────────────────────────────────────────────────────────────────────────


def test_every_row_starts_with_think(all_rows):
    bad = [r["id"] for r in all_rows if not r["generated_cot"].startswith("<think>")]
    assert not bad, f"{len(bad)} rows don't start with <think>: first 5 = {bad[:5]}"


def test_no_row_has_double_think_close(all_rows):
    """The v0.2.0 catastrophic bug: rows with two </think> tags."""
    bad = [r["id"] for r in all_rows if r["generated_cot"].count("</think>") != 1]
    assert not bad, f"{len(bad)} rows have wrong </think> count; first 5 = {bad[:5]}"


def test_no_row_has_double_think_open(all_rows):
    bad = [r["id"] for r in all_rows if r["generated_cot"].count("<think>") != 1]
    assert not bad, f"{len(bad)} rows have wrong <think> count; first 5 = {bad[:5]}"


def test_every_row_has_exactly_one_boxed(all_rows):
    bad = []
    for r in all_rows:
        boxes = re.findall(r"\\boxed\{[^}]*\}", r["generated_cot"])
        if len(boxes) != 1:
            bad.append((r["id"], len(boxes)))
    assert not bad, f"{len(bad)} rows have ≠1 \\boxed; first 5 = {bad[:5]}"


def test_no_row_has_empty_boxed(all_rows):
    bad = [r["id"] for r in all_rows if not _extract_boxed(r["generated_cot"])]
    assert not bad, f"{len(bad)} rows have empty \\boxed{{}}; first 5 = {bad[:5]}"


def test_every_row_has_router(all_rows):
    bad = [r["id"] for r in all_rows if "<r>" not in r["generated_cot"] or "</r>" not in r["generated_cot"]]
    assert not bad, f"{len(bad)} rows missing <r>...</r>; first 5 = {bad[:5]}"


def test_every_row_has_at_least_one_subtag(all_rows):
    bad = []
    for r in all_rows:
        cot = r["generated_cot"]
        if not any(f"<{t}>" in cot for t in ("m", "c", "l")):
            bad.append(r["id"])
    assert not bad, f"{len(bad)} rows missing subtag; first 5 = {bad[:5]}"


def test_every_row_balances_its_subtags(all_rows):
    bad = []
    for r in all_rows:
        cot = r["generated_cot"]
        for tag in ("m", "c", "l", "r"):
            if cot.count(f"<{tag}>") != cot.count(f"</{tag}>"):
                bad.append((r["id"], tag, cot.count(f"<{tag}>"), cot.count(f"</{tag}>")))
                break
    assert not bad, f"{len(bad)} rows unbalanced; first 3 = {bad[:3]}"


def test_boxed_comes_after_think_close(all_rows):
    bad = []
    for r in all_rows:
        cot = r["generated_cot"]
        if cot.rfind("\\boxed{") < cot.rfind("</think>"):
            bad.append(r["id"])
    assert not bad, f"{len(bad)} rows have \\boxed BEFORE </think>; first 5 = {bad[:5]}"


# ─────────────────────────────────────────────────────────────────────────────
# Answer correctness: the boxed answer must match gold (using Kaggle's verifier)
# ─────────────────────────────────────────────────────────────────────────────


def test_every_row_boxed_matches_gold(all_rows):
    """The boxed answer in each row must match the gold answer with the official metric.

    This is the most important test: if the model trains on rows where boxed ≠ gold,
    it learns to produce WRONG answers. Catastrophic for the LB.
    """
    mismatches = []
    for r in all_rows:
        gold = r["answer"]
        boxed = _extract_boxed(r["generated_cot"])
        if not _verify_official(gold, boxed):
            mismatches.append((r["id"], r["type"], gold, boxed))
    if mismatches:
        # Per-family breakdown for diagnostics
        by_family: dict[str, int] = {}
        for _, fam, _, _ in mismatches:
            by_family[fam] = by_family.get(fam, 0) + 1
        msg = (
            f"{len(mismatches)}/{len(all_rows)} rows: boxed answer ≠ gold.\n"
            f"per-family: {sorted(by_family.items(), key=lambda x: -x[1])}\n"
            f"first 5: {mismatches[:5]}"
        )
        assert False, msg


# ─────────────────────────────────────────────────────────────────────────────
# Sanity: row counts and family coverage
# ─────────────────────────────────────────────────────────────────────────────


def test_row_count_matches_input(regenerated_dataset, all_rows):
    assert len(all_rows) == regenerated_dataset["input_rows"]
    assert len(all_rows) == regenerated_dataset["output_rows"]


def test_all_9_families_present(all_rows):
    expected = {
        "gravity", "unit_conversion", "numeral", "cipher", "bit_manipulation",
        "cryptarithm_deduce", "cryptarithm_guess",
        "equation_numeric_deduce", "equation_numeric_guess",
    }
    present = {r["type"] for r in all_rows}
    assert expected.issubset(present), f"missing families: {expected - present}"


# ─────────────────────────────────────────────────────────────────────────────
# CoT length sanity
# ─────────────────────────────────────────────────────────────────────────────


def test_cot_length_reasonable(all_rows):
    """Flag rows that are suspiciously short (no real reasoning) or
    catastrophically long (would consume the model's token budget)."""
    too_short = [r["id"] for r in all_rows if len(r["generated_cot"]) < 100]
    too_long = [r["id"] for r in all_rows if len(r["generated_cot"]) > 30000]
    assert not too_short, f"{len(too_short)} rows < 100 chars (no real reasoning)"
    assert not too_long, f"{len(too_long)} rows > 30000 chars (token budget risk)"


# ─────────────────────────────────────────────────────────────────────────────
# v3 dataset integration: same invariants on the fully-synthetic 9000-row file
# ─────────────────────────────────────────────────────────────────────────────


V3_CSV = ROOT / "data" / "synthetic" / "structured_cot_v3" / "train.csv"


@pytest.fixture(scope="module")
def v3_rows():
    if not V3_CSV.exists():
        pytest.skip(f"v3 dataset not yet built at {V3_CSV}")
    rows = []
    with open(V3_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def test_v3_has_9000_rows(v3_rows):
    assert len(v3_rows) == 9000


def test_v3_balanced_per_family(v3_rows):
    """6 families matching the real train.csv distribution, ~1/6 each."""
    from collections import Counter

    expected_families = {
        "gravity", "unit_conversion", "numeral",
        "cipher", "bit_manipulation", "transformation",
    }
    counts = Counter(r["type"] for r in v3_rows)
    assert set(counts) == expected_families, f"unexpected families: {set(counts)}"
    per = len(v3_rows) // len(expected_families)
    for fam, n in counts.items():
        assert n == per, f"family {fam} has {n} rows, expected {per}"


def test_v3_every_row_boxed_matches_gold(v3_rows):
    mismatches = []
    for r in v3_rows:
        gold = r["answer"]
        boxed = _extract_boxed(r["generated_cot"])
        if not _verify_official(gold, boxed):
            mismatches.append((r["id"], r["type"], gold, boxed))
    assert not mismatches, f"v3: {len(mismatches)} rows with boxed≠gold; first 5: {mismatches[:5]}"


def test_v3_all_envelopes_clean(v3_rows):
    bad = []
    for r in v3_rows:
        cot = r["generated_cot"]
        if cot.count("<think>") != 1 or cot.count("</think>") != 1:
            bad.append((r["id"], "think"))
            continue
        boxes = re.findall(r"\\boxed\{[^}]*\}", cot)
        if len(boxes) != 1 or not _extract_boxed(cot):
            bad.append((r["id"], "boxed"))
    assert not bad, f"{len(bad)} v3 rows with envelope issues; first 5: {bad[:5]}"


def test_v3_cot_compact(v3_rows):
    """v3 CoTs should be much shorter than v0.2.0 (which had 3400+ char gravity rows)."""
    avg_len = sum(len(r["generated_cot"]) for r in v3_rows) / len(v3_rows)
    assert 200 < avg_len < 2000, f"avg CoT length {avg_len} out of expected range"
