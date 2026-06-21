"""Orchestrator that produces the v2 structured dataset.

For each row in the CoT-tong dataset:
  1. Parse prompt by family.
  2. Run the programmatic CoT generator.
  3. If the programmatic answer matches gold (within tolerance) → use the new CoT.
  4. Otherwise → fall back to the v1 heuristic wrapper (`rewriter.build_structured_cot`).

Writes the new train CSV plus a stats report.
"""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.data import prompt_parsers as pp
from src.data import programmatic_cot as pc
from src.data.rewriter import build_structured_cot

# Map type → (parser, generator). The generator takes the parsed dict and returns (answer, cot).
FAMILY_HANDLERS = {
    "gravity": (
        pp.parse_gravity,
        lambda parsed: pc.gravity_cot(parsed["pairs"], parsed["query_t"]),
    ),
    "unit_conversion": (
        pp.parse_unit_conversion,
        lambda parsed: pc.unit_conversion_cot(parsed["pairs"], parsed["query_x"]),
    ),
    "numeral": (
        pp.parse_numeral,
        lambda parsed: pc.numeral_cot(parsed["pairs"], parsed["query_n"]),
    ),
    "cipher": (
        pp.parse_cipher,
        lambda parsed: pc.cipher_cot(parsed["pairs"], parsed["query"]),
    ),
    "bit_manipulation": (
        pp.parse_bit_manipulation,
        lambda parsed: pc.bit_manipulation_cot(parsed["pairs"], parsed["query"]),
    ),
    "cryptarithm_deduce": (
        pp.parse_transformation,
        lambda parsed: pc.transformation_cot(parsed["pairs"], parsed["query"]),
    ),
    "cryptarithm_guess": (
        pp.parse_transformation,
        lambda parsed: pc.transformation_cot(parsed["pairs"], parsed["query"]),
    ),
    "equation_numeric_deduce": (
        pp.parse_transformation,
        lambda parsed: pc.transformation_cot(parsed["pairs"], parsed["query"]),
    ),
    "equation_numeric_guess": (
        pp.parse_transformation,
        lambda parsed: pc.transformation_cot(parsed["pairs"], parsed["query"]),
    ),
}


def _answers_match(predicted: str, gold: str) -> bool:
    """Match Kaggle's verifier: numeric within rel_tol=1e-2 OR string equality (lower)."""
    p, g = predicted.strip(), gold.strip()
    try:
        pn, gn = Decimal(p), Decimal(g)
        # Same logic as competition verifier: rel_tol 1e-2 or abs_tol 1e-5
        from decimal import Decimal as D
        denom = abs(gn) if gn != 0 else D(1)
        return abs(pn - gn) / denom < D("0.01") or abs(pn - gn) < D("0.00001")
    except (InvalidOperation, ValueError):
        return p.lower() == g.lower()


def rewrite_v2(in_csv: Path, out_csv: Path) -> dict:
    rows_in: list[dict] = []
    with open(in_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            id_key = next((k for k in row if k.lstrip("﻿") == "id"), None)
            if id_key and id_key != "id":
                row["id"] = row.pop(id_key)
            rows_in.append(row)

    stats: dict = {
        "input_rows": len(rows_in),
        "per_family": {},
        "programmatic_used": 0,
        "programmatic_wrong_answer": 0,
        "fallback_to_heuristic": 0,
        "parser_failed": 0,
        "exception": 0,
    }

    rows_out: list[dict] = []
    for row in rows_in:
        family = (row.get("type") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        gold = (row.get("answer") or "").strip()
        original_cot = (row.get("generated_cot") or "").strip()

        fam_stats = stats["per_family"].setdefault(
            family,
            {"total": 0, "programmatic_ok": 0, "wrong_answer": 0, "parser_failed": 0, "fallback": 0, "exception": 0},
        )
        fam_stats["total"] += 1

        new_cot: str | None = None
        handler = FAMILY_HANDLERS.get(family)
        if handler is not None:
            parser, generator = handler
            try:
                parsed = parser(prompt)
                if parsed is None:
                    fam_stats["parser_failed"] += 1
                    stats["parser_failed"] += 1
                else:
                    result = generator(parsed)
                    if result is None:
                        # transformation_cot can return None if no rule fits
                        fam_stats["fallback"] += 1
                        stats["fallback_to_heuristic"] += 1
                    else:
                        predicted, candidate_cot = result
                        if _answers_match(predicted, gold):
                            new_cot = candidate_cot
                            fam_stats["programmatic_ok"] += 1
                            stats["programmatic_used"] += 1
                        else:
                            fam_stats["wrong_answer"] += 1
                            stats["programmatic_wrong_answer"] += 1
            except Exception as e:  # noqa: BLE001
                fam_stats["exception"] += 1
                stats["exception"] += 1

        if new_cot is None:
            # Fallback: keep the v0.2.0 heuristic wrapping (so we never go backwards on coverage)
            new_cot = build_structured_cot(prompt, original_cot, family, gold)

        rows_out.append({
            "id": row.get("id", ""),
            "prompt": prompt,
            "answer": gold,
            "type": family,
            "generated_cot": new_cot,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "type", "generated_cot"])
        writer.writeheader()
        writer.writerows(rows_out)

    stats["output_rows"] = len(rows_out)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    stats = rewrite_v2(args.in_csv, args.out_csv)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
