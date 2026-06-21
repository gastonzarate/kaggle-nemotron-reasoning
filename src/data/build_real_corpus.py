"""Build the REAL-prompt training corpus (the Progress Prize winner's recipe).

For every row of the real train.csv:
  1. Classify the family from the prompt template.
  2. Parse the prompt and solve it with our verified programmatic solvers.
  3. rule_found gate: emit our reasoning ONLY if the solver's answer matches the
     real gold under the OFFICIAL metric (rel 1% for floats, case-insensitive
     exact otherwise). The emitted `answer` is the solver-derived value, so the
     SFT target (reasoning + re-boxed answer) is coherent.
  4. Fallback: rows the solver can't crack take the cot-tong teacher CoT
     (the corpus behind the proven 0.84 run), `answer` = real gold.
  5. Rows with neither are skipped (no noise).

Output CSV: id, prompt, answer, type, generated_cot, source(solver|teacher).

Usage:
    uv run python -m src.data.build_real_corpus \
        --train-csv data/raw/train.csv \
        --teacher-csv data/raw/cot-tong/problem_ids_matched.csv \
        --out-csv data/synthetic/real_corpus_v1/train.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from src.data import programmatic_cot as pc
from src.data import prompt_parsers as pp
from src.data.transformation_extended import solve_transformation_extended

FLOAT_FAMILIES = {"gravity", "unit_conversion"}

# Wall-clock cap per extended-transformation solve (the symbol CSP can blow up).
EXT_TIMEOUT_S = 8


def _solve_transformation_ext(prompt: str, gold: str) -> tuple[str, str] | None:
    """Extended transformation solve (gold-conditioned) under a hard timeout."""
    import signal

    parsed = pp.parse_transformation(prompt)
    if not parsed:
        return None

    class _TO(Exception):
        pass

    def _hdl(*_):
        raise _TO()

    old = signal.signal(signal.SIGALRM, _hdl)
    signal.alarm(EXT_TIMEOUT_S)
    try:
        out = solve_transformation_extended(parsed["pairs"], parsed["query"], gold=gold)
    except (_TO, Exception):
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    if out is None:
        return None
    ans, body = out
    cot = (
        "<think>\n"
        "<r>family=transformation; plan: recover the hidden per-operator rule from its "
        "examples, then apply it to the query</r>\n<c>\n" + body + "\n</c>\n</think>\n"
        f"\\boxed{{{ans}}}"
    )
    return ans, cot


def classify(prompt: str) -> str | None:
    if "gravitational constant" in prompt:
        return "gravity"
    if "unit conversion" in prompt:
        return "unit_conversion"
    if "numeral system" in prompt:
        return "numeral"
    if "encryption rules" in prompt:
        return "cipher"
    if "bit manipulation" in prompt:
        return "bit_manipulation"
    if "transformation rules is applied to equations" in prompt:
        return "transformation"
    return None


def official_match(predicted: str, gold: str) -> bool:
    p, g = predicted.strip(), gold.strip()
    try:
        return math.isclose(float(p), float(g), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        return p.lower() == g.lower()


def solve(family: str, prompt: str) -> tuple[str, str] | None:
    """Run the family's parser + CoT generator. Returns (predicted, cot) or None."""
    try:
        if family == "gravity":
            parsed = pp.parse_gravity(prompt)
            return pc.gravity_cot(**parsed, commit_early=True) if parsed else None
        if family == "unit_conversion":
            parsed = pp.parse_unit_conversion(prompt)
            return pc.unit_conversion_cot(**parsed) if parsed else None
        if family == "numeral":
            parsed = pp.parse_numeral(prompt)
            return pc.numeral_cot(**parsed) if parsed else None
        if family == "cipher":
            parsed = pp.parse_cipher(prompt)
            return pc.cipher_cot(**parsed) if parsed else None
        if family == "bit_manipulation":
            parsed = pp.parse_bit_manipulation(prompt)
            return pc.bit_manipulation_cot(**parsed, commit_early=True) if parsed else None
        if family == "transformation":
            parsed = pp.parse_transformation(prompt)
            return pc.arithmetic_transformation_cot(**parsed) if parsed else None
    except Exception:
        return None
    return None


def build(
    train_csv: Path, teacher_csv: Path | None, out_csv: Path,
    upsample_hard: bool = False,
) -> dict:
    rows_in = list(csv.DictReader(open(train_csv, encoding="utf-8-sig")))

    teacher: dict[str, dict] = {}
    if teacher_csv and teacher_csv.exists():
        for r in csv.DictReader(open(teacher_csv, encoding="utf-8-sig")):
            rid = (r.get("id") or r.get("﻿id") or "").strip()
            if rid:
                teacher[rid] = r

    out_rows: list[dict] = []
    stats_solved: Counter = Counter()
    stats_teacher: Counter = Counter()
    stats_skipped: Counter = Counter()
    stats_total: Counter = Counter()

    for r in rows_in:
        rid = (r.get("id") or "").strip()
        prompt = r["prompt"]
        gold = str(r["answer"]).strip()
        fam = classify(prompt)
        if fam is None:
            stats_skipped["UNCLASSIFIED"] += 1
            continue
        stats_total[fam] += 1

        result = solve(fam, prompt)
        if result is not None:
            predicted, cot = result
            if official_match(predicted, gold):
                out_rows.append({
                    "id": rid,
                    "prompt": prompt,
                    "answer": predicted.strip(),
                    "type": fam,
                    "generated_cot": cot,
                    "source": "solver",
                })
                stats_solved[fam] += 1
                continue

        # Extended transformation solver (gold-conditioned): broader op space +
        # symbol→digit CSP. Only fires for rows the base solver missed.
        if fam == "transformation":
            ext = _solve_transformation_ext(prompt, gold)
            if ext is not None and official_match(ext[0], gold):
                out_rows.append({
                    "id": rid,
                    "prompt": prompt,
                    "answer": ext[0].strip(),
                    "type": fam,
                    "generated_cot": ext[1],
                    "source": "solver_ext",
                })
                stats_solved[fam] += 1
                continue

        # Fallback: teacher CoT (cot-tong) keyed by the same real id.
        trow = teacher.get(rid)
        if trow and str(trow.get("generated_cot", "")).strip():
            out_rows.append({
                "id": rid,
                "prompt": prompt,
                "answer": gold,
                "type": fam,
                "generated_cot": trow["generated_cot"],
                "source": "teacher",
            })
            stats_teacher[fam] += 1
        else:
            stats_skipped[fam] += 1

    if upsample_hard:
        # Hard families see the eval's lost points: repeat them once (local
        # 2-epoch effect — "repetition beats scaling" for long-CoT SFT).
        extra = []
        for r in out_rows:
            if r["type"] in ("bit_manipulation", "transformation"):
                d = dict(r)
                d["id"] = r["id"] + "-r2"
                extra.append(d)
        out_rows.extend(extra)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "type", "generated_cot", "source"])
        w.writeheader()
        w.writerows(out_rows)

    stats = {
        "total_rows_out": len(out_rows),
        "per_family": {
            fam: {
                "total": stats_total[fam],
                "solver": stats_solved[fam],
                "teacher": stats_teacher[fam],
                "skipped": stats_skipped[fam],
                "solver_pct": round(100 * stats_solved[fam] / stats_total[fam], 1) if stats_total[fam] else 0,
            }
            for fam in sorted(stats_total)
        },
        "unclassified": stats_skipped["UNCLASSIFIED"],
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    ap.add_argument("--teacher-csv", type=Path, default=Path("data/raw/cot-tong/problem_ids_matched.csv"))
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--upsample-hard", action="store_true")
    args = ap.parse_args()
    stats = build(args.train_csv, args.teacher_csv, args.out_csv, upsample_hard=args.upsample_hard)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
