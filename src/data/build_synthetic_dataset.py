"""Build the fully-synthetic v3 dataset: 1 000 rows per family × 9 families = 9 000 total.

For every row:
  1. Call a puzzle generator → (prompt, gold, parsed)
  2. Feed parsed into matching CoT generator → (predicted, structured_cot)
  3. Sanity check: predicted == gold (must hold by construction; assert)
  4. Emit row {id, prompt, answer=gold, type, generated_cot=structured_cot}

Output CSV is drop-in compatible with the training notebook.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable

from src.data import programmatic_cot as pc
from src.data import puzzle_generators as pg


# Only these two families are genuinely real-valued; the official metric scores
# them with a 1% relative tolerance. EVERY other family is a structured string
# (binary, Roman, ciphertext, char-transform) whose answer must match byte-exact.
FLOAT_FAMILIES = {"gravity", "unit_conversion"}


def _solves_puzzle(predicted: str, gold: str, family: str) -> bool:
    """Does `predicted` solve the puzzle?

    For float families we accept anything within the metric's 1% tolerance —
    the true hidden constant is not exactly recoverable from the rounded
    examples, so the value derivable from the prompt is the legitimate target.
    For all other families we require an EXACT match: using a numeric tolerance
    there was a latent bug — e.g. ``float("11110110")`` parses a binary string as
    a decimal, so two *different* 8-bit outputs that happen to be numerically
    close slipped through (this is exactly how 12 wrong-rule bit_manipulation
    rows reached v3).
    """
    p, g = predicted.strip(), gold.strip()
    if family in FLOAT_FAMILIES:
        try:
            return math.isclose(float(p), float(g), rel_tol=1e-2, abs_tol=1e-5)
        except (InvalidOperation, ValueError):
            return p.lower() == g.lower()
    return p == g


def _extract_last_boxed(text: str) -> str | None:
    matches = list(re.findall(r"\\boxed\{([^}]*)\}", text))
    return matches[-1] if matches else None


def _stable_id(prompt: str, idx: int) -> str:
    """8-char stable id from prompt hash + idx (mirrors train.csv format)."""
    h = hashlib.sha256(f"{idx}|{prompt}".encode()).hexdigest()
    return h[:8]


# Each family has: (puzzle_generator, cot_generator, type_label)
# For the transformation family, we have 4 sub-types (each 1000 rows).
# Six families matching the REAL train.csv distribution (~1/6 each). The
# "transformation" family is hidden-operator 2-operand arithmetic — the
# previous v3 split it into 4 keep/drop/move sub-types that matched ~0% of the
# real transformation prompts (see data/analysis/transformation_taxonomy.md),
# which is what sank the v3 leaderboard score.
FAMILIES: list[tuple[str, Callable, Callable]] = [
    ("gravity", pg.gen_gravity_puzzle, lambda parsed: pc.gravity_cot(**parsed)),
    ("unit_conversion", pg.gen_unit_conversion_puzzle, lambda parsed: pc.unit_conversion_cot(**parsed)),
    ("numeral", pg.gen_numeral_puzzle, lambda parsed: pc.numeral_cot(**parsed)),
    ("cipher", pg.gen_cipher_puzzle, lambda parsed: pc.cipher_cot(**parsed)),
    ("bit_manipulation", pg.gen_bit_manipulation_puzzle, lambda parsed: pc.bit_manipulation_cot(**parsed)),
    (
        "transformation",
        lambda rng: pg.gen_transformation_arithmetic(rng, "transformation"),
        lambda parsed: pc.arithmetic_transformation_cot(**parsed),
    ),
]


def build_dataset(
    out_csv: Path,
    rows_per_family: int = 1500,
    seed: int = 2026,
    max_retries: int = 10,
) -> dict:
    rng = random.Random(seed)
    stats: dict = {
        "rows_per_family_target": rows_per_family,
        "per_family": {},
        "total_rows": 0,
        "total_retries": 0,
    }

    rows: list[dict] = []
    idx = 0

    for type_label, gen_puzzle, gen_cot in FAMILIES:
        produced = 0
        retries = 0
        while produced < rows_per_family:
            # Each generator gets its own subseed for reproducibility
            sub_rng = random.Random(rng.randint(0, 1 << 30))
            try:
                puzzle = gen_puzzle(sub_rng)
            except Exception as e:
                retries += 1
                if retries > rows_per_family * max_retries:
                    raise RuntimeError(f"{type_label}: generator crashed too often: {e}")
                continue

            try:
                cot_result = gen_cot(puzzle["parsed"])
            except Exception as e:
                retries += 1
                continue

            if cot_result is None:
                retries += 1
                continue

            predicted, structured_cot = cot_result

            # Gate 1: the CoT-derived value must actually solve the puzzle.
            if not _solves_puzzle(predicted, puzzle["answer"], type_label):
                retries += 1
                continue

            # Emit the value the CoT actually derives — NOT the hidden gold.
            # The training notebook re-boxes `\boxed{answer}`, so `answer` must be
            # exactly what the reasoning concludes; otherwise the target teaches
            # "reason to X, then emit a different Y" (the v3 incoherence bug).
            emit_answer = predicted.strip()

            # Gate 2 (hard invariant): the boxed payload equals the emitted
            # answer byte-for-byte, for EVERY family.
            boxed = _extract_last_boxed(structured_cot)
            if boxed is None or boxed.strip() != emit_answer:
                retries += 1
                continue

            idx += 1
            rows.append({
                "id": _stable_id(puzzle["prompt"], idx),
                "prompt": puzzle["prompt"],
                "answer": emit_answer,
                "type": type_label,
                "generated_cot": structured_cot,
            })
            produced += 1

        stats["per_family"][type_label] = {
            "produced": produced,
            "retries": retries,
            "yield_rate": round(produced / (produced + retries), 4) if produced + retries else 0.0,
        }
        stats["total_retries"] += retries

    stats["total_rows"] = len(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "type", "generated_cot"])
        writer.writeheader()
        writer.writerows(rows)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--rows-per-family", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    stats = build_dataset(args.out_csv, args.rows_per_family, args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
