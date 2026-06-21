"""Heuristic CoT rewriter — wraps existing generated_cot with structured tags + router.

Reads CoT-tong dataset and produces a new dataset where each generated_cot is wrapped
in our `<think><r>...</r><tag>...</tag></think>` schema, then re-attached to `\\boxed{answer}`
during training (we don't touch the boxing — that's done by the training notebook).

Family → tag plan mapping:
- bit_manipulation:     <c>   (algorithmic search of bitwise rule)
- cipher:               <l>   (build substitution map, then apply)
- gravity:              <m>   (compute k = d/(0.5·t²), then apply)
- unit_conversion:      <m>   (fit linear Y = a·X + b, then apply)
- numeral:              <m>   (identify system, then convert)
- cryptarithm_deduce:   <m>   (assign digits, verify arithmetic)
- cryptarithm_guess:    <m>   (idem; ambiguous → most-likely assignment)
- equation_numeric_deduce: <m> (find pattern, apply)
- equation_numeric_guess:  <m> (idem; ambiguous)

For each family, the router declares the plan, then the existing CoT (slightly cleaned)
goes inside the matching tag. Synthesis line is added at the end.

Additional fixes applied per-family:
- gravity / unit_conversion: round any explicit "Result:" line to 2dp half-up.
"""

from __future__ import annotations

import argparse
import csv
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

FAMILY_PLAN = {
    "bit_manipulation": (
        "c",
        "family=bit_manipulation; plan: enumerate per-bit boolean rules with code-mode, verify against all 10 examples",
    ),
    "cipher": (
        "l",
        "family=cipher; plan: build substitution map from examples with ling-mode, apply to query",
    ),
    "gravity": (
        "m",
        "family=gravity; plan: extract k via math-mode from (t, d) pairs, apply to query t",
    ),
    "unit_conversion": (
        "m",
        "family=unit_conversion; plan: fit linear Y = a·X + b via math-mode, apply to query X",
    ),
    "numeral": (
        "m",
        "family=numeral; plan: identify the hidden numeral system via math-mode, convert query",
    ),
    "cryptarithm_deduce": (
        "m",
        "family=cryptarithm_deduce; plan: assign digits via math-mode, verify arithmetic",
    ),
    "cryptarithm_guess": (
        "m",
        "family=cryptarithm_guess; plan: best-effort digit assignment via math-mode",
    ),
    "equation_numeric_deduce": (
        "m",
        "family=equation_numeric_deduce; plan: find numeric pattern via math-mode, apply to query",
    ),
    "equation_numeric_guess": (
        "m",
        "family=equation_numeric_guess; plan: best-effort numeric extrapolation via math-mode",
    ),
}


_NUM_3DP_RE = re.compile(r"(?<![\d.])(-?\d+\.\d{3,})")


def round_to_2dp_half_up(match: re.Match) -> str:
    raw = match.group(1)
    try:
        rounded = Decimal(raw).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception:
        return raw
    s = format(rounded, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".") if False else s
    return s


def fix_precision_for_numeric_family(cot: str) -> str:
    return _NUM_3DP_RE.sub(round_to_2dp_half_up, cot)


def strip_existing_boxed(cot: str) -> str:
    return re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()


_THINK_TAG_RE = re.compile(r"</?think\s*>", re.IGNORECASE)
_BOILERPLATE_RE = re.compile(
    r"I will (now )?put my final answer inside.*?\\boxed\{[^}]*\}\s*\.?",
    re.IGNORECASE | re.DOTALL,
)
_INSTRUCTION_LINE_RE = re.compile(
    r"^[^\n]*I will (now )?put.*?inside.*?\\boxed.*?$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_cot_body(cot: str, family: str) -> str:
    """Strip anything that would conflict with our outer envelope:
    - the original \\boxed{} (we add our own after </think>)
    - any <think> or </think> from the teacher CoT (we wrap with our own)
    - boilerplate "I will put my final answer inside \\boxed{}" instructions
    """
    cot = strip_existing_boxed(cot)
    cot = _THINK_TAG_RE.sub("", cot)
    cot = _BOILERPLATE_RE.sub("", cot)
    cot = _INSTRUCTION_LINE_RE.sub("", cot)
    if family in {"gravity", "unit_conversion"}:
        cot = fix_precision_for_numeric_family(cot)
    cot = cot.strip()
    return cot


def build_structured_cot(prompt: str, cot: str, family: str, answer: str) -> str:
    """Wrap a CoT body inside the v2 envelope. Always emits one \\boxed{} after </think>.

    Output:
        <think>
        <r>plan</r>
        <tag>
        body (cleaned of conflicting envelope tokens)
        </tag>
        </think>
        \\boxed{answer}
    """
    plan_meta = FAMILY_PLAN.get(family)
    if plan_meta is None:
        plan_meta = ("m", f"family={family}; plan: generic reasoning via math-mode")
    tag, plan_str = plan_meta
    body = clean_cot_body(cot, family)
    structured = (
        f"<think>\n"
        f"<r>{plan_str}</r>\n"
        f"<{tag}>\n"
        f"{body}\n"
        f"</{tag}>\n"
        f"</think>\n"
        f"\\boxed{{{answer}}}"
    )
    return structured


def rewrite_dataset(in_csv: Path, out_csv: Path) -> dict:
    rows_in: list[dict] = []
    with open(in_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_key = next((k for k in row if k.lstrip("﻿") == "id"), None)
            if id_key:
                row["id"] = row.pop(id_key)
            rows_in.append(row)

    per_family_counts = {}
    dropped = {"missing_cot": 0, "missing_answer": 0, "missing_prompt": 0}

    rows_out = []
    for row in rows_in:
        cot = (row.get("generated_cot") or "").strip()
        answer = (row.get("answer") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        family = (row.get("type") or "").strip()

        if not cot:
            dropped["missing_cot"] += 1
            continue
        if not answer:
            dropped["missing_answer"] += 1
            continue
        if not prompt:
            dropped["missing_prompt"] += 1
            continue

        structured = build_structured_cot(prompt, cot, family, answer)
        rows_out.append(
            {
                "id": row.get("id", ""),
                "prompt": prompt,
                "answer": answer,
                "type": family,
                "generated_cot": structured,
            }
        )
        per_family_counts[family] = per_family_counts.get(family, 0) + 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "prompt", "answer", "type", "generated_cot"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    return {
        "input_rows": len(rows_in),
        "output_rows": len(rows_out),
        "dropped": dropped,
        "per_family": dict(sorted(per_family_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    stats = rewrite_dataset(args.in_csv, args.out_csv)
    import json

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
