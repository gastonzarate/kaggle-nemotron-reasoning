"""Scaffold of the structured-CoT regenerator pipeline.

NOTE: this module is intentionally a SKELETON. It defines:

  - The dataclasses for a CoT-generation request and response.
  - The per-family prompt templates that wrap the few-shot examples
    (`data/synthetic/schema/examples/<family>.md`) plus the new prompt.
  - The placeholder teacher-LLM client (`TeacherClient`) whose `.generate`
    is unimplemented — the caller wires in a vLLM / HF endpoint at runtime.
  - The validation pipeline that parses the teacher's output, runs the
    family-specific verifier (Section 6 of `SCHEMA.md`), and either keeps
    or rejects the row.

We do NOT call any teacher API in this file. Tests exercise everything
except the actual `TeacherClient.generate` call (which is monkeypatched
in the tests).

Public entry point:
    regenerate_dataset(
        input_df: pd.DataFrame,
        teacher: TeacherClient,
        out_path: pathlib.Path,
    ) -> RegenerationReport

Each surviving row is appended to `out_path` (JSONL) with schema:
    {id, prompt, answer, type, structured_cot, schema_version}
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import time
from collections import defaultdict
from typing import Iterable, Optional

# Local import — robust against being invoked as a script (where `src.` is not
# on sys.path) or as a module (`python -m src.data.regenerate_cot`).
try:
    from src.eval.parse_structured_cot import (
        ParsedCoT,
        parse_structured_cot,
        verify_against_gold,
    )
except ImportError:  # pragma: no cover - script-style invocation
    import sys as _sys

    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from src.eval.parse_structured_cot import (  # type: ignore[no-redef]
        ParsedCoT,
        parse_structured_cot,
        verify_against_gold,
    )


SCHEMA_VERSION = "v1"

FAMILIES = (
    "bit_manipulation",
    "cipher",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "gravity",
    "numeral",
    "unit_conversion",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CoTRequest:
    """One row of the input dataset to be regenerated."""

    id: str
    prompt: str
    answer: str
    family: str  # one of FAMILIES

    def __post_init__(self) -> None:  # pragma: no cover - simple guard
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown family: {self.family}")


@dataclasses.dataclass
class CoTResponse:
    """One regenerated CoT, post-verification."""

    request: CoTRequest
    structured_cot: str  # full <think>...</think>\boxed{...} string
    parsed: ParsedCoT
    accepted: bool
    rejection_reason: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    teacher_latency_ms: float = 0.0


@dataclasses.dataclass
class RegenerationReport:
    accepted: dict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    rejected: dict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    rejection_reasons: dict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    def add(self, response: CoTResponse) -> None:
        if response.accepted:
            self.accepted[response.request.family] += 1
        else:
            self.rejected[response.request.family] += 1
            if response.rejection_reason:
                self.rejection_reasons[response.rejection_reason] += 1

    def as_dict(self) -> dict:
        total_acc = sum(self.accepted.values())
        total_rej = sum(self.rejected.values())
        return {
            "accepted": dict(self.accepted),
            "rejected": dict(self.rejected),
            "rejection_reasons": dict(self.rejection_reasons),
            "totals": {
                "accepted": total_acc,
                "rejected": total_rej,
                "acceptance_rate": (
                    total_acc / (total_acc + total_rej) if (total_acc + total_rej) else 0.0
                ),
            },
        }


# ---------------------------------------------------------------------------
# Teacher client (placeholder)
# ---------------------------------------------------------------------------


class TeacherClient:
    """Abstract teacher-LLM client. Subclass and implement `.generate`.

    Concrete implementations might wrap:
      - a local vLLM server,
      - the HuggingFace text-generation endpoint,
      - a `transformers.pipeline` instance.

    We do NOT ship a default — generation is not allowed in this branch.
    """

    def generate(self, system: str, user: str, max_tokens: int = 4096) -> str:
        raise NotImplementedError(
            "TeacherClient.generate is intentionally unimplemented in the scaffold. "
            "Subclass with a real backend before calling regenerate_dataset()."
        )


# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------


_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "synthetic" / "schema" / "examples"


def load_family_examples(family: str) -> str:
    """Return the contents of data/synthetic/schema/examples/<family>.md.

    Used as few-shot context for the teacher.
    """
    path = _EXAMPLES_DIR / f"{family}.md"
    if not path.exists():
        raise FileNotFoundError(f"No few-shot examples found for family {family!r} at {path}")
    return path.read_text(encoding="utf-8")


def build_teacher_prompt(req: CoTRequest) -> tuple[str, str]:
    """Build (system, user) messages for the teacher.

    Returns plain strings; the caller adapts to its chat template.
    """
    examples_md = load_family_examples(req.family)
    system = (
        "You generate chain-of-thought traces in the structured-CoT v1 schema. "
        "Inside <think>...</think> you must tag every reasoning block with exactly one of "
        "<m>...</m> (math), <c>...</c> (code/algorithmic), or <l>...</l> (linguistic). "
        "Tags do not nest, do not overlap, and follow the family-specific order from the schema. "
        "Each block ends with concrete intermediate values (no filler). The last block ends with "
        "a line 'FINAL: <answer>' that must exactly match the value placed in \\boxed{...}. "
        "Apply all bug-fix conventions: 2-dp half-up rounding for numeric families, brace-safe "
        "trailing-backslash padding, output-width zero-padding for equation families."
    )
    user = (
        f"Family: {req.family}\n\n"
        f"Reference few-shot examples for this family:\n\n{examples_md}\n\n"
        f"------\n"
        f"Now produce the structured CoT for the following prompt. The gold answer is "
        f"{req.answer!r}; your FINAL: line and \\boxed{{}} must agree with it after the "
        f"family-specific normalisation.\n\n"
        f"Prompt:\n{req.prompt}\n"
    )
    return system, user


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------


def regenerate_one(req: CoTRequest, teacher: TeacherClient) -> CoTResponse:
    """Generate one structured CoT, parse it, verify against gold."""
    system, user = build_teacher_prompt(req)
    t0 = time.monotonic()
    raw = teacher.generate(system, user)
    latency_ms = (time.monotonic() - t0) * 1000.0

    try:
        parsed = parse_structured_cot(raw)
    except Exception as exc:  # noqa: BLE001 — we want the reason string
        empty = ParsedCoT(
            math_steps=[],
            code_steps=[],
            ling_steps=[],
            tag_order=[],
            final_line=None,
            boxed=None,
            agree=False,
        )
        return CoTResponse(
            request=req,
            structured_cot=raw,
            parsed=empty,
            accepted=False,
            rejection_reason=f"parse_error: {type(exc).__name__}",
            teacher_latency_ms=latency_ms,
        )

    ok, reason = verify_against_gold(parsed, gold_answer=req.answer, family=req.family)
    return CoTResponse(
        request=req,
        structured_cot=raw,
        parsed=parsed,
        accepted=ok,
        rejection_reason=None if ok else reason,
        teacher_latency_ms=latency_ms,
    )


def regenerate_dataset(
    requests: Iterable[CoTRequest],
    teacher: TeacherClient,
    out_path: pathlib.Path,
    *,
    max_retries: int = 2,
) -> RegenerationReport:
    """Stream-generate the whole dataset, writing accepted rows to JSONL.

    Rejected rows are dropped silently but counted in the returned report.
    Each request is retried up to `max_retries` times if the first attempt
    fails verification.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = RegenerationReport()

    with out_path.open("a", encoding="utf-8") as fh:
        for req in requests:
            response: Optional[CoTResponse] = None
            for _ in range(max_retries + 1):
                response = regenerate_one(req, teacher)
                if response.accepted:
                    break
            assert response is not None
            report.add(response)
            if response.accepted:
                fh.write(
                    json.dumps(
                        {
                            "id": req.id,
                            "prompt": req.prompt,
                            "answer": req.answer,
                            "type": req.family,
                            "structured_cot": response.structured_cot,
                            "schema_version": response.schema_version,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return report


# ---------------------------------------------------------------------------
# Smoke tests (run with `python -m src.data.regenerate_cot`)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Minimal smoke test — verifies dataclasses, template loading, and the
    parse + verify path on a hand-rolled accepted CoT (no teacher call)."""

    req = CoTRequest(
        id="dummy",
        prompt="...",
        answer="LX",
        family="numeral",
    )
    sys_msg, user_msg = build_teacher_prompt(req)
    assert "<m>" in sys_msg
    assert "Family: numeral" in user_msg
    # Make sure all 9 example files exist.
    for fam in FAMILIES:
        load_family_examples(fam)
    # Stub teacher that returns a hand-crafted valid response.
    canned = (
        "<think>\n"
        "<c>\nexamples decode as Roman.\n</c>\n"
        "<c>\n60 -> LX. apply.\nFINAL: LX\n</c>\n"
        "</think>\n"
        "\\boxed{LX}\n"
    )

    class StubTeacher(TeacherClient):
        def generate(self, system: str, user: str, max_tokens: int = 4096) -> str:
            return canned

    resp = regenerate_one(req, StubTeacher())
    assert resp.accepted, resp.rejection_reason
    assert resp.parsed.boxed == "LX"
    assert resp.parsed.final_line == "LX"
    print("self-test OK:", resp.accepted, resp.parsed.tag_order)


if __name__ == "__main__":  # pragma: no cover
    _self_test()
