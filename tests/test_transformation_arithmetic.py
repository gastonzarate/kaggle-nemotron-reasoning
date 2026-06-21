"""Tests for the hidden-operator arithmetic transformation generator + CoT."""

from __future__ import annotations

import random
import re

import pytest

from src.data.programmatic_cot import (
    arith_result,
    arithmetic_transformation_cot,
    solve_arith_operator,
)
from src.data.puzzle_generators import gen_transformation_arithmetic


def _boxed(cot: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", cot)
    return m[-1].strip() if m else ""


# ── arith_result primitives ──────────────────────────────────────────────────


def test_arith_result_forward():
    assert arith_result("add", "fwd", "21", "75") == "96"
    assert arith_result("sub", "fwd", "39", "42") == "-3"
    assert arith_result("absdiff", "fwd", "39", "42") == "3"
    assert arith_result("mult", "fwd", "96", "54") == "5184"
    assert arith_result("concat", None, "20", "44") == "2044"
    assert arith_result("rconcat", None, "05", "12") == "1205"


def test_arith_result_rr():
    # reverse 96->69, 36->63, 69*63=4347, reverse -> 7434
    assert arith_result("mult", "RR", "96", "36") == "7434"
    # reverse 12->21, 34->43, 21+43=64, reverse -> 46
    assert arith_result("add", "RR", "12", "34") == "46"


# ── discoverer ────────────────────────────────────────────────────────────────


def test_solve_operator_recovers_mult():
    rows = [("96", "54", "5184"), ("12", "11", "132")]
    assert solve_arith_operator(rows) == ("mult", "fwd")


def test_solve_operator_none_when_inconsistent():
    rows = [("10", "20", "30"), ("10", "20", "999")]
    assert solve_arith_operator(rows) is None


# ── generator + CoT end-to-end ────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(200))
def test_generated_puzzle_is_wellposed_and_coherent(seed):
    rng = random.Random(seed)
    try:
        p = gen_transformation_arithmetic(rng)
    except ValueError:
        pytest.skip("under-determined puzzle (expected to be retried by orchestrator)")

    # LHS always 5 chars
    body = p["prompt"].split("examples:\n", 1)[1]
    lines = [ln for ln in body.splitlines() if " = " in ln]
    for ln in lines:
        lhs = ln.split(" = ", 1)[0]
        assert len(lhs) == 5, f"LHS not 5 chars: {lhs!r}"
    assert len(p["parsed"]["query"]) == 5

    # CoT derives the gold answer, boxed == answer exactly
    out = arithmetic_transformation_cot(**p["parsed"])
    assert out is not None, "CoT failed to recover the rule on a well-posed puzzle"
    predicted, cot = out
    assert predicted == p["answer"], f"{predicted!r} != {p['answer']!r}"
    assert _boxed(cot) == p["answer"]
    # structural envelope
    assert cot.count("<think>") == 1 and cot.count("</think>") == 1
    assert cot.count("\\boxed{") == 1


def test_query_operator_always_demonstrated():
    """The query's operator must appear in the examples (else unsolvable)."""
    for seed in range(100):
        rng = random.Random(seed)
        try:
            p = gen_transformation_arithmetic(rng)
        except ValueError:
            continue
        q_op = p["parsed"]["query"][2]
        example_ops = {lhs[2] for lhs, _ in p["parsed"]["pairs"]}
        assert q_op in example_ops
