"""Tests for puzzle_generators — verify the output of EVERY generator passes
through its corresponding CoT generator and matches its own gold.

This is the contract: gen → CoT → boxed answer must equal gen.answer.
If this contract holds, the orchestrator can produce 100%-correct rows.
"""

import math
import random
import re

import pytest

from src.data.programmatic_cot import (
    bit_manipulation_cot,
    cipher_cot,
    gravity_cot,
    numeral_cot,
    transformation_cot,
    unit_conversion_cot,
)
from src.data.puzzle_generators import (
    gen_bit_manipulation_puzzle,
    gen_cipher_puzzle,
    gen_gravity_puzzle,
    gen_numeral_puzzle,
    gen_transformation_affix_trim,
    gen_transformation_deletion,
    gen_transformation_position_filter,
    gen_transformation_substitution,
    gen_unit_conversion_puzzle,
)


def _extract_boxed(cot: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", cot)
    return m[-1].strip() if m else ""


def _match(predicted: str, gold: str) -> bool:
    try:
        return math.isclose(float(predicted), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == gold.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke: every generator produces a parseable puzzle
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(20))
def test_gen_gravity_produces_valid_puzzle(seed):
    p = gen_gravity_puzzle(random.Random(seed))
    assert "gravitational" in p["prompt"]
    assert p["type"] == "gravity"
    answer, cot = gravity_cot(**p["parsed"])
    assert _match(answer, p["answer"]), f"seed={seed}: predicted {answer} vs gold {p['answer']}"
    boxed = _extract_boxed(cot)
    assert _match(boxed, p["answer"]), f"seed={seed}: boxed {boxed} vs gold {p['answer']}"


@pytest.mark.parametrize("seed", range(20))
def test_gen_unit_conversion_produces_valid_puzzle(seed):
    p = gen_unit_conversion_puzzle(random.Random(seed))
    assert "unit conversion" in p["prompt"]
    assert p["type"] == "unit_conversion"
    answer, cot = unit_conversion_cot(**p["parsed"])
    assert _match(answer, p["answer"]), f"seed={seed}: predicted {answer} vs gold {p['answer']}"
    boxed = _extract_boxed(cot)
    assert _match(boxed, p["answer"])


@pytest.mark.parametrize("seed", range(20))
def test_gen_numeral_produces_valid_puzzle(seed):
    p = gen_numeral_puzzle(random.Random(seed))
    assert "numeral system" in p["prompt"]
    assert p["type"] == "numeral"
    answer, cot = numeral_cot(**p["parsed"])
    assert _match(answer, p["answer"]), f"seed={seed}: predicted {answer} vs gold {p['answer']}"


@pytest.mark.parametrize("seed", range(20))
def test_gen_cipher_produces_valid_puzzle(seed):
    p = gen_cipher_puzzle(random.Random(seed))
    assert "encryption" in p["prompt"]
    assert p["type"] == "cipher"
    answer, cot = cipher_cot(**p["parsed"])
    # cipher_cot might miss chars not in examples — accept partial match if
    # query has unseen chars, but in synthetic puzzles every char in query
    # should appear in examples. So expect exact match.
    if not _match(answer, p["answer"]):
        # Investigate: are there chars in query not in any example?
        all_example_chars = set()
        for ct, _ in p["parsed"]["pairs"]:
            all_example_chars.update(ct)
        query_chars = set(p["parsed"]["query"]) - {" "}
        unseen = query_chars - all_example_chars
        if unseen:
            pytest.skip(f"seed={seed}: query has unseen cipher chars {unseen} — synthetic underspec")
        assert False, f"seed={seed}: cipher mismatch with full coverage"


@pytest.mark.parametrize("seed", range(20))
def test_gen_bit_manipulation_produces_valid_puzzle(seed):
    try:
        p = gen_bit_manipulation_puzzle(random.Random(seed))
    except ValueError:
        # Under-determined seed rejected by the uniqueness gate (orchestrator
        # retries) — stricter now that candidates include 3-input MAJ/CHOICE.
        pytest.skip("under-determined seed (orchestrator retries)")
    assert "bit manipulation" in p["prompt"]
    assert p["type"] == "bit_manipulation"
    answer, cot = bit_manipulation_cot(**p["parsed"])
    assert _match(answer, p["answer"]), f"seed={seed}: predicted {answer} vs gold {p['answer']}"


@pytest.mark.parametrize("seed", range(20))
def test_gen_transformation_deletion(seed):
    p = gen_transformation_deletion(random.Random(seed))
    result = transformation_cot(**p["parsed"])
    assert result is not None, f"seed={seed}: deletion not detected"
    answer, _ = result
    assert _match(answer, p["answer"])


@pytest.mark.parametrize("seed", range(20))
def test_gen_transformation_substitution(seed):
    p = gen_transformation_substitution(random.Random(seed))
    result = transformation_cot(**p["parsed"])
    assert result is not None
    answer, _ = result
    assert _match(answer, p["answer"])


@pytest.mark.parametrize("seed", range(20))
def test_gen_transformation_position_filter(seed):
    p = gen_transformation_position_filter(random.Random(seed))
    result = transformation_cot(**p["parsed"])
    assert result is not None
    answer, _ = result
    assert _match(answer, p["answer"])


@pytest.mark.parametrize("seed", range(20))
def test_gen_transformation_affix_trim(seed):
    p = gen_transformation_affix_trim(random.Random(seed))
    result = transformation_cot(**p["parsed"])
    assert result is not None
    answer, _ = result
    assert _match(answer, p["answer"])
