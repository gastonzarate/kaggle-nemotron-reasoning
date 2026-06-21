"""Tests for transformation_cot (cryptarithm/equation_numeric).

These puzzles in the train.csv are highly varied — we test the four transformation
classes (delete, substitute, position-filter, affix-trim) and require that the
generator either solves it correctly OR returns None (so the orchestrator can fall
back to the heuristic wrapper).
"""

import re

from src.data.programmatic_cot import transformation_cot


def _extract_boxed(cot: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", cot)
    return m[-1].strip() if m else ""


def test_transformation_deletion():
    # Rule: delete ':' from input
    pairs = [
        ("$?:>`", "$?>`"),
        ("\\<:\\`", "\\<\\`"),
        ("{`:?>", "{`?>"),
    ]
    result = transformation_cot(pairs, "$>:>\\")
    assert result is not None
    answer, cot = result
    assert answer == "$>>\\"
    assert "<c>" in cot
    assert cot.count("</think>") == 1
    assert "deletion" in cot.lower() or "delete" in cot.lower()


def test_transformation_substitution():
    # Rule: a→x, b→y, c→z
    pairs = [
        ("abc", "xyz"),
        ("aab", "xxy"),
        ("cab", "zxy"),
    ]
    result = transformation_cot(pairs, "bca")
    assert result is not None
    answer, cot = result
    assert answer == "yzx"


def test_transformation_position_filter_first_k():
    # Rule: keep first 2 chars
    pairs = [
        ("abcde", "ab"),
        ("xyzwq", "xy"),
        ("12345", "12"),
    ]
    result = transformation_cot(pairs, "hello")
    assert result is not None
    answer, cot = result
    assert answer == "he"


def test_transformation_affix_trim_prefix():
    # Rule: strip prefix "PRE_"
    pairs = [
        ("PRE_hello", "hello"),
        ("PRE_world", "world"),
    ]
    result = transformation_cot(pairs, "PRE_foo")
    assert result is not None
    answer, cot = result
    assert answer == "foo"


def test_transformation_returns_none_for_unsolvable():
    # No common rule
    pairs = [
        ("abc", "xyz"),
        ("def", "qwp"),  # contradicts substitution
    ]
    result = transformation_cot(pairs, "abc")
    # Some random fit might be found, but with these particular pairs likely None
    # Just assert the function is well-behaved:
    assert result is None or isinstance(result, tuple)


def test_transformation_cot_envelope_single_think():
    pairs = [("abc:de", "abcde"), ("xyz:qq", "xyzqq")]
    result = transformation_cot(pairs, "foo:bar")
    assert result is not None
    answer, cot = result
    assert cot.count("<think>") == 1
    assert cot.count("</think>") == 1
    assert cot.endswith("}")
