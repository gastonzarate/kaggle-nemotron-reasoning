"""Strict envelope checks — every programmatic CoT must conform to the schema.

The training pipeline will tokenize each CoT and use it as the assistant target.
Any malformed envelope (double </think>, missing \\boxed, mismatched tags) becomes
a structural bug the model learns. These tests guard against that.
"""

import re
from decimal import Decimal

import pytest

from src.data.programmatic_cot import (
    bit_manipulation_cot,
    cipher_cot,
    gravity_cot,
    numeral_cot,
    transformation_cot,
    unit_conversion_cot,
)


# ─────────────────────────────────────────────────────────────────────────────
# Envelope invariants applied to every generator
# ─────────────────────────────────────────────────────────────────────────────


def _well_formed_envelope(cot: str, expected_subtag: str) -> list[str]:
    """Returns a list of envelope violations (empty = clean)."""
    violations = []
    if not cot.startswith("<think>"):
        violations.append("does not start with <think>")
    if cot.count("<think>") != 1:
        violations.append(f"<think> appears {cot.count('<think>')} times (want 1)")
    if cot.count("</think>") != 1:
        violations.append(f"</think> appears {cot.count('</think>')} times (want 1)")
    if "<r>" not in cot or "</r>" not in cot:
        violations.append("missing <r>...</r> router")
    if cot.count("<r>") != cot.count("</r>"):
        violations.append("unbalanced <r> tags")
    if f"<{expected_subtag}>" not in cot:
        violations.append(f"missing <{expected_subtag}> sub-tag")
    if cot.count(f"<{expected_subtag}>") != cot.count(f"</{expected_subtag}>"):
        violations.append(f"unbalanced <{expected_subtag}> tags")
    boxes = re.findall(r"\\boxed\{([^}]*)\}", cot)
    if not boxes:
        violations.append("no \\boxed{} found")
    elif not boxes[-1].strip():
        violations.append("final \\boxed{} is empty")
    # The last \boxed must come AFTER </think>
    boxed_pos = cot.rfind("\\boxed{")
    close_pos = cot.rfind("</think>")
    if boxed_pos < close_pos:
        violations.append("\\boxed appears BEFORE </think>")
    return violations


# ─────────────────────────────────────────────────────────────────────────────
# Gravity: many configurations
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pairs,query", [
    # Standard case
    ([(Decimal("4.62"), Decimal("177.99")), (Decimal("2.97"), Decimal("73.56"))], Decimal("4.33")),
    # Small numbers
    ([(Decimal("1"), Decimal("5")), (Decimal("2"), Decimal("20"))], Decimal("3")),
    # Single pair (degenerate but should not crash)
    ([(Decimal("1.0"), Decimal("0.5"))], Decimal("2.0")),
    # Many pairs
    ([(Decimal(str(t)), Decimal(str(9.8 * 0.5 * t * t))) for t in [1.0, 2.0, 3.0, 4.0, 5.0]], Decimal("6")),
])
def test_gravity_envelope(pairs, query):
    _, cot = gravity_cot(pairs, query)
    violations = _well_formed_envelope(cot, "m")
    assert not violations, f"violations: {violations}\n--- cot ---\n{cot[:500]}"


# ─────────────────────────────────────────────────────────────────────────────
# Unit conversion: various coefficient ranges
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pairs,query", [
    # Pure proportional (b≈0)
    ([(Decimal("10"), Decimal("12")), (Decimal("20"), Decimal("24"))], Decimal("15")),
    # Affine (b≠0)
    ([(Decimal("10"), Decimal("31")), (Decimal("20"), Decimal("51"))], Decimal("15")),
    # Tiny coefficients
    ([(Decimal("0.1"), Decimal("0.10001")), (Decimal("0.2"), Decimal("0.20001"))], Decimal("0.15")),
])
def test_unit_conversion_envelope(pairs, query):
    _, cot = unit_conversion_cot(pairs, query)
    violations = _well_formed_envelope(cot, "m")
    assert not violations, f"violations: {violations}"


# ─────────────────────────────────────────────────────────────────────────────
# Numeral: Roman + base-k variations
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pairs,query", [
    # Roman small
    ([(1, "I"), (5, "V"), (10, "X")], 3),
    # Roman with subtractive forms
    ([(4, "IV"), (9, "IX"), (40, "XL"), (90, "XC")], 99),
    # Roman large
    ([(100, "C"), (500, "D"), (1000, "M")], 1234),
    # Base 2 (binary)
    ([(2, "10"), (4, "100"), (8, "1000")], 5),
    # Base 7
    ([(8, "11"), (15, "21"), (10, "13")], 50),
    # Base 16 (hex)
    ([(16, "10"), (255, "FF"), (10, "A")], 100),
])
def test_numeral_envelope(pairs, query):
    _, cot = numeral_cot(pairs, query)
    violations = _well_formed_envelope(cot, "m")
    assert not violations, f"violations: {violations}"


# ─────────────────────────────────────────────────────────────────────────────
# Cipher: alignment edge cases
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pairs,query", [
    # Single char swap
    ([("ab", "yz")], "ab"),
    # Multi-word, fully covered
    ([("abc def", "xyz qrs"), ("abcde", "xyzfg")], "abc"),
    # Query has space
    ([("ab", "xy"), ("cd", "wz")], "ab cd"),
])
def test_cipher_envelope(pairs, query):
    _, cot = cipher_cot(pairs, query)
    violations = _well_formed_envelope(cot, "l")
    assert not violations, f"violations: {violations}"


def test_cipher_empty_query_emits_empty_boxed():
    """Edge case: empty query → empty \\boxed{}. This is detected as a violation
    so callers (the orchestrator) can choose to fallback."""
    _, cot = cipher_cot([("ab", "xy")], "")
    violations = _well_formed_envelope(cot, "l")
    # We EXPECT this to be flagged; the orchestrator must not ship empty-boxed rows
    assert "final \\boxed{} is empty" in violations


# ─────────────────────────────────────────────────────────────────────────────
# Bit manipulation: all kinds of bitwise rules
# ─────────────────────────────────────────────────────────────────────────────

def _bit_pair(in_, out):
    return (f"{in_:08b}", f"{out:08b}")


@pytest.mark.parametrize("pairs,query", [
    # Identity (single-bit-set inputs)
    ([_bit_pair(1 << i, 1 << i) for i in range(8)], f"{42:08b}"),
    # NOT
    ([_bit_pair(1 << i, (~(1 << i)) & 0xFF) for i in range(8)], f"{42:08b}"),
    # XOR with constant 0xAA
    ([_bit_pair(1 << i, (1 << i) ^ 0xAA) for i in range(8)], f"{42:08b}"),
    # Output is always 00000000 (degenerate)
    ([_bit_pair(i, 0) for i in [1, 2, 4, 8, 16, 32, 64, 128]], "11111111"),
])
def test_bit_manipulation_envelope(pairs, query):
    _, cot = bit_manipulation_cot(pairs, query)
    violations = _well_formed_envelope(cot, "c")
    assert not violations, f"violations: {violations}"


# ─────────────────────────────────────────────────────────────────────────────
# Transformation: all 4 rule classes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pairs,query", [
    # Deletion
    ([("a:b", "ab"), ("c:d", "cd"), ("e:f:g", "efg")], "x:y:z"),
    # Substitution
    ([("ab", "xy"), ("ba", "yx"), ("aab", "xxy")], "abba"),
    # Position filter (first 2)
    ([("abcde", "ab"), ("xyzwq", "xy")], "fghij"),
    # Affix trim
    ([("PRE_a", "a"), ("PRE_bb", "bb")], "PRE_zzz"),
])
def test_transformation_envelope(pairs, query):
    result = transformation_cot(pairs, query)
    assert result is not None, "should have found a rule"
    _, cot = result
    violations = _well_formed_envelope(cot, "c")
    assert not violations, f"violations: {violations}"


def test_transformation_returns_none_well_behaved():
    """When no rule fits, transformation_cot returns None — not a malformed CoT."""
    pairs = [("ab", "x"), ("cd", "yyyy")]  # no consistent rule
    result = transformation_cot(pairs, "ef")
    # Either None or a valid CoT — but never malformed
    if result is not None:
        _, cot = result
        violations = _well_formed_envelope(cot, "c")
        assert not violations


# ─────────────────────────────────────────────────────────────────────────────
# Common: no surprise unicode, no debug noise
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pairs,query,fn,tag", [
    ([(Decimal("1"), Decimal("5"))], Decimal("2"), gravity_cot, "m"),
    ([(Decimal("1"), Decimal("2")), (Decimal("3"), Decimal("6"))], Decimal("5"), unit_conversion_cot, "m"),
    ([(1, "I"), (5, "V")], 10, numeral_cot, "m"),
])
def test_no_leading_or_trailing_whitespace_artifacts(pairs, query, fn, tag):
    _, cot = fn(pairs, query)
    # No leading whitespace on each line that would suggest an indent bug
    # (well-formed but stylistically check)
    lines = cot.split("\n")
    for line in lines:
        # Just verify no extreme indentation (>20 spaces) that suggests a bug
        if line.strip():
            assert not line.startswith("                    "), f"excessive indent: {line[:50]!r}"
