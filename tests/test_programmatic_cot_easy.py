"""Tests for the easy-tier programmatic CoTs (gravity, unit_conv, numeral)."""

import re
from decimal import Decimal

from src.data.programmatic_cot import (
    gravity_cot,
    numeral_cot,
    to_base_k,
    to_roman,
    unit_conversion_cot,
)


def _extract_boxed(cot: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", cot)
    assert m, f"no \\boxed in CoT: {cot[:200]}"
    return m[-1].strip()


def test_to_roman_basic_values():
    assert to_roman(1) == "I"
    assert to_roman(4) == "IV"
    assert to_roman(9) == "IX"
    assert to_roman(40) == "XL"
    assert to_roman(43) == "XLIII"
    assert to_roman(94) == "XCIV"
    assert to_roman(1994) == "MCMXCIV"


def test_to_base_k():
    assert to_base_k(10, 2) == "1010"
    assert to_base_k(255, 16) == "FF"
    assert to_base_k(8, 8) == "10"
    assert to_base_k(0, 7) == "0"


def test_gravity_cot_classic_case():
    pairs = [
        (Decimal("4.62"), Decimal("177.99")),
        (Decimal("2.97"), Decimal("73.56")),
        (Decimal("4.74"), Decimal("187.36")),
    ]
    answer, cot = gravity_cot(pairs, Decimal("4.33"))
    # Expected gold = 156.35 (within 1% rel_tol of competition verifier)
    boxed = _extract_boxed(cot)
    assert abs(Decimal(boxed) - Decimal("156.35")) < Decimal("0.5")
    assert "<r>" in cot and "</r>" in cot
    assert "<m>" in cot and "</m>" in cot
    assert "<think>" in cot and "</think>" in cot
    assert cot.count("</think>") == 1  # no duplicate close tags (the v0.2.0 bug)


def test_unit_conversion_cot_linear_factor():
    # Y = 1.062·X (b=0); examples cooked accordingly
    pairs = [
        (Decimal("18.75"), Decimal("19.91")),
        (Decimal("46.61"), Decimal("49.50")),
        (Decimal("29.36"), Decimal("31.18")),
    ]
    answer, cot = unit_conversion_cot(pairs, Decimal("15.19"))
    boxed = _extract_boxed(cot)
    # 1.062 · 15.19 ≈ 16.13
    assert abs(Decimal(boxed) - Decimal("16.13")) < Decimal("0.2")
    assert "<m>" in cot
    assert cot.count("</think>") == 1


def test_numeral_cot_roman():
    pairs = [(3, "III"), (29, "XXIX"), (79, "LXXIX"), (4, "IV")]
    answer, cot = numeral_cot(pairs, 43)
    boxed = _extract_boxed(cot)
    assert boxed == "XLIII"
    assert "Roman" in cot
    assert cot.count("</think>") == 1


def test_numeral_cot_base_k():
    # Base 7 examples
    pairs = [(8, "11"), (10, "13"), (15, "21")]
    answer, cot = numeral_cot(pairs, 50)
    boxed = _extract_boxed(cot)
    # 50 in base 7 = 101
    assert boxed == "101"
    assert "base-7" in cot or "base 7" in cot


def test_cot_envelope_well_formed():
    """No structural bugs (the double-</think> issue from v0.2.0 must not recur)."""
    pairs = [(Decimal("2"), Decimal("8")), (Decimal("3"), Decimal("18"))]
    _, cot = gravity_cot(pairs, Decimal("4"))
    assert cot.startswith("<think>")
    assert cot.endswith("}")
    assert cot.count("<think>") == 1
    assert cot.count("</think>") == 1
    # Router must come BEFORE the math tag
    assert cot.index("<r>") < cot.index("<m>")
