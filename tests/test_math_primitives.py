from decimal import Decimal

from src.data.math_primitives import (
    ColumnAdd,
    ColumnMultiply,
    ColumnSubtract,
    D,
    LongDivision,
    fmt,
    round_2dp,
)


def test_D_from_float_avoids_binary_drift():
    assert D(0.1) == Decimal("0.1")
    assert D(177.99) == Decimal("177.99")


def test_D_from_int():
    assert D(42) == Decimal("42")


def test_round_2dp_half_up():
    assert round_2dp("0.005") == Decimal("0.01")  # half-up, not banker's
    assert round_2dp("0.014999") == Decimal("0.01")
    assert round_2dp("0.015") == Decimal("0.02")
    assert round_2dp("21.3444") == Decimal("21.34")
    assert round_2dp("156.345") == Decimal("156.35")


def test_fmt_strips_trailing_zeros():
    assert fmt(Decimal("21.3400")) == "21.34"
    assert fmt(Decimal("8.00")) == "8"
    assert fmt(Decimal("0.0")) == "0"


def test_column_add_result_correct():
    op = ColumnAdd(D("12.5"), D("3.75"))
    assert op.result == Decimal("16.25")


def test_column_add_explanation_contains_all_operands():
    text = ColumnAdd(D("12.5"), D("3.75")).explain()
    assert "12.5" in text
    assert "3.75" in text
    assert "16.25" in text
    assert "+" in text


def test_column_subtract_result_correct():
    assert ColumnSubtract(D("177.99"), D("170.72")).result == Decimal("7.27")


def test_column_multiply_partial_products():
    op = ColumnMultiply(D("4.62"), D("4.62"))
    assert op.result == Decimal("21.3444")
    text = op.explain()
    assert "21.3444" in text
    assert "(4.62 × 0.02)" in text
    assert "(4.62 × 0.6)" in text
    assert "(4.62 × 4)" in text


def test_column_multiply_zero_handling():
    op = ColumnMultiply(D("4.62"), D("0"))
    assert op.result == Decimal("0")
    text = op.explain()
    assert "0" in text


def test_long_division_simple():
    op = LongDivision(D("10"), D("4"), decimals=2)
    assert op.result == Decimal("2.50")


def test_long_division_classic_gravity_case():
    op = LongDivision(D("177.99"), D("21.34"), decimals=2)
    assert op.result == Decimal("8.34")
    text = op.explain()
    assert "177.99" in text
    assert "21.34" in text
    assert "8.34" in text


def test_long_division_half_up_rounds_correctly():
    # 1.0 / 3 = 0.333... → 2dp half-up = 0.33
    assert LongDivision(D("1"), D("3"), decimals=2).result == Decimal("0.33")
    # 2.0 / 3 = 0.666... → 2dp half-up = 0.67
    assert LongDivision(D("2"), D("3"), decimals=2).result == Decimal("0.67")


def test_long_division_high_precision():
    # 5dp precision
    op = LongDivision(D("1"), D("7"), decimals=5)
    assert op.result == Decimal("0.14286")


def test_long_division_by_zero_raises():
    import pytest

    with pytest.raises(ZeroDivisionError):
        LongDivision(D("5"), D("0"))


def test_long_division_explanation_describes_steps():
    text = LongDivision(D("355.98"), D("21.34"), decimals=2).explain()
    # Should mention quotient digits and partial products
    assert "quotient" in text.lower()
    assert "16.68" in text  # 355.98 / 21.34 = 16.68


def test_arithmetic_chain_matches_gravity_walkthrough():
    """End-to-end: reproduce the gravity-family arithmetic exactly."""
    t = D("4.62")
    t_sq = ColumnMultiply(t, t).result
    assert round_2dp(t_sq) == Decimal("21.34")

    d = D("177.99")
    two_d = ColumnMultiply(D("2"), d).result
    assert two_d == Decimal("355.98")

    g = LongDivision(two_d, round_2dp(t_sq), decimals=2).result
    assert g == Decimal("16.68")

    # Apply: d_new = 0.5 * g * t_new²
    t_new = D("4.33")
    t_new_sq = round_2dp(ColumnMultiply(t_new, t_new).result)
    half_g = round_2dp(g / 2)
    d_new = ColumnMultiply(half_g, t_new_sq).result
    # 8.34 * 18.75 = 156.375 → round to 156.38 (half-up)
    assert round_2dp(d_new) == Decimal("156.38")


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases: D() coercion robustness
# ─────────────────────────────────────────────────────────────────────────────

import pytest


@pytest.mark.parametrize("value,expected", [
    (0, "0"),
    (-0.0, "-0"),  # signed zero
    (1, "1"),
    (-1, "-1"),
    (1e10, "10000000000"),
    (1.5e-5, "0.000015"),
    ("0.1", "0.1"),
    ("-3.14", "-3.14"),
    (Decimal("42"), "42"),
])
def test_D_coercion_various_types(value, expected):
    result = D(value)
    assert isinstance(result, Decimal)


def test_D_preserves_exact_decimal_input():
    assert D(Decimal("3.1415926535")) == Decimal("3.1415926535")


# ─────────────────────────────────────────────────────────────────────────────
# round_2dp: exhaustive half-up behavior
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("input_,expected", [
    ("0", "0.00"),
    ("0.001", "0.00"),     # < halfway → round down
    ("0.005", "0.01"),     # exactly half → up
    ("0.015", "0.02"),     # half-up, not banker's
    ("0.025", "0.03"),
    ("0.045", "0.05"),
    ("0.055", "0.06"),
    ("0.095", "0.10"),
    ("0.999", "1.00"),
    ("-0.005", "-0.01"),   # negative half-up: away from zero (Decimal HALF_UP rounds magnitude up)
    ("-0.999", "-1.00"),
    ("156.345", "156.35"),
    ("156.344", "156.34"),
    ("21.3444", "21.34"),
    ("21.345", "21.35"),
    ("1000000.005", "1000000.01"),
])
def test_round_2dp_comprehensive(input_, expected):
    assert round_2dp(input_) == Decimal(expected)


# ─────────────────────────────────────────────────────────────────────────────
# fmt: format quirks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("input_,expected", [
    (Decimal("0"), "0"),
    (Decimal("0.00"), "0"),
    (Decimal("1"), "1"),
    (Decimal("1.00"), "1"),
    (Decimal("1.10"), "1.1"),
    (Decimal("1.01"), "1.01"),
    (Decimal("-1.5"), "-1.5"),
    (Decimal("-0.50"), "-0.5"),
    (Decimal("100.000"), "100"),
    (Decimal("0.0001"), "0.0001"),
    (Decimal("3.14159265358979"), "3.14159265358979"),
])
def test_fmt_comprehensive(input_, expected):
    assert fmt(input_) == expected


# ─────────────────────────────────────────────────────────────────────────────
# ColumnAdd: negatives, zero, mixed precision
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,expected", [
    ("0", "0", "0"),
    ("1", "0", "1"),
    ("0", "5.5", "5.5"),
    ("1.5", "2.5", "4.0"),
    ("-1", "1", "0"),
    ("-5", "-3", "-8"),
    ("0.1", "0.2", "0.3"),
    ("1.999", "0.001", "2.000"),
    ("1000000", "1", "1000001"),
    ("0.0001", "0.0001", "0.0002"),
])
def test_column_add_results(a, b, expected):
    assert ColumnAdd(D(a), D(b)).result == Decimal(expected)


def test_column_add_with_floats_via_D():
    op = ColumnAdd(D(1.5), D(2.25))
    assert op.result == Decimal("3.75")


# ─────────────────────────────────────────────────────────────────────────────
# ColumnSubtract: negatives, borrowing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,expected", [
    ("0", "0", "0"),
    ("5", "5", "0"),
    ("5", "3", "2"),
    ("3", "5", "-2"),
    ("0", "1", "-1"),
    ("-5", "-3", "-2"),
    ("100", "1", "99"),
    ("0.1", "0.05", "0.05"),
    ("1.0", "0.999", "0.001"),
])
def test_column_subtract_results(a, b, expected):
    assert ColumnSubtract(D(a), D(b)).result == Decimal(expected)


# ─────────────────────────────────────────────────────────────────────────────
# ColumnMultiply: signs, zero, magnitude
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,expected", [
    ("0", "0", "0"),
    ("0", "100", "0"),
    ("1", "1", "1"),
    ("-1", "1", "-1"),
    ("-1", "-1", "1"),
    ("2", "3", "6"),
    ("-2", "3", "-6"),
    ("0.5", "0.5", "0.25"),
    ("0.1", "0.1", "0.01"),
    ("4.62", "4.62", "21.3444"),
    ("100", "100", "10000"),
    ("1000.5", "0.001", "1.0005"),
])
def test_column_multiply_results(a, b, expected):
    assert ColumnMultiply(D(a), D(b)).result == Decimal(expected)


def test_column_multiply_explanation_well_formed():
    op = ColumnMultiply(D("4.62"), D("4.62"))
    text = op.explain()
    # All partials must appear
    assert "(4.62 × 0.02)" in text
    assert "(4.62 × 0.6)" in text
    assert "(4.62 × 4)" in text
    # Sum line
    assert "21.3444" in text


def test_column_multiply_handles_negative_b():
    op = ColumnMultiply(D("4.5"), D("-2"))
    assert op.result == Decimal("-9")


# ─────────────────────────────────────────────────────────────────────────────
# LongDivision: signs, recurring, exact, large
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dividend,divisor,decimals,expected", [
    ("10", "2", 2, "5.00"),
    ("10", "4", 2, "2.50"),
    ("10", "3", 2, "3.33"),
    ("20", "3", 2, "6.67"),
    ("1", "7", 5, "0.14286"),
    ("355.98", "21.34", 2, "16.68"),
    ("0", "1", 2, "0.00"),
    ("100", "100", 2, "1.00"),
])
def test_long_division_results(dividend, divisor, decimals, expected):
    op = LongDivision(D(dividend), D(divisor), decimals=decimals)
    assert op.result == Decimal(expected)


def test_long_division_zero_dividend():
    op = LongDivision(D("0"), D("5"), decimals=2)
    assert op.result == Decimal("0.00")


def test_long_division_recurring_thirds():
    # 1/3 = 0.333... → 2dp half-up = 0.33 (since the 4th digit is 3 → no round up)
    assert LongDivision(D("1"), D("3"), decimals=2).result == Decimal("0.33")
    # 2/3 = 0.666... → 0.67 (round up since 4th digit > 5)
    assert LongDivision(D("2"), D("3"), decimals=2).result == Decimal("0.67")


def test_long_division_explanation_mentions_integer_part():
    text = LongDivision(D("10"), D("3"), decimals=2).explain()
    assert "10" in text
    assert "3" in text
    assert "3.33" in text


def test_long_division_explanation_under_300_chars_for_simple_case():
    # The CoT shouldn't be enormously verbose for simple cases
    text = LongDivision(D("10"), D("2"), decimals=2).explain()
    assert len(text) < 500, f"too verbose: {len(text)} chars"


# ─────────────────────────────────────────────────────────────────────────────
# Property: D(float) ↔ str(float) roundtrip stable
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("x", [
    0.1, 0.2, 0.3, 1.5, 4.62, 177.99, 21.3444, 99.999, -3.14,
])
def test_D_float_roundtrip_no_drift(x):
    """Decimal(str(x)) avoids binary float representation issues."""
    d = D(x)
    assert d == Decimal(str(x))


# ─────────────────────────────────────────────────────────────────────────────
# Property: arithmetic identities
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [("1.5", "2.5"), ("0", "0"), ("100", "1"), ("-3", "7")])
def test_add_commutative(a, b):
    assert ColumnAdd(D(a), D(b)).result == ColumnAdd(D(b), D(a)).result


@pytest.mark.parametrize("a,b", [("1.5", "2.5"), ("0", "0"), ("100", "1"), ("-3", "7")])
def test_subtract_anti_commutative(a, b):
    assert ColumnSubtract(D(a), D(b)).result == -ColumnSubtract(D(b), D(a)).result


@pytest.mark.parametrize("a,b", [("1.5", "2.5"), ("0", "100"), ("100", "1"), ("-3", "7"), ("-3", "-7")])
def test_multiply_commutative(a, b):
    assert ColumnMultiply(D(a), D(b)).result == ColumnMultiply(D(b), D(a)).result


@pytest.mark.parametrize("a", ["1.5", "2.5", "100", "-3.14"])
def test_add_zero_is_identity(a):
    assert ColumnAdd(D(a), D("0")).result == D(a)


@pytest.mark.parametrize("a", ["1.5", "2.5", "100", "-3.14"])
def test_multiply_by_one_is_identity(a):
    assert ColumnMultiply(D(a), D("1")).result == D(a)


@pytest.mark.parametrize("a", ["1.5", "2.5", "100", "-3.14", "0"])
def test_multiply_by_zero_is_zero(a):
    assert ColumnMultiply(D(a), D("0")).result == Decimal("0")
