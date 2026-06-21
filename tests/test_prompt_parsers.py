from decimal import Decimal

from src.data.prompt_parsers import (
    parse_bit_manipulation,
    parse_cipher,
    parse_gravity,
    parse_numeral,
    parse_transformation,
    parse_unit_conversion,
)


def test_parse_gravity_extracts_pairs_and_query():
    p = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        "For t = 4.62s, distance = 177.99 m\n"
        "For t = 2.97s, distance = 73.56 m\n"
        "Now, determine the falling distance for t = 4.33s given d = 0.5*g*t^2."
    )
    out = parse_gravity(p)
    assert out is not None
    assert out["pairs"] == [(Decimal("4.62"), Decimal("177.99")), (Decimal("2.97"), Decimal("73.56"))]
    assert out["query_t"] == Decimal("4.33")


def test_parse_unit_conversion():
    p = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
        "18.75 m becomes 19.91\n"
        "46.61 m becomes 49.48\n"
        "Now, convert the following measurement: 15.19 m"
    )
    out = parse_unit_conversion(p)
    assert out is not None
    assert out["pairs"][0] == (Decimal("18.75"), Decimal("19.91"))
    assert out["query_x"] == Decimal("15.19")


def test_parse_numeral():
    p = (
        "In Alice's Wonderland, numbers are secretly converted into a different numeral system.\n"
        "3 -> III\n"
        "29 -> XXIX\n"
        "Now, write the number 43 in the Wonderland numeral system."
    )
    out = parse_numeral(p)
    assert out is not None
    assert out["pairs"] == [(3, "III"), (29, "XXIX")]
    assert out["query_n"] == 43


def test_parse_cipher():
    p = (
        "In Alice's Wonderland, secret encryption rules are used on text.\n"
        "wcjz ivufex -> king dreams\n"
        "ivfznj tvufyux -> dragon creates\n"
        "Now, decrypt the following text: ysu fjtcujy"
    )
    out = parse_cipher(p)
    assert out is not None
    assert out["pairs"] == [("wcjz ivufex", "king dreams"), ("ivfznj tvufyux", "dragon creates")]
    assert out["query"] == "ysu fjtcujy"


def test_parse_bit_manipulation():
    p = (
        "Here are some examples of input -> output:\n"
        "01010001 -> 11011101\n"
        "00001001 -> 01101101\n"
        "Now, determine the output for: 11111110"
    )
    out = parse_bit_manipulation(p)
    assert out is not None
    assert out["pairs"] == [("01010001", "11011101"), ("00001001", "01101101")]
    assert out["query"] == "11111110"


def test_parse_transformation():
    p = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations.\n"
        "$?:>` = $?>`\n"
        "{`:?> = {`?>\n"
        "Now, determine the result for: $>:>\\"
    )
    out = parse_transformation(p)
    assert out is not None
    assert out["pairs"][0] == ("$?:>`", "$?>`")
    assert out["query"] == "$>:>\\"


def test_parse_returns_none_when_no_match():
    assert parse_gravity("totally unrelated") is None
    assert parse_unit_conversion("just words") is None


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases: parser robustness against real-world prompt variations
# ─────────────────────────────────────────────────────────────────────────────

import pytest


def test_parse_gravity_handles_no_s_after_t():
    p = "For t = 4.5, distance = 50 m\nNow, determine the falling distance for t = 3.0 given d = 0.5*g*t^2."
    out = parse_gravity(p)
    assert out is not None
    assert out["pairs"] == [(Decimal("4.5"), Decimal("50"))]


def test_parse_gravity_handles_many_pairs():
    pairs_text = "\n".join(f"For t = {i}.0s, distance = {i*i*10}.0 m" for i in range(1, 11))
    p = pairs_text + "\nNow, determine the falling distance for t = 5.5s given d = 0.5*g*t^2."
    out = parse_gravity(p)
    assert out is not None
    assert len(out["pairs"]) == 10
    assert out["query_t"] == Decimal("5.5")


def test_parse_gravity_integer_only_values():
    p = "For t = 2s, distance = 20 m\nNow, determine the falling distance for t = 3s given d=0.5*g*t^2."
    out = parse_gravity(p)
    assert out is not None
    assert out["query_t"] == Decimal("3")


def test_parse_unit_conversion_decimal_precision():
    p = (
        "0.001 m becomes 0.00106\n"
        "1000.0 m becomes 1062.0\n"
        "Now, convert the following measurement: 5.123456 m"
    )
    out = parse_unit_conversion(p)
    assert out is not None
    assert out["query_x"] == Decimal("5.123456")


@pytest.mark.parametrize("n,roman", [
    (1, "I"), (4, "IV"), (9, "IX"), (49, "XLIX"), (100, "C"),
    (500, "D"), (999, "CMXCIX"), (1000, "M"),
])
def test_parse_numeral_various_values(n, roman):
    p = f"{n} -> {roman}\nNow, write the number 50 in the Wonderland numeral system."
    out = parse_numeral(p)
    assert out is not None
    assert (n, roman) in out["pairs"]


def test_parse_cipher_preserves_case_lower():
    # All examples lowercase as per the train.csv format
    p = (
        "abc def -> xyz qrs\n"
        "ghi -> tuv\n"
        "Now, decrypt the following text: abc"
    )
    out = parse_cipher(p)
    assert out is not None
    assert out["query"] == "abc"


def test_parse_bit_manipulation_all_zeros_and_ones():
    p = (
        "00000000 -> 11111111\n"
        "11111111 -> 00000000\n"
        "Now, determine the output for: 10101010"
    )
    out = parse_bit_manipulation(p)
    assert out is not None
    assert out["query"] == "10101010"
    assert ("00000000", "11111111") in out["pairs"]


def test_parse_bit_manipulation_rejects_non_8_bit():
    # Should not match strings that aren't exactly 8 bits
    p = "0101 -> 1010\nNow, determine the output for: 0000"
    out = parse_bit_manipulation(p)
    # No 8-bit pairs → None
    assert out is None


def test_parse_bit_manipulation_rejects_non_binary():
    p = "010100ab -> 11011101\nNow, determine the output for: 11111110"
    out = parse_bit_manipulation(p)
    assert out is None


def test_parse_transformation_with_special_chars():
    p = (
        "$?:>` = $?>`\n"
        "\\<:\\` = \\<\\`\n"
        "Now, determine the result for: $>:>\\"
    )
    out = parse_transformation(p)
    assert out is not None
    assert ("$?:>`", "$?>`") in out["pairs"]
    assert out["query"] == "$>:>\\"


def test_parse_transformation_empty_when_only_intro():
    p = "In Alice's Wonderland, mysterious things happen.\nNow, determine the result for: x"
    # No "X = Y" pairs → None
    out = parse_transformation(p)
    assert out is None


@pytest.mark.parametrize("garbage", ["", "   ", "\n\n", "random text without structure"])
def test_all_parsers_return_none_on_garbage(garbage):
    assert parse_gravity(garbage) is None
    assert parse_unit_conversion(garbage) is None
    assert parse_numeral(garbage) is None
    assert parse_cipher(garbage) is None
    assert parse_bit_manipulation(garbage) is None
    assert parse_transformation(garbage) is None


def test_parsers_dont_crash_on_special_chars():
    # The cot-tong dataset has prompts with weird unicode and special chars in some rows
    p = "🎁 In Alice's Wonderland 🎁 t = 1.0s, distance = 5.0 m\nNow, determine the falling distance for t = 2.0s given d=0.5*g*t^2."
    out = parse_gravity(p)
    assert out is not None  # should still extract the math
