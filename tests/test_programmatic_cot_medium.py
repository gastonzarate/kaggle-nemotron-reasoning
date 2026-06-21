"""Tests for the medium-tier programmatic CoTs (cipher, bit_manipulation)."""

import re

from src.data.programmatic_cot import (
    bit_manipulation_cot,
    build_substitution_map,
    cipher_cot,
    discover_bit_rules,
)


def _extract_boxed(cot: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", cot)
    return m[-1].strip() if m else ""


def test_build_substitution_map_simple():
    pairs = [
        ("abc", "xyz"),
        ("ab", "xy"),
    ]
    m = build_substitution_map(pairs)
    assert m == {"a": "x", "b": "y", "c": "z"}


def test_build_substitution_map_keeps_first_on_conflict():
    pairs = [
        ("abc", "xyz"),
        ("abd", "xyq"),  # 'd' is new, but 'c' was already mapped; no conflict here for c
    ]
    m = build_substitution_map(pairs)
    assert m["a"] == "x"
    assert m["c"] == "z"
    assert m["d"] == "q"


def test_cipher_cot_simple_alphabet_shift():
    # Simple cipher: 'a'→'k', 'b'→'i', 'c'→'n', 'd'→'g' (KING)
    pairs = [
        ("ab", "ki"),
        ("cd", "ng"),
        ("abcd", "king"),
    ]
    answer, cot = cipher_cot(pairs, "ab cd")
    boxed = _extract_boxed(cot)
    assert boxed == "ki ng"
    assert "<l>" in cot
    assert cot.count("</think>") == 1


def test_cipher_cot_with_realistic_wonderland_example():
    pairs = [
        ("wcjz ivufex", "king dreams"),
        ("ivfznj tvufyux efq", "dragon creates map"),
    ]
    answer, cot = cipher_cot(pairs, "wcjz")
    boxed = _extract_boxed(cot)
    assert boxed == "king"


def test_discover_bit_rules_identity():
    # Output equals input (identity): each out[i] = in[i]
    # Use 8 single-bit-set patterns so each bit position is independently identifiable.
    pairs = [(f"{1 << i:08b}", f"{1 << i:08b}") for i in range(8)]
    pairs.append(("11111111", "11111111"))
    rules = discover_bit_rules(pairs)
    for i, rule in enumerate(rules):
        assert rule is not None, f"no rule for bit {i}"
        assert rule[0] == f"in[{i}]", f"expected diagonal in[{i}], got {rule[0]}"


def test_discover_bit_rules_invert():
    # Output is bitwise NOT of input
    pairs = [(f"{1 << i:08b}", f"{(~(1 << i)) & 0xFF:08b}") for i in range(8)]
    pairs.append(("11111111", "00000000"))
    rules = discover_bit_rules(pairs)
    for i, rule in enumerate(rules):
        assert rule is not None
        assert rule[0] == f"NOT in[{i}]", f"expected NOT in[{i}], got {rule[0]}"


def test_bit_manipulation_cot_identity():
    pairs = [(f"{1 << i:08b}", f"{1 << i:08b}") for i in range(8)]
    pairs.append(("11111111", "11111111"))
    answer, cot = bit_manipulation_cot(pairs, "11001100")
    boxed = _extract_boxed(cot)
    assert boxed == "11001100"
    assert "<c>" in cot
    assert cot.count("</think>") == 1


def test_bit_manipulation_cot_xor_with_constant():
    # XOR each input with 10101010 = flip bits at positions 1, 3, 5, 7
    # Use 8 single-bit-set inputs (plus all-ones) to fully disambiguate
    pairs = []
    for i in range(8):
        inp = 1 << i
        out = inp ^ 0b10101010
        pairs.append((f"{inp:08b}", f"{out:08b}"))
    pairs.append(("11111111", f"{0xFF ^ 0b10101010:08b}"))
    answer, cot = bit_manipulation_cot(pairs, f"{42:08b}")
    boxed = _extract_boxed(cot)
    assert boxed == f"{42 ^ 0b10101010:08b}"
