from src.eval.verifier import verify


def test_exact_string_match():
    assert verify("11010011", "11010011") is True


def test_string_case_insensitive():
    assert verify("XLVII", "xlvii") is True


def test_string_mismatch():
    assert verify("hello", "goodbye") is False


def test_float_within_rel_tolerance():
    # 134.5 vs 134.0 → 0.37% difference, within 1e-2 rel_tol
    assert verify("134.5", "134.0") is True


def test_float_outside_rel_tolerance():
    # 134.5 vs 130.0 → 3.3% difference, outside 1e-2 rel_tol
    assert verify("134.5", "130.0") is False


def test_float_abs_tolerance_for_near_zero():
    # rel_tol fails for 0.0 vs 0.000001 (relative inf), but abs_tol 1e-5 catches it
    assert verify("0.0", "0.000001") is True


def test_float_vs_string_falls_back_to_string_compare():
    # 42 vs "forty-two" → float parse fails, string compare fails
    assert verify("42", "forty-two") is False


def test_string_vs_string_with_whitespace():
    assert verify("  XLVII  ", "XLVII") is True


def test_one_float_one_string_falls_back_to_string():
    # If only one is parseable as float, the try block raises and falls to string compare
    assert verify("42", "answer:42") is False
