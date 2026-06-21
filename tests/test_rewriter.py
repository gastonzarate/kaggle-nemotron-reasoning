from src.data.rewriter import (
    build_structured_cot,
    clean_cot_body,
    fix_precision_for_numeric_family,
    strip_existing_boxed,
)


def test_strip_existing_boxed():
    cot = "reasoning then\n\\boxed{99} done"
    assert strip_existing_boxed(cot) == "reasoning then\n done"


def test_round_precision_2dp_half_up():
    cot = "k = 21.3444 m → d = 134.499 m"
    fixed = fix_precision_for_numeric_family(cot)
    assert "21.34" in fixed
    assert "134.50" in fixed
    assert "21.3444" not in fixed
    assert "134.499" not in fixed


def test_no_change_for_2dp():
    cot = "k = 9.81 m"
    assert fix_precision_for_numeric_family(cot) == "k = 9.81 m"


def test_clean_cot_body_strips_box_and_rounds():
    cot = "k = 21.3444 m\n\\boxed{134.5}"
    out = clean_cot_body(cot, "gravity")
    assert "\\boxed" not in out
    assert "21.34" in out


def test_build_structured_cot_has_think_and_router_and_tag():
    s = build_structured_cot(
        prompt="In Alice's Wonderland, gravity...",
        cot="k = 9.81\nd = 0.5*g*t^2 = 134.5",
        family="gravity",
        answer="134.5",
    )
    assert s.startswith("<think>")
    assert s.endswith("\\boxed{134.5}")  # new contract: \\boxed AFTER </think>
    assert "</think>" in s
    assert s.count("</think>") == 1  # the v0.2.0 double-</think> bug must not return
    assert "<r>family=gravity" in s
    assert "<m>" in s and "</m>" in s
    assert "134.5" in s


def test_bit_manipulation_uses_c_tag():
    s = build_structured_cot(
        prompt="bit manipulation puzzle",
        cot="bit 7 of input...",
        family="bit_manipulation",
        answer="11010011",
    )
    assert "<c>" in s and "</c>" in s
    assert "<m>" not in s
    assert "<l>" not in s


def test_cipher_uses_l_tag():
    s = build_structured_cot(
        prompt="cipher puzzle",
        cot="mapping w -> k...",
        family="cipher",
        answer="the queen",
    )
    assert "<l>" in s and "</l>" in s


def test_unknown_family_falls_back_to_m():
    s = build_structured_cot(
        prompt="weird puzzle",
        cot="some reasoning",
        family="weird_new_family",
        answer="42",
    )
    assert "<m>" in s and "</m>" in s
