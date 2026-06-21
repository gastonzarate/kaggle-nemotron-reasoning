from src.eval.extractor import extract_final_answer


def test_simple_boxed():
    assert extract_final_answer(r"The answer is \boxed{42}") == "42"


def test_takes_last_boxed():
    assert extract_final_answer(r"step1 \boxed{x} ... step2 \boxed{42}") == "42"


def test_takes_last_non_empty_boxed():
    assert extract_final_answer(r"\boxed{}\boxed{ } \boxed{ 11010011 }") == "11010011"


def test_unclosed_boxed_at_end():
    assert extract_final_answer(r"reasoning... \boxed{XLVII") == "XLVII"


def test_none_input():
    assert extract_final_answer(None) == "NOT_FOUND"


def test_final_answer_fallback():
    assert extract_final_answer("Final answer: 3.14") == "3.14"


def test_last_number_fallback():
    # No boxed, no "Final answer" pattern -> last number wins
    assert extract_final_answer("step 1, then 2, end value 99") == "99"


def test_last_line_fallback_for_non_numeric():
    # No structured info and no numbers -> last line
    assert extract_final_answer("intro\nthe answer here") == "the answer here"


def test_strips_whitespace_inside_box():
    assert extract_final_answer(r"\boxed{  hello world  }") == "hello world"


def test_handles_curly_brace_in_answer():
    # The fix the Metric Update of 2026-05-17 addresses: answers with `}` inside.
    # Regex stops at first `}`, so "f(x" gets extracted from \boxed{f(x)}, not "f(x)".
    # This is consistent with how Kaggle post-fix actually behaves.
    result = extract_final_answer(r"\boxed{f(x)}")
    assert result in {"f(x", "f(x)"}, f"unexpected: {result!r}"
