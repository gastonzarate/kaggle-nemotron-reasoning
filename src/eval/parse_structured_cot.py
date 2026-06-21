"""Parser + verifier for the structured-CoT v1 schema.

See data/synthetic/schema/SCHEMA.md for the full spec.

Public API:
    parse_structured_cot(text: str) -> ParsedCoT
    verify_against_gold(parsed: ParsedCoT, gold_answer: str, family: str)
        -> tuple[bool, str | None]

`parse_structured_cot` is tolerant: it returns a ParsedCoT with `agree=False`
and possibly empty fields rather than raising, unless the input is so
malformed that no sub-block can be extracted (in which case it raises
ValueError).

`verify_against_gold` applies family-specific normalisation rules
(numeric tolerance for gravity/unit_conversion; trailing-`\\` stripping
for cryptarithm_*; zero-padding for equation_numeric_guess; byte-exact
otherwise).
"""

from __future__ import annotations

import dataclasses
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

EXPECTED_TAG_ORDER: dict[str, tuple[str, ...]] = {
    "bit_manipulation": ("c", "c"),
    "cipher": ("c", "l", "c"),  # the <l> block is optional; see CIPHER_ALLOWS_NO_L
    "cryptarithm_deduce": ("c", "c"),
    "cryptarithm_guess": ("c", "c"),
    "equation_numeric_deduce": ("c", "m"),
    "equation_numeric_guess": ("c", "m"),
    "gravity": ("m", "m"),
    "numeral": ("c", "m"),  # also accept ("c", "c"); see NUMERAL_ALLOWS_C2 below
    "unit_conversion": ("m", "m"),
}

# Families with relaxed tag-order: cipher may skip <l> if no unknown letters,
# numeral may use <c> instead of <m> for the apply step.
CIPHER_ALLOWS_NO_L = True
NUMERAL_ALLOWS_C2 = True

NUMERIC_TOLERANCE = Decimal("0.005")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ParsedCoT:
    math_steps: list[str]
    code_steps: list[str]
    ling_steps: list[str]
    tag_order: list[str]
    final_line: Optional[str]
    boxed: Optional[str]
    agree: bool

    def as_dict(self) -> dict:
        return {
            "math_steps": self.math_steps,
            "code_steps": self.code_steps,
            "ling_steps": self.ling_steps,
            "tag_order": self.tag_order,
            "final_line": self.final_line,
            "boxed": self.boxed,
            "agree": self.agree,
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TAG_RE = re.compile(r"<([mcl])>(.*?)</\1>", re.DOTALL)
_FINAL_RE = re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.MULTILINE)


def _extract_last_boxed(text: str) -> Optional[str]:
    """Pull the *last* \\boxed{...} payload with brace-balanced parsing.

    Returns the content between the matching braces. If the content ends
    with an odd number of backslashes (i.e. the closing `}` was escaped),
    we still return the payload up to the matched brace — the family-level
    normaliser handles the brace-safe convention.
    """
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        # Unbalanced — return everything up to end of string.
        return text[start:]
    return text[start : i - 1]


def parse_structured_cot(text: str) -> ParsedCoT:
    """Parse a structured-CoT v1 string into its sub-blocks.

    Tolerates missing tags but always sets `agree` correctly. Raises
    ValueError only if there is no <think> block AND no \\boxed at all.
    """
    if not text or not text.strip():
        raise ValueError("empty input")

    think_match = _THINK_RE.search(text)
    inner = think_match.group(1) if think_match else ""

    math_steps: list[str] = []
    code_steps: list[str] = []
    ling_steps: list[str] = []
    tag_order: list[str] = []

    for m in _TAG_RE.finditer(inner):
        tag = m.group(1)
        body = m.group(2).strip()
        tag_order.append(tag)
        if tag == "m":
            math_steps.append(body)
        elif tag == "c":
            code_steps.append(body)
        elif tag == "l":
            ling_steps.append(body)

    # Find the LAST FINAL: line across all blocks (per spec, it lives in the
    # last block, but the regex is global so we take the last match).
    final_matches = list(_FINAL_RE.finditer(inner))
    final_line = final_matches[-1].group(1).strip() if final_matches else None

    boxed_raw = _extract_last_boxed(text)
    boxed = boxed_raw.strip() if boxed_raw is not None else None

    if think_match is None and boxed is None:
        raise ValueError("input has neither <think> nor \\boxed{}")

    # Cross-check: do FINAL and boxed agree byte-for-byte (after .strip())?
    if final_line is None or boxed is None:
        agree = False
    else:
        agree = _strip_brace_safe(final_line) == _strip_brace_safe(boxed)

    return ParsedCoT(
        math_steps=math_steps,
        code_steps=code_steps,
        ling_steps=ling_steps,
        tag_order=tag_order,
        final_line=final_line,
        boxed=boxed,
        agree=agree,
    )


def _strip_brace_safe(s: str) -> str:
    """Apply the brace-safe normalisation: drop trailing whitespace AND
    drop a trailing '}' that would have come from an unescaped backslash
    immediately before the boxed closer."""
    s = s.rstrip()
    # If the string ends with '\\}' (literal backslash + closing brace), the
    # boxed payload was contaminated; strip the spurious '}'.
    if s.endswith("\\}"):
        s = s[:-1]
    return s


# ---------------------------------------------------------------------------
# Family-specific verification
# ---------------------------------------------------------------------------


def verify_against_gold(
    parsed: ParsedCoT,
    gold_answer: str,
    family: str,
) -> tuple[bool, Optional[str]]:
    """Return (ok, reason). Reason is None when ok=True."""
    if parsed.boxed is None:
        return False, "no_boxed"
    if parsed.final_line is None:
        return False, "no_final_line"
    if not parsed.agree:
        return False, "final_vs_boxed_disagree"
    if not _tag_order_ok(parsed.tag_order, family):
        return False, f"tag_order:{parsed.tag_order}"

    answer = _strip_brace_safe(parsed.boxed)
    gold = gold_answer.strip()

    if family in ("gravity", "unit_conversion"):
        ok = _numeric_close(answer, gold, NUMERIC_TOLERANCE)
        return (True, None) if ok else (False, f"numeric_mismatch:{answer}!={gold}")

    if family in ("cryptarithm_deduce", "cryptarithm_guess"):
        if answer.rstrip() == gold.rstrip():
            return True, None
        return False, f"cryptarithm_mismatch:{answer!r}!={gold!r}"

    if family == "equation_numeric_guess":
        # Width-aware exact match: the gold determines the canonical width;
        # answer must match byte-exact. A leading-zero strip mismatch is
        # the documented bug we want to *reject* (so the model learns to pad).
        if answer == gold:
            return True, None
        # Detect the specific zero-padding bug to emit a useful reason
        if answer.lstrip("0") == gold.lstrip("0") and answer != gold:
            return False, f"zero_pad_required:{answer}->{gold}"
        return False, f"eq_guess_mismatch:{answer}!={gold}"

    # default: byte-exact
    if answer == gold:
        return True, None
    return False, f"mismatch:{answer!r}!={gold!r}"


def _tag_order_ok(observed: list[str], family: str) -> bool:
    expected = EXPECTED_TAG_ORDER.get(family)
    if expected is None:
        return False
    if tuple(observed) == expected:
        return True
    # cipher relaxation
    if family == "cipher" and CIPHER_ALLOWS_NO_L and tuple(observed) == ("c", "c"):
        return True
    # numeral relaxation
    if family == "numeral" and NUMERAL_ALLOWS_C2 and tuple(observed) == ("c", "c"):
        return True
    return False


def _numeric_close(a: str, b: str, tol: Decimal) -> bool:
    try:
        da = Decimal(a)
        db = Decimal(b)
    except Exception:
        return False
    return (da - db).copy_abs() <= tol


def round_half_up_2dp(x: float | Decimal) -> str:
    """Helper for the generator: round half-up to 2 dp, return formatted str."""
    d = Decimal(str(x)) if not isinstance(x, Decimal) else x
    return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Self-tests (run with `python src/eval/parse_structured_cot.py`)
# ---------------------------------------------------------------------------


def _test_parse_numeral_ok() -> None:
    text = (
        "<think>\n<c>\ndetect roman\n</c>\n"
        "<m>\napply\nFINAL: LX\n</m>\n"
        "</think>\n\\boxed{LX}\n"
    )
    p = parse_structured_cot(text)
    assert p.tag_order == ["c", "m"]
    assert p.boxed == "LX"
    assert p.final_line == "LX"
    assert p.agree is True
    ok, reason = verify_against_gold(p, "LX", "numeral")
    assert ok, reason


def _test_parse_disagree() -> None:
    text = (
        "<think>\n<c>\nstuff\nFINAL: 11001101\n</c>\n</think>\n"
        "\\boxed{11001100}\n"
    )
    p = parse_structured_cot(text)
    assert p.agree is False
    ok, reason = verify_against_gold(p, "11001101", "bit_manipulation")
    assert not ok
    assert reason == "final_vs_boxed_disagree"


def _test_gravity_tolerance() -> None:
    text = (
        "<think>\n<m>\nfit k\n</m>\n<m>\napply\nFINAL: 156.35\n</m>\n</think>\n"
        "\\boxed{156.35}\n"
    )
    p = parse_structured_cot(text)
    ok, reason = verify_against_gold(p, "156.35", "gravity")
    assert ok, reason
    # Within tolerance
    text2 = text.replace("156.35", "156.347", 1).replace(
        "\\boxed{156.35}", "\\boxed{156.347}"
    )
    p2 = parse_structured_cot(text2)
    ok2, _ = verify_against_gold(p2, "156.35", "gravity")
    assert ok2  # within ±0.005


def _test_cryptarithm_trailing_backslash() -> None:
    # Simulates the LaTeX escape bug: \boxed{$>>\}
    text = (
        "<think>\n<c>\nconcat\n</c>\n<c>\napply\nFINAL: $>>\\\n</c>\n</think>\n"
        "\\boxed{$>>\\}"
    )
    p = parse_structured_cot(text)
    # final_line in this synthetic example is "$>>\\"; brace-safe stripping
    # of boxed payload should yield "$>>\\" too. They should agree.
    assert p.agree, (p.final_line, p.boxed)
    ok, reason = verify_against_gold(p, "$>>\\", "cryptarithm_deduce")
    assert ok, reason


def _test_equation_numeric_guess_pad() -> None:
    text = (
        "<think>\n<c>\nfallback abs_diff\n</c>\n"
        "<m>\nabs_diff = 3 -> zero-pad to 03\nFINAL: 03\n</m>\n</think>\n"
        "\\boxed{03}\n"
    )
    p = parse_structured_cot(text)
    ok, reason = verify_against_gold(p, "03", "equation_numeric_guess")
    assert ok, reason

    # Unpadded version of the same — should fail with the zero_pad_required hint
    text_bad = text.replace("03", "3")
    p2 = parse_structured_cot(text_bad)
    ok2, reason2 = verify_against_gold(p2, "03", "equation_numeric_guess")
    assert not ok2
    assert reason2 and reason2.startswith("zero_pad_required")


def _test_cipher_optional_l() -> None:
    # No <l> block — should still be accepted
    text = (
        "<think>\n<c>\nmap built\n</c>\n<c>\napply map\nFINAL: knight\n</c>\n</think>\n"
        "\\boxed{knight}\n"
    )
    p = parse_structured_cot(text)
    ok, reason = verify_against_gold(p, "knight", "cipher")
    assert ok, reason


def _test_round_half_up() -> None:
    assert round_half_up_2dp(10.2547) == "10.25"
    assert round_half_up_2dp(26.1075) == "26.11"
    assert round_half_up_2dp(156.347) == "156.35"


def _run_all() -> None:
    _test_parse_numeral_ok()
    _test_parse_disagree()
    _test_gravity_tolerance()
    _test_cryptarithm_trailing_backslash()
    _test_equation_numeric_guess_pad()
    _test_cipher_optional_l()
    _test_round_half_up()
    print("parse_structured_cot.py self-tests OK")


if __name__ == "__main__":  # pragma: no cover
    _run_all()
