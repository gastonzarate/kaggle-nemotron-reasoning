"""Programmatic CoT generators — one per family.

Each function takes the family-specific parsed data and returns a tuple
(answer_str, cot_text). The cot_text is wrapped in the structured
`<think><r>...</r><tag>...</tag></think>\\boxed{answer}` envelope, so it's
directly usable as the SFT target.

The answer is computed by REAL arithmetic / algorithm — no LLM hallucination, no
guess-and-verify. If the produced answer matches the gold, we ship this CoT to
the training set; if not, the row gets discarded (validated by the orchestrator).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from src.data.math_primitives import (
    ColumnAdd,
    ColumnMultiply,
    ColumnSubtract,
    D,
    LongDivision,
    fmt,
    round_2dp,
)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _indent(text: str, spaces: int) -> str:
    """Indent every line of a multi-line walkthrough block."""
    pad = " " * spaces
    return "\n".join(pad + ln if ln else ln for ln in text.split("\n"))


def _wrap(router_text: str, tag: str, body_text: str, answer: str) -> str:
    return (
        f"{THINK_OPEN}\n"
        f"<r>{router_text}</r>\n"
        f"<{tag}>\n"
        f"{body_text}\n"
        f"</{tag}>\n"
        f"{THINK_CLOSE}\n"
        f"\\boxed{{{answer}}}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# gravity:   d = 0.5 · g · t²  →  g = 2d/t²
# ─────────────────────────────────────────────────────────────────────────────

def gravity_cot(
    pairs: list[tuple[Decimal, Decimal]], query_t: Decimal, commit_early: bool = False
) -> tuple[str, str]:
    body = [
        "Concept. The falling distance follows d = 0.5·g·t², where g is a single",
        "hidden constant shared by every observation. Rearranging, g = 2·d / t².",
        "So I will recover g from each (t, d) example by doing the squaring and the",
        "division by hand, confirm the examples agree, then plug the query t into the",
        "same formula. I keep two decimals throughout, matching the answer format.",
        "",
        "Step 1 — recover g from each example.",
    ]

    g_values: list[Decimal] = []
    for i, (t, d) in enumerate(pairs, 1):
        t_sq_op = ColumnMultiply(t, t)
        t_sq = round_2dp(t_sq_op.result)
        two_d = ColumnMultiply(D(2), d).result
        g_op = LongDivision(two_d, t_sq, decimals=2)
        body.append(f"\n  Example {i}: t = {fmt(t)} s, d = {fmt(d)} m.")
        body.append(f"  • t² = {fmt(t)} × {fmt(t)}:")
        body.append(_indent(t_sq_op.explain(), 4))
        body.append(f"    Rounded to 2dp, t² = {fmt(t_sq)}.")
        body.append(f"  • 2·d = 2 × {fmt(d)} = {fmt(two_d)}.")
        body.append(f"  • g = 2d ÷ t² = {fmt(two_d)} ÷ {fmt(t_sq)}:")
        body.append(_indent(g_op.explain(), 4))
        body.append(f"    So g ≈ {fmt(g_op.result)}.")
        g_values.append(g_op.result)

    g_med = sorted(g_values)[len(g_values) // 2]
    body.append(
        f"\nStep 2 — agree on g. The recovered values are "
        f"{', '.join(fmt(g) for g in g_values)}; they converge, so g = {fmt(g_med)}."
    )

    half_g = round_2dp(g_med / 2)
    chk_t, chk_d = pairs[0]
    chk_tsq = round_2dp(ColumnMultiply(chk_t, chk_t).result)
    chk_pred = round_2dp(ColumnMultiply(half_g, chk_tsq).result)
    sanity_check = (
        f"sanity check on Example 1: 0.5·g·t² = {fmt(half_g)} × {fmt(chk_tsq)} "
        f"= {fmt(chk_pred)} m vs the given {fmt(chk_d)} m — agrees within rounding. Good."
    )

    qt_sq_op = ColumnMultiply(query_t, query_t)
    qt_sq = round_2dp(qt_sq_op.result)
    d_result_op = ColumnMultiply(half_g, qt_sq)
    d_result = round_2dp(d_result_op.result)
    answer = fmt(d_result)

    apply_lines = [
        f"apply to the query t = {fmt(query_t)} s.",
        f"  • t² = {fmt(query_t)} × {fmt(query_t)}:",
        _indent(qt_sq_op.explain(), 4),
        f"    Rounded to 2dp, t² = {fmt(qt_sq)}.",
        f"  • 0.5·g = {fmt(g_med)} ÷ 2 = {fmt(half_g)}.",
        f"  • d = 0.5·g · t² = {fmt(half_g)} × {fmt(qt_sq)}:",
        _indent(d_result_op.explain(), 4),
        f"    Rounded to 2dp, d = {fmt(d_result)} m.",
    ]

    if commit_early:
        # Commit-then-verify: box the answer as soon as it is derived, THEN run
        # the sanity check. If generation is cut off mid-verification at eval,
        # the official extractor (last \boxed) still recovers the answer.
        body.append("\nStep 3 — " + apply_lines[0])
        body.extend(apply_lines[1:])
        body.append(f"\nAnswer: \\boxed{{{answer}}}")
        body.append("\nStep 4 — " + sanity_check)
    else:
        body.append("\nStep 3 — " + sanity_check)
        body.append("\nStep 4 — " + apply_lines[0])
        body.extend(apply_lines[1:])
    cot = _wrap(
        "family=gravity; plan: g = 2d/t² recovered by hand from each example, verified, "
        "then applied to the query",
        "m",
        "\n".join(body),
        answer,
    )
    return answer, cot


# ─────────────────────────────────────────────────────────────────────────────
# unit_conversion:   Y = a·X + b  (linear, often b≈0)
# ─────────────────────────────────────────────────────────────────────────────

def unit_conversion_cot(pairs: list[tuple[Decimal, Decimal]], query_x: Decimal) -> tuple[str, str]:
    body = [
        "Concept. The conversion is linear: Y = a·X + b, with a fixed slope a and",
        "offset b. Two (X, Y) points determine the line: the slope is the change in Y",
        "over the change in X, and b is whatever shifts the line to pass through a",
        "point. I find a and b from two examples, verify on the others, then convert",
        "the query.",
        "",
    ]

    if len(pairs) < 2:
        x, y = pairs[0]
        a_op = LongDivision(y, x, decimals=4)
        a = a_op.result
        b = D("0")
        body.append(f"Step 1 — only one example, so assume b = 0 and a = Y/X = {fmt(y)} ÷ {fmt(x)}:")
        body.append(_indent(a_op.explain(), 2))
        body.append(f"  So a = {fmt(a)}, b = 0.")
    else:
        (x1, y1), (x2, y2) = pairs[0], pairs[1]
        if x1 == x2:
            (x2, y2) = pairs[2] if len(pairs) > 2 else pairs[1]
        dy_op = ColumnSubtract(y2, y1)
        dx_op = ColumnSubtract(x2, x1)
        a_op = LongDivision(dy_op.result, dx_op.result, decimals=4)
        a = a_op.result
        ax1_op = ColumnMultiply(a, x1)
        b = round_2dp(y1 - ax1_op.result)
        body.append(
            f"Step 1 — slope a from points ({fmt(x1)}→{fmt(y1)}) and ({fmt(x2)}→{fmt(y2)})."
        )
        body.append(f"  • ΔY = {fmt(y2)} − {fmt(y1)}:")
        body.append(_indent(dy_op.explain(), 4))
        body.append(f"  • ΔX = {fmt(x2)} − {fmt(x1)}:")
        body.append(_indent(dx_op.explain(), 4))
        body.append(f"  • a = ΔY ÷ ΔX = {fmt(dy_op.result)} ÷ {fmt(dx_op.result)}:")
        body.append(_indent(a_op.explain(), 4))
        body.append(f"    So a = {fmt(a)}.")
        body.append(f"\nStep 2 — offset b = Y₁ − a·X₁.")
        body.append(f"  • a·X₁ = {fmt(a)} × {fmt(x1)}:")
        body.append(_indent(ax1_op.explain(), 4))
        body.append(
            f"  • b = {fmt(y1)} − {fmt(ax1_op.result)} = {fmt(round_2dp(y1 - ax1_op.result))} "
            f"(≈ {fmt(b)})."
        )

    # Step 3 — verify on remaining pairs.
    if len(pairs) > 2:
        body.append("\nStep 3 — verify the line on the remaining examples:")
        for x, y in pairs[2:5]:
            pred = round_2dp(a * x + b)
            ok = "✓" if pred == round_2dp(y) else "≈"
            body.append(f"  X = {fmt(x)} → a·X+b = {fmt(a)}×{fmt(x)} + {fmt(b)} = {fmt(pred)} vs {fmt(y)} {ok}")

    # Step 4 — apply to query.
    y_pred_op = ColumnMultiply(a, query_x)
    y_pred = round_2dp(y_pred_op.result + b)
    body.append(f"\nStep 4 — convert the query X = {fmt(query_x)}.")
    body.append(f"  • a·X = {fmt(a)} × {fmt(query_x)}:")
    body.append(_indent(y_pred_op.explain(), 4))
    body.append(
        f"  • Y = a·X + b = {fmt(y_pred_op.result)} + {fmt(b)} = {fmt(y_pred)} "
        f"(rounded to 2dp)."
    )

    answer = fmt(y_pred)
    cot = _wrap(
        "family=unit_conversion; plan: slope a=ΔY/ΔX and offset b from two points, "
        "verified, then applied to the query",
        "m",
        "\n".join(body),
        answer,
    )
    return answer, cot


# ─────────────────────────────────────────────────────────────────────────────
# numeral:  arabic → hidden numeral system
#   Try Roman first (most common), then base-k detection.
# ─────────────────────────────────────────────────────────────────────────────

ROMAN_NUMERALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def to_roman(n: int) -> str:
    if n <= 0:
        return ""
    out = []
    for val, sym in ROMAN_NUMERALS:
        while n >= val:
            out.append(sym)
            n -= val
    return "".join(out)


def to_base_k(n: int, k: int, alphabet: str = "0123456789ABCDEF") -> str:
    if n == 0:
        return alphabet[0]
    digits = []
    while n > 0:
        digits.append(alphabet[n % k])
        n //= k
    return "".join(reversed(digits))


def detect_numeral_system(pairs: list[tuple[int, str]]) -> Optional[tuple[str, int]]:
    """Returns ('roman', None) or ('base_k', k) or None if neither fits."""
    # Try Roman
    if all(to_roman(n) == s for n, s in pairs):
        return ("roman", 0)
    # Try base-k for k in 2..16
    for k in range(2, 17):
        if all(to_base_k(n, k) == s.upper() for n, s in pairs):
            return ("base_k", k)
    return None


def numeral_cot(pairs: list[tuple[int, str]], query_n: int) -> tuple[str, str]:
    detection = detect_numeral_system(pairs)
    if detection is None:
        raise ValueError(f"Could not detect numeral system from pairs: {pairs}")

    system, k = detection
    body = [
        "Concept. The examples map decimal numbers to a hidden numeral system. I first",
        "identify the system by checking which known scheme reproduces every example,",
        "then convert the query with that scheme's algorithm.",
        "",
        "Examples given:",
    ]
    for n, s in pairs:
        body.append(f"  {n} → {s}")

    if system == "roman":
        body.append(
            "\nStep 1 — identify. These are Roman numerals "
            "(I=1, V=5, X=10, L=50, C=100, D=500, M=1000; a smaller symbol before a "
            "larger one subtracts, as in IV=4, IX=9)."
        )
        # Verify on the first example.
        ex_n, ex_s = pairs[0]
        body.append(f"  Check: {ex_n} → {to_roman(ex_n)}, matches the given {ex_s}. ✓")
        answer_str = to_roman(query_n)
        body.append(
            f"\nStep 2 — convert {query_n} by repeatedly taking the largest symbol "
            f"value that fits and subtracting it:"
        )
        n = query_n
        for val, sym in ROMAN_NUMERALS:
            while n >= val:
                body.append(f"  {n} ≥ {val} → take '{sym}', remainder = {n - val}")
                n -= val
        body.append(f"  Reading the symbols in order: {query_n} = {answer_str}.")
    else:
        body.append(
            f"\nStep 1 — identify. Each output is the number written in base {k} "
            f"(positional, digits 0..{k-1})."
        )
        ex_n, ex_s = pairs[0]
        body.append(f"  Check: {ex_n} in base {k} = {to_base_k(ex_n, k)}, matches the given {ex_s}. ✓")
        answer_str = to_base_k(query_n, k)
        body.append(
            f"\nStep 2 — convert {query_n} to base {k} by repeated division, "
            f"collecting remainders (read bottom-up):"
        )
        n = query_n
        while n > 0:
            r = n % k
            body.append(f"  {n} ÷ {k} = {n // k} remainder {r}  → digit '{to_base_k(r, k) if r < k else r}'")
            n //= k
        body.append(f"  Reading remainders bottom-up: {query_n} = {answer_str}.")

    cot = _wrap(
        "family=numeral; plan: identify the numeral system, verify on an example, "
        "then convert the query digit by digit",
        "m",
        "\n".join(body),
        answer_str,
    )
    return answer_str, cot


# ─────────────────────────────────────────────────────────────────────────────
# cipher: monoalphabetic substitution. Build char→char map from examples, apply.
# ─────────────────────────────────────────────────────────────────────────────

def build_substitution_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Build a char→char mapping from (ciphertext, plaintext) example pairs.

    Aligns examples word-by-word. Skips spaces. On conflict, keeps the first
    mapping (consistent with typical substitution-cipher semantics).
    """
    mapping: dict[str, str] = {}
    for ct, pt in pairs:
        ct_words = ct.split()
        pt_words = pt.split()
        if len(ct_words) != len(pt_words):
            continue  # malformed, skip
        for cw, pw in zip(ct_words, pt_words):
            if len(cw) != len(pw):
                continue
            for c, p in zip(cw, pw):
                if c not in mapping:
                    mapping[c] = p
    return mapping


def cipher_cot(pairs: list[tuple[str, str]], query: str) -> tuple[str, str]:
    body = [
        "Concept. This is a monoalphabetic substitution cipher: each ciphertext letter",
        "always maps to the same plaintext letter. Each example pairs a ciphertext",
        "phrase with its plaintext; since the words line up one-to-one, I can read off",
        "the letter mapping by aligning them character by character, then decode the",
        "query with the completed map.",
        "",
        "Step 1 — build the char→char map by aligning examples word by word:",
    ]
    mapping: dict[str, str] = {}
    for ct, pt in pairs[:6]:  # cap to 6 examples in the explanation
        cw = ct.split()
        pw = pt.split()
        if len(cw) != len(pw):
            continue
        for ciph_word, plain_word in zip(cw, pw):
            if len(ciph_word) != len(plain_word):
                continue
            body.append(f"  {ciph_word} → {plain_word}:")
            for c, p in zip(ciph_word, plain_word):
                if c in mapping:
                    if mapping[c] != p:
                        body.append(f"    {c} → {p}  (conflict; keep existing {mapping[c]})")
                else:
                    mapping[c] = p
                    body.append(f"    {c} → {p}")

    # Build a sorted view of the map
    body.append(f"\nFinal map ({len(mapping)} chars): " +
                ", ".join(f"{c}→{p}" for c, p in sorted(mapping.items())))

    # Apply to query
    body.append(f"\nStep 2 — decode the query '{query}' letter by letter using the map:")
    result_chars = []
    for c in query:
        if c == " ":
            result_chars.append(" ")
            continue
        if c in mapping:
            result_chars.append(mapping[c])
            body.append(f"  '{c}' → '{mapping[c]}'")
        else:
            # unseen char — emit as-is and flag
            result_chars.append(c)
            body.append(f"  '{c}' → '?' (not in map; keep as-is)")

    answer = "".join(result_chars)
    uncovered = [c for c in query if c != " " and c not in mapping]
    body.append(
        f"\nEvery query letter was present in the learned map "
        f"({'full coverage' if not uncovered else 'missing: ' + ','.join(uncovered)}), "
        f"so the decoding is determined."
    )
    body.append(f"Decoded query: {answer}")

    cot = _wrap(
        "family=cipher; plan: build the substitution map by aligning examples, confirm "
        "coverage, then decode the query letter by letter",
        "l",
        "\n".join(body),
        answer,
    )
    return answer, cot


# ─────────────────────────────────────────────────────────────────────────────
# bit_manipulation: per-bit rule discovery.
# For each output bit, search among candidate boolean functions of input bits
# until one matches all example rows. Then apply to query.
# ─────────────────────────────────────────────────────────────────────────────


def _bit(byte_str: str, i: int) -> int:
    """Bit i (0=LSB) of an 8-bit binary string. Note: byte_str[0] is the MSB."""
    return int(byte_str[7 - i])


def _byte_from_bits(bits: list[int]) -> str:
    """bits[0] is LSB. Returns 8-char MSB-first string."""
    return "".join(str(b) for b in reversed(bits))


def _candidate_rules_for(out_bit: int):
    """Yield candidate rules ordered by simplicity, with the "diagonal" in[out_bit] first.

    Order matters: among multiple rules that match the examples, we want the SIMPLEST
    and most natural one to win. The diagonal rule in[out_bit] is preferred (identity-like
    transformations are the most common single-bit operation).
    """
    # 1. Diagonal: in[out_bit] itself
    yield (f"in[{out_bit}]", lambda b, j=out_bit: b[j])
    # 2. Other single-input copies
    for j in range(8):
        if j != out_bit:
            yield (f"in[{j}]", lambda b, j=j: b[j])
    # 3. Negated diagonal
    yield (f"NOT in[{out_bit}]", lambda b, j=out_bit: 1 - b[j])
    # 4. Other negated single inputs
    for j in range(8):
        if j != out_bit:
            yield (f"NOT in[{j}]", lambda b, j=j: 1 - b[j])
    # 5. 2-input boolean ops
    for j in range(8):
        for k in range(j + 1, 8):
            yield (f"AND(in[{j}], in[{k}])", lambda b, j=j, k=k: b[j] & b[k])
            yield (f"OR(in[{j}], in[{k}])", lambda b, j=j, k=k: b[j] | b[k])
            yield (f"XOR(in[{j}], in[{k}])", lambda b, j=j, k=k: b[j] ^ b[k])
            yield (f"NAND(in[{j}], in[{k}])", lambda b, j=j, k=k: 1 - (b[j] & b[k]))
            yield (f"NOR(in[{j}], in[{k}])", lambda b, j=j, k=k: 1 - (b[j] | b[k]))
            yield (f"XNOR(in[{j}], in[{k}])", lambda b, j=j, k=k: 1 - (b[j] ^ b[k]))
    # 6. 3-input ops seen in the REAL puzzles (majority / choice-mux) — outside
    # the 1-2 input space, needed to crack the rows the per-bit search misses.
    for j in range(8):
        for k in range(j + 1, 8):
            for m in range(k + 1, 8):
                yield (
                    f"MAJ(in[{j}], in[{k}], in[{m}])",
                    lambda b, j=j, k=k, m=m: 1 if (b[j] + b[k] + b[m]) >= 2 else 0,
                )
    for j in range(8):
        for k in range(8):
            for m in range(8):
                if len({j, k, m}) == 3:
                    yield (
                        f"CHOICE(in[{j}] ? in[{k}] : in[{m}])",
                        lambda b, j=j, k=k, m=m: b[k] if b[j] else b[m],
                    )


def discover_bit_rules(pairs: list[tuple[str, str]]) -> list[Optional[tuple[str, callable]]]:
    """Per output bit (0..7), find a candidate function consistent with all examples.
    Returns a list of 8 (label, func) tuples (or None if no rule found)."""
    rules: list[Optional[tuple[str, callable]]] = [None] * 8

    example_bits = [
        ([_bit(inp, j) for j in range(8)], [_bit(out, j) for j in range(8)])
        for inp, out in pairs
    ]

    for out_bit in range(8):
        target_col = [outs[out_bit] for _, outs in example_bits]
        for label, func in _candidate_rules_for(out_bit):
            if all(func(ins) == target for (ins, _), target in zip(example_bits, target_col)):
                rules[out_bit] = (label, func)
                break

    return rules


def bit_query_unique(pairs: list[tuple[str, str]], query: str) -> bool:
    """Is the query output uniquely determined by the examples?

    For each of the 8 output bits, collect EVERY candidate function consistent
    with all examples and check they all agree on the query bit. If any output
    bit has consistent candidates that disagree on the query, the puzzle is
    under-determined (the model can't infer that bit from the prompt) → False.
    Used as a well-posedness gate by the generator.
    """
    example_bits = [
        ([_bit(inp, j) for j in range(8)], [_bit(out, j) for j in range(8)])
        for inp, out in pairs
    ]
    q_bits = [_bit(query, j) for j in range(8)]
    for out_bit in range(8):
        target_col = [outs[out_bit] for _, outs in example_bits]
        q_vals = set()
        for _label, func in _candidate_rules_for(out_bit):
            if all(func(ins) == t for (ins, _), t in zip(example_bits, target_col)):
                q_vals.add(func(q_bits))
        if len(q_vals) != 1:
            return False
    return True


def bit_manipulation_cot(
    pairs: list[tuple[str, str]], query: str, commit_early: bool = False
) -> tuple[str, str]:
    rules = discover_bit_rules(pairs)
    body = [
        "Concept. The transformation acts independently on each of the 8 output bits:",
        "every output bit is a simple boolean function of the input bits — either a",
        "copy/NOT of one input bit, or a 2-input gate (AND/OR/XOR and their negations).",
        "I index bits from the right, so in[0] is the least-significant (rightmost) bit",
        "and in[7] the leftmost. For each output bit I find the function that matches",
        "ALL the examples, then apply the 8 rules to the query.",
        "",
        "Step 1 — derive each output bit's rule (consistent with every example):",
    ]
    for out_bit, rule in enumerate(rules):
        if rule is None:
            body.append(f"  out[{out_bit}]: no simple rule found (default 0)")
        else:
            body.append(f"  out[{out_bit}] = {rule[0]}")

    # Verification text (against ALL examples).
    all_ok = True
    mismatch = None
    for inp, out in pairs:
        ib = [_bit(inp, j) for j in range(8)]
        pred = _byte_from_bits([(r[1](ib) if r else 0) for r in rules])
        if pred != out:
            all_ok = False
            mismatch = (inp, pred, out)
            break
    if all_ok:
        verify_text = (
            f"verify: applying these 8 rules to all {len(pairs)} example "
            f"inputs reproduces every example output exactly. For instance input "
            f"{pairs[0][0]} → {pairs[0][1]} ✓."
        )
    else:
        verify_text = f"verify: mismatch on {mismatch[0]} (got {mismatch[1]}, expected {mismatch[2]})."

    # Application text (query, bit by bit).
    in_bits = [_bit(query, j) for j in range(8)]
    apply_lines = [
        f"apply to the query input {query}.",
        "  Input bits (in[7..0], left to right): "
        + " ".join(f"in[{7-i}]={query[i]}" for i in range(8)),
    ]
    out_bits = []
    for out_bit, rule in enumerate(rules):
        v = rule[1](in_bits) if rule else 0
        out_bits.append(v)
        rule_label = rule[0] if rule else "0 (default)"
        apply_lines.append(f"  out[{out_bit}] = {rule_label} = {v}")

    answer = _byte_from_bits(out_bits)
    apply_lines.append(
        f"\nAssemble the output bits from out[7] down to out[0] "
        f"(most-significant first): {answer}."
    )

    if commit_early:
        # Commit-then-verify (see gravity_cot): answer first, verification after.
        body.append("\nStep 2 — " + apply_lines[0])
        body.extend(apply_lines[1:])
        body.append(f"\nAnswer: \\boxed{{{answer}}}")
        body.append("\nStep 3 — " + verify_text)
    else:
        body.append("\nStep 2 — " + verify_text)
        body.append("\nStep 3 — " + apply_lines[0])
        body.extend(apply_lines[1:])

    cot = _wrap(
        "family=bit_manipulation; plan: find each output bit's boolean rule, verify on "
        "all examples, then apply bit by bit to the query",
        "c",
        "\n".join(body),
        answer,
    )
    return answer, cot


# ─────────────────────────────────────────────────────────────────────────────
# Hard families: cryptarithm + equation_numeric
#
# The train.csv "transformation" family is a generic "LHS = RHS" string puzzle.
# We search for a small set of common transformation classes; whichever one
# fits ALL examples is the rule. If none fit, we return None and the orchestrator
# falls back to heuristic wrapping for that row.
#
# Transformation classes we try (ordered by simplicity):
#   1. CHAR DELETION: delete every occurrence of a specific char-set
#   2. CHAR SUBSTITUTION: replace each occurrence of char X with char Y
#   3. POSITION FILTER: keep only chars at specific indices (e.g. even positions)
#   4. AFFIX TRIM: strip a constant prefix/suffix
# ─────────────────────────────────────────────────────────────────────────────


def _try_char_deletion(pairs: list[tuple[str, str]]) -> Optional[tuple[set[str], str]]:
    """Try: 'rhs = lhs with characters in DELETE_SET removed'.
    Returns (DELETE_SET, explanation) or None.
    """
    # Candidate delete sets: chars in lhs but not in rhs (per example, intersect across)
    candidates_per_example = []
    for lhs, rhs in pairs:
        # what got deleted: chars present in lhs but missing the same count from rhs
        from collections import Counter
        lc, rc = Counter(lhs), Counter(rhs)
        deleted = set()
        for ch, cnt in lc.items():
            if rc.get(ch, 0) < cnt:
                deleted.add(ch)
        candidates_per_example.append(deleted)
    if not candidates_per_example:
        return None
    common = set.intersection(*candidates_per_example) if candidates_per_example[0] else set()
    if not common:
        return None
    # Verify: for each example, removing all chars in `common` from lhs gives rhs
    for lhs, rhs in pairs:
        if "".join(c for c in lhs if c not in common) != rhs:
            return None
    return common, f"delete all occurrences of {sorted(common)} from input"


def _try_char_substitution(pairs: list[tuple[str, str]]) -> Optional[tuple[dict, str]]:
    """Try: 'rhs = lhs with each occurrence of char X replaced by char Y'.
    Only valid when len(lhs)==len(rhs) for all examples.
    """
    if not all(len(l) == len(r) for l, r in pairs):
        return None
    mapping: dict[str, str] = {}
    for lhs, rhs in pairs:
        for a, b in zip(lhs, rhs):
            if a in mapping:
                if mapping[a] != b:
                    return None
            else:
                mapping[a] = b
    # Verify
    for lhs, rhs in pairs:
        if "".join(mapping[c] for c in lhs) != rhs:
            return None
    return mapping, f"substitute chars per map: {mapping}"


def _try_position_filter(pairs: list[tuple[str, str]]) -> Optional[tuple[list[int], str]]:
    """Try: 'rhs = chars at specific indices of lhs'.
    Returns (kept_indices, explanation).
    """
    for lhs, rhs in pairs:
        if len(rhs) > len(lhs):
            return None
    # All examples must have same lhs length (otherwise positions vary)
    lhs_lens = {len(l) for l, _ in pairs}
    if len(lhs_lens) > 1:
        return None
    n = lhs_lens.pop()
    # The set of indices to keep must produce rhs in every example
    # Try: for each subset of positions of size len(rhs[0]), check
    # Too many subsets in general; restrict to "all positions where lhs[i] is not in SKIP_SET"
    # which is equivalent to char_deletion (already tried).
    # Try positional rules: even indices, odd indices, [0:k], [-k:]
    candidates = []
    for r in range(1, n + 1):
        candidates.append(("first k", list(range(r))))
        candidates.append(("last k", list(range(n - r, n))))
    candidates.append(("even idx", [i for i in range(n) if i % 2 == 0]))
    candidates.append(("odd idx", [i for i in range(n) if i % 2 == 1]))
    for label, idx in candidates:
        if all("".join(lhs[i] for i in idx if i < len(lhs)) == rhs for lhs, rhs in pairs):
            return idx, f"keep positions {label}"
    return None


def _try_affix_trim(pairs: list[tuple[str, str]]) -> Optional[tuple[tuple[str, str], str]]:
    """Try: 'rhs = lhs stripped of constant prefix P and suffix S'."""
    # Find common prefix and suffix from all (lhs, rhs)
    def diff_around(lhs, rhs):
        if rhs not in lhs:
            return None
        i = lhs.index(rhs)
        prefix = lhs[:i]
        suffix = lhs[i + len(rhs):]
        return (prefix, suffix)

    diffs = [diff_around(l, r) for l, r in pairs]
    if any(d is None for d in diffs):
        return None
    prefixes = {d[0] for d in diffs}
    suffixes = {d[1] for d in diffs}
    if len(prefixes) == 1 and len(suffixes) == 1:
        p, s = prefixes.pop(), suffixes.pop()
        if p or s:
            return (p, s), f"strip prefix {p!r}, suffix {s!r}"
    return None


def transformation_cot(pairs: list[tuple[str, str]], query: str) -> Optional[tuple[str, str]]:
    """Generic LHS→RHS transformation. Returns (answer, cot) or None if no rule fits."""

    body = ["Examples (LHS = RHS):"]
    for lhs, rhs in pairs:
        body.append(f"  {lhs!r} = {rhs!r}")

    # Try transformation classes in order of simplicity
    body.append("\nSearch for a consistent transformation rule:")

    # 1. Deletion
    body.append("\n[1] Try character deletion (rhs = lhs minus some chars):")
    res = _try_char_deletion(pairs)
    if res:
        delete_set, expl = res
        body.append(f"   ✓ Rule found: {expl}")
        answer = "".join(c for c in query if c not in delete_set)
        body.append(f"\nApply to query {query!r}: remove {sorted(delete_set)} → {answer!r}")
        cot = _wrap(
            "family=transformation; plan: search rule via code-mode, apply",
            "c",
            "\n".join(body),
            answer,
        )
        return answer, cot
    body.append("   ✗ no consistent deletion set")

    # 2. Substitution
    body.append("\n[2] Try character substitution (rhs = lhs with chars replaced):")
    res = _try_char_substitution(pairs)
    if res:
        mapping, expl = res
        body.append(f"   ✓ Rule found: {expl}")
        try:
            answer = "".join(mapping[c] for c in query)
            body.append(f"\nApply to query {query!r}: → {answer!r}")
            cot = _wrap(
                "family=transformation; plan: search rule via code-mode, apply",
                "c",
                "\n".join(body),
                answer,
            )
            return answer, cot
        except KeyError:
            body.append("   ✗ query has chars not in substitution map; skip")
    else:
        body.append("   ✗ length mismatch or inconsistent substitution")

    # 3. Position filter
    body.append("\n[3] Try positional filter (rhs = lhs at specific indices):")
    res = _try_position_filter(pairs)
    if res:
        idx, expl = res
        body.append(f"   ✓ Rule found: {expl}")
        if all(i < len(query) for i in idx):
            answer = "".join(query[i] for i in idx)
            body.append(f"\nApply to query {query!r}: positions {idx} → {answer!r}")
            cot = _wrap(
                "family=transformation; plan: search rule via code-mode, apply",
                "c",
                "\n".join(body),
                answer,
            )
            return answer, cot
        body.append("   ✗ query too short for those positions")
    else:
        body.append("   ✗ no positional rule fits")

    # 4. Affix trim
    body.append("\n[4] Try affix trim (rhs = lhs with constant prefix/suffix removed):")
    res = _try_affix_trim(pairs)
    if res:
        (p, s), expl = res
        body.append(f"   ✓ Rule found: {expl}")
        if query.startswith(p) and query.endswith(s) and len(query) >= len(p) + len(s):
            answer = query[len(p): len(query) - len(s) if len(s) else None]
            body.append(f"\nApply to query {query!r}: strip → {answer!r}")
            cot = _wrap(
                "family=transformation; plan: search rule via code-mode, apply",
                "c",
                "\n".join(body),
                answer,
            )
            return answer, cot

    # No rule matched
    return None


# ─────────────────────────────────────────────────────────────────────────────
# transformation (REAL distribution): hidden-operator 2-operand arithmetic.
#
# Every real "transformation" prompt is `AA <op> BB` (5 chars): two 2-char
# operands with an operator char at index 2. The operator glyph is puzzle-local
# (the same glyph means different things in different puzzles). Operations seen
# in train.csv: add, sub, abs_diff, mult, concat, reverse_concat — rendered
# either big-endian ("fwd") or little-endian ("RR": reverse each operand's
# digits, compute, reverse the result string).
#
# See data/analysis/transformation_taxonomy.md.
# ─────────────────────────────────────────────────────────────────────────────


def arith_result(op: str, mode: Optional[str], sa: str, sb: str) -> Optional[str]:
    """Apply one operator to two 2-char operand strings; return the result string.

    `op` ∈ {add, sub, absdiff, mult, concat, rconcat}. `mode` ∈ {"fwd", "RR", None}.
    concat/rconcat ignore `mode`. Returns None for unknown ops.
    """
    if op == "concat":
        return sa + sb
    if op == "rconcat":
        return sb + sa
    a, b = int(sa), int(sb)
    if mode == "RR":
        a, b = int(sa[::-1]), int(sb[::-1])
    if op == "add":
        v = a + b
    elif op == "sub":
        v = a - b
    elif op == "absdiff":
        v = abs(a - b)
    elif op == "mult":
        v = a * b
    else:
        return None
    s = str(v)
    if mode == "RR":
        neg = s.startswith("-")
        if neg:
            s = s[1:]
        s = s[::-1]
        if neg:
            s = "-" + s
    return s


# Candidate (op, mode) pairs, ordered simplest/most-common first. The discoverer
# returns the first one consistent with all of an operator's examples.
ARITH_CANDIDATES: list[tuple[str, Optional[str]]] = [
    ("add", "fwd"),
    ("sub", "fwd"),
    ("absdiff", "fwd"),
    ("mult", "fwd"),
    ("concat", None),
    ("rconcat", None),
    ("add", "RR"),
    ("absdiff", "RR"),
    ("mult", "RR"),
    # sub/RR is never *generated* as a rule, but it must be in the candidate
    # space so the well-posedness gate rejects puzzles that are ambiguous w.r.t.
    # it — e.g. when every example of a sub/absdiff operator has a non-negative
    # result, sub and absdiff are indistinguishable from the examples and only
    # diverge on a negative query (the residual ambiguity full-sweep QA found).
    # Listed last so the discoverer still prefers absdiff/RR for genuine puzzles.
    ("sub", "RR"),
]

_OP_DESC = {
    ("add", "fwd"): "add the two numbers (a + b)",
    ("sub", "fwd"): "subtract (a - b)",
    ("absdiff", "fwd"): "absolute difference |a - b|",
    ("mult", "fwd"): "multiply (a * b)",
    ("concat", None): "concatenate the operands (a then b)",
    ("rconcat", None): "concatenate the operands reversed (b then a)",
    ("add", "RR"): "reverse each operand's digits, add, then reverse the result",
    ("absdiff", "RR"): "reverse each operand's digits, take |a - b|, then reverse the result",
    ("mult", "RR"): "reverse each operand's digits, multiply, then reverse the result",
    ("sub", "RR"): "reverse each operand's digits, subtract, then reverse the result",
}


def solve_arith_operator(rows: list[tuple[str, str, str]]) -> Optional[tuple[str, Optional[str]]]:
    """rows: list of (sa, sb, rhs). Return the first (op, mode) consistent with all."""
    for op, mode in ARITH_CANDIDATES:
        if all(arith_result(op, mode, sa, sb) == rhs for sa, sb, rhs in rows):
            return (op, mode)
    return None


def _split_lhs(lhs: str) -> tuple[str, str, str]:
    """`AAoBB` -> (operand_a, op_char, operand_b)."""
    return lhs[:2], lhs[2], lhs[3:5]


def _explain_arith_apply(op: str, mode: Optional[str], sa: str, sb: str) -> list[str]:
    """Step-by-step worked computation for one (op, mode) on operands sa, sb.
    The displayed work is consistent with `arith_result`."""
    lines: list[str] = []
    if op in ("concat", "rconcat"):
        order = f"{sa} then {sb}" if op == "concat" else f"{sb} then {sa}"
        lines.append(f"  This operator just writes the operands {('in order' if op=='concat' else 'swapped')}: {order}.")
        lines.append(f"  → {arith_result(op, mode, sa, sb)}")
        return lines

    wa, wb = (sa[::-1], sb[::-1]) if mode == "RR" else (sa, sb)
    if mode == "RR":
        lines.append(f"  RR rule: first reverse each operand's digits — {sa}→{wa}, {sb}→{wb}.")
    a, b = int(wa), int(wb)

    if op == "add":
        work = ColumnAdd(D(a), D(b))
        lines.append(f"  Add {wa} + {wb}:")
        lines.append(_indent(work.explain(), 4))
        raw = str(a + b)
    elif op == "sub":
        work = ColumnSubtract(D(a), D(b))
        lines.append(f"  Subtract {wa} − {wb}:")
        lines.append(_indent(work.explain(), 4))
        raw = str(a - b)
    elif op == "absdiff":
        hi, lo = (a, b) if a >= b else (b, a)
        work = ColumnSubtract(D(hi), D(lo))
        lines.append(f"  Absolute difference |{wa} − {wb}| = {hi} − {lo}:")
        lines.append(_indent(work.explain(), 4))
        raw = str(abs(a - b))
    elif op == "mult":
        work = ColumnMultiply(D(a), D(b))
        lines.append(f"  Multiply {wa} × {wb}:")
        lines.append(_indent(work.explain(), 4))
        raw = str(a * b)
    else:
        raw = ""

    if mode == "RR":
        lines.append(f"  Then reverse the result string: {raw} → {raw[::-1] if not raw.startswith('-') else '-' + raw[1:][::-1]}.")
    lines.append(f"  → {arith_result(op, mode, sa, sb)}")
    return lines


def arithmetic_transformation_cot(
    pairs: list[tuple[str, str]], query: str
) -> Optional[tuple[str, str]]:
    """Hidden-operator arithmetic CoT. Returns (answer, cot) or None if the rule
    cannot be recovered (then the orchestrator retries with a fresh puzzle)."""
    # Group examples by their operator char (index 2).
    groups: dict[str, list[tuple[str, str, str]]] = {}
    order: list[str] = []
    for lhs, rhs in pairs:
        a, opc, b = _split_lhs(lhs)
        if opc not in groups:
            groups[opc] = []
            order.append(opc)
        groups[opc].append((a, b, rhs))

    rules: dict[str, tuple[str, Optional[str]]] = {}
    for opc in order:
        sol = solve_arith_operator(groups[opc])
        if sol is None:
            return None
        rules[opc] = sol

    qa, q_op, qb = _split_lhs(query)
    if q_op not in rules:
        return None  # query operator never demonstrated -> unsolvable

    body = [
        "Concept. Each line reads `AA<op>BB = result`: two 2-digit operands joined by",
        "a symbol that stands for a hidden operation (it is NOT the literal math",
        "symbol). I figure out what each symbol does by checking which operation",
        "reproduces all of its examples, then apply that rule to the query.",
        "",
        "Examples (LHS = RHS):",
    ]
    for lhs, rhs in pairs:
        body.append(f"  {lhs} = {rhs}")

    body.append("\nStep 1 — identify each operator by testing operations against its examples:")
    for opc in order:
        op, mode = rules[opc]
        ex = groups[opc][0]
        body.append(
            f"  '{opc}': the rule '{_OP_DESC[(op, mode)]}' reproduces its examples "
            f"(e.g. {ex[0]}'{opc}'{ex[1]} = {ex[2]})."
        )

    op, mode = rules[q_op]
    answer = arith_result(op, mode, qa, qb)
    body.append(
        f"\nStep 2 — apply operator '{q_op}' ({_OP_DESC[(op, mode)]}) to the query "
        f"{query} → operands {qa} and {qb}:"
    )
    body.extend(_explain_arith_apply(op, mode, qa, qb))

    cot = _wrap(
        "family=transformation; plan: identify each hidden operator from its examples, "
        "then compute the query operator step by step",
        "c",
        "\n".join(body),
        answer,
    )
    return answer, cot
