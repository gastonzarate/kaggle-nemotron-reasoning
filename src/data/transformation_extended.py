"""Extended solver for the REAL `transformation` family.

Covers what the basic arithmetic solver misses, based on the verified taxonomy
(data/analysis/transformation_taxonomy.md):

DIGIT-OPERAND extensions
  - ops: add, sub, absdiff, mult, concat, rconcat, floordiv, mod
  - modes: fwd and RR (reverse operand digits, compute, reverse result string)
  - constant offsets ±1 on add/mult
  - negative rendering: '-' OR the operator glyph as the sign prefix

SYMBOL-OPERAND (cipher-arithmetic)
  - operands and RHS are symbols encoding digits via a per-puzzle injective
    symbol→digit map; recovered by backtracking CSP, then the same op space.

Every recovered rule must reproduce ALL examples of the query's operator glyph
AND the produced answer is later gated against the real gold by the corpus
builder — so nothing unverified ships.
"""

from __future__ import annotations

from itertools import permutations
from typing import Callable, Optional

# ───────────────────────── digit-operand op space ─────────────────────────


def _rev(s: str) -> str:
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


def _mk_arith(op: str, mode: str, offset: int = 0, neg_glyph: bool = False):
    """Build fn(sa, sb, glyph) -> result-string for a digit-operand rule."""

    def fn(sa: str, sb: str, glyph: str) -> Optional[str]:
        if not (sa.isdigit() and sb.isdigit()):
            return None
        a, b = int(sa), int(sb)
        if mode == "RR":
            a, b = int(sa[::-1]), int(sb[::-1])
        if op == "add":
            v = a + b + offset
        elif op == "sub":
            v = a - b + offset
        elif op == "absdiff":
            v = abs(a - b) + offset
        elif op == "mult":
            v = a * b + offset
        elif op == "floordiv":
            if b == 0:
                return None
            v = a // b + offset
        elif op == "mod":
            if b == 0:
                return None
            v = a % b + offset
        else:
            return None
        s = str(v)
        if mode == "RR":
            s = _rev(s)
        if neg_glyph and s.startswith("-"):
            s = glyph + s[1:]
        return s

    return fn


def _digit_candidates() -> list[tuple[str, Callable]]:
    cands: list[tuple[str, Callable]] = []
    # concat / rconcat (mode-free)
    cands.append(("concatenate the operands (a then b)", lambda sa, sb, g: sa + sb))
    cands.append(("concatenate the operands reversed (b then a)", lambda sa, sb, g: sb + sa))
    for mode in ("fwd", "RR"):
        mdesc = "" if mode == "fwd" else " on digit-reversed operands, result string reversed"
        for op, odesc in (
            ("add", "add"), ("sub", "subtract"), ("absdiff", "absolute difference"),
            ("mult", "multiply"), ("floordiv", "integer-divide"), ("mod", "take the remainder of"),
        ):
            for offset in (0, 1, -1):
                if offset and op in ("floordiv", "mod"):
                    continue
                off = "" if offset == 0 else (" then add 1" if offset == 1 else " then subtract 1")
                for neg_glyph in (False, True):
                    if neg_glyph and op in ("absdiff", "mult", "add", "floordiv", "mod"):
                        continue  # only subtraction goes negative
                    ng = " (negative sign written as the operator glyph)" if neg_glyph else ""
                    cands.append((
                        f"{odesc} the two numbers{off}{mdesc}{ng}",
                        _mk_arith(op, mode, offset, neg_glyph),
                    ))
    return cands


_DIGIT_CANDS = _digit_candidates()


def solve_digit_group(
    rows: list[tuple[str, str, str, str]],
    query: Optional[tuple[str, str, str]] = None,
    gold: Optional[str] = None,
) -> Optional[tuple[str, Callable]]:
    """rows: (sa, sb, glyph, rhs). Returns a candidate consistent with all rows.

    Gold-conditioning (training-data generation only): when several candidates
    fit the examples, prefer the one whose query prediction equals the known
    gold — that one IS the latent rule. Without gold, first match wins.
    """
    consistent = [
        (desc, fn) for desc, fn in _DIGIT_CANDS
        if all(fn(sa, sb, g) == rhs for sa, sb, g, rhs in rows)
    ]
    if not consistent:
        return None
    if gold is not None and query is not None:
        qa, qg, qb = query
        for desc, fn in consistent:
            if fn(qa, qb, qg) == gold:
                return desc, fn
    return consistent[0]


# ───────────────────────── symbol-operand CSP ─────────────────────────

_SYM_OPS: list[tuple[str, Callable[[int, int], Optional[int]]]] = [
    ("add the two numbers", lambda a, b: a + b),
    ("subtract (a - b)", lambda a, b: a - b),
    ("absolute difference |a - b|", lambda a, b: abs(a - b)),
    ("multiply the two numbers", lambda a, b: a * b),
]


def _sym_concat_try(rows: list[tuple[str, str, str]], query_ab: tuple[str, str]) -> Optional[tuple[str, str]]:
    """concat/rconcat need no digit map — pure string ops on the symbols."""
    if all(rhs == sa + sb for sa, sb, rhs in rows):
        return ("concatenate the operands (a then b)", query_ab[0] + query_ab[1])
    if all(rhs == sb + sa for sa, sb, rhs in rows):
        return ("concatenate the operands reversed (b then a)", query_ab[1] + query_ab[0])
    return None


def solve_symbol_group(
    rows: list[tuple[str, str, str]], query_ab: tuple[str, str],
    gold: Optional[str] = None,
) -> Optional[tuple[str, dict, str]]:
    """rows: (sa, sb, rhs) all in symbol space. Returns (desc, sym2dig, answer) or None.

    Backtracking over injective symbol→digit assignments; for each op (incl. the
    RR variant), a full assignment must satisfy every example row, and the
    answer is the computed value re-encoded through the inverse map.
    """
    symbols = sorted({c for sa, sb, rhs in rows for c in sa + sb + rhs} |
                     {c for c in query_ab[0] + query_ab[1]})
    if len(symbols) > 10:
        return None

    def encode(val: int, dig2sym: dict) -> Optional[str]:
        s = str(val)
        if s.startswith("-"):
            return None
        try:
            return "".join(dig2sym[d] for d in s)
        except KeyError:
            return None

    def value(tok: str, sym2dig: dict) -> Optional[int]:
        try:
            return int("".join(sym2dig[c] for c in tok))
        except KeyError:
            return None

    for mode in ("fwd", "RR"):
        for desc, op in _SYM_OPS:
            # DFS over digit assignments, pruning on any fully-assigned row.
            def consistent(sym2dig: dict) -> bool:
                dig2sym = {v: k for k, v in sym2dig.items()}
                for sa, sb, rhs in rows:
                    va, vb = value(sa, sym2dig), value(sb, sym2dig)
                    if va is None or vb is None:
                        continue  # not fully assigned yet
                    if mode == "RR":
                        va, vb = int(str(va).zfill(len(sa))[::-1]), int(str(vb).zfill(len(sb))[::-1])
                    out = op(va, vb)
                    if out is None or out < 0:
                        return False
                    s = str(out)
                    if mode == "RR":
                        s = s[::-1]
                    if len(s) != len(rhs):
                        return False
                    enc = encode(int(s) if not s.startswith("0") else 0, dig2sym)  # placeholder
                    # encode digit-string directly (preserving leading zeros):
                    try:
                        enc = "".join(dig2sym[d] for d in s)
                    except KeyError:
                        return False
                    if enc != rhs:
                        return False
                return True

            def solutions(i: int, sym2dig: dict, used: set):
                if i == len(symbols):
                    if consistent(sym2dig):
                        yield dict(sym2dig)
                    return
                for d in "0123456789":
                    if d in used:
                        continue
                    sym2dig[symbols[i]] = d
                    if consistent(sym2dig):
                        used.add(d)
                        yield from solutions(i + 1, sym2dig, used)
                        used.discard(d)
                    del sym2dig[symbols[i]]

            def answer_for(sol: dict) -> Optional[str]:
                va, vb = value(query_ab[0], sol), value(query_ab[1], sol)
                if va is None or vb is None:
                    return None
                a2, b2 = va, vb
                if mode == "RR":
                    a2 = int(str(va).zfill(len(query_ab[0]))[::-1])
                    b2 = int(str(vb).zfill(len(query_ab[1]))[::-1])
                out = op(a2, b2)
                if out is None or out < 0:
                    return None
                s = str(out)
                if mode == "RR":
                    s = s[::-1]
                dig2sym = {v: k for k, v in sol.items()}
                try:
                    return "".join(dig2sym[d] for d in s)
                except KeyError:
                    return None

            mdesc = "" if mode == "fwd" else " (operands digit-reversed, result reversed)"
            first_valid: Optional[tuple[str, dict, str]] = None
            # Gold-conditioning: scan up to 200 consistent maps; prefer the one
            # whose query answer equals the known gold (the true latent map).
            for n, sol in enumerate(solutions(0, {}, set())):
                if n >= 200:
                    break
                ans = answer_for(sol)
                if ans is None:
                    continue
                if gold is not None and ans == gold:
                    return (desc + mdesc, sol, ans)
                if first_valid is None:
                    first_valid = (desc + mdesc, sol, ans)
            if gold is None and first_valid is not None:
                return first_valid
    return None


# ───────────────────────── top-level solve + reasoning ─────────────────────────


def solve_transformation_extended(
    pairs: list[tuple[str, str]], query: str, gold: Optional[str] = None
) -> Optional[tuple[str, str]]:
    """Returns (answer, reasoning_body) or None. LHS must be 5 chars `AAgBB`.

    `gold` enables gold-conditioned disambiguation (training-data generation)."""
    if len(query) != 5 or any(len(l) != 5 for l, _ in pairs):
        return None

    groups: dict[str, list[tuple[str, str, str]]] = {}
    for lhs, rhs in pairs:
        groups.setdefault(lhs[2], []).append((lhs[:2], lhs[3:5], rhs))
    qa, q_glyph, qb = query[:2], query[2], query[3:5]
    if q_glyph not in groups:
        return None
    qrows = groups[q_glyph]

    body = [
        "Each line is `AA<op>BB = result`: two 2-character operands joined by a symbol",
        "that encodes a hidden operation. I work out what the query's operator does by",
        "finding the rule that reproduces every one of its examples, then apply it.",
        "",
        "Examples for operator '" + q_glyph + "':",
    ]
    for sa, sb, rhs in qrows:
        body.append(f"  {sa}{q_glyph}{sb} = {rhs}")

    digit_rows = [(sa, sb, q_glyph, rhs) for sa, sb, rhs in qrows]
    if all(sa.isdigit() and sb.isdigit() for sa, sb, _, _ in digit_rows):
        sol = solve_digit_group(digit_rows, query=(qa, q_glyph, qb), gold=gold)
        if sol is None:
            return None
        desc, fn = sol
        ans = fn(qa, qb, q_glyph)
        if ans is None:
            return None
        body.append(f"\nRule found: '{q_glyph}' means: {desc}.")
        for sa, sb, g, rhs in digit_rows[:2]:
            body.append(f"  Check {sa}{g}{sb}: {fn(sa, sb, g)} = {rhs} ✓")
        body.append(f"\nApply to the query {query}: operands {qa} and {qb} → {ans}")
        return ans, "\n".join(body)

    # symbol-operand path
    sym = solve_symbol_group(qrows, (qa, qb), gold=gold)
    if sym is None:
        return None
    desc, sym2dig, ans = sym
    body.append(
        "\nThe operands are symbols standing for digits. Solving the consistent "
        "symbol→digit assignment from the examples:"
    )
    body.append("  " + ", ".join(f"{s}→{d}" for s, d in sorted(sym2dig.items())))
    body.append(f"Operation: {desc}.")
    body.append(
        f"\nApply to the query {query}: decode {qa},{qb} with the map, compute, "
        f"re-encode the digits back into symbols → {ans}"
    )
    return ans, "\n".join(body)
