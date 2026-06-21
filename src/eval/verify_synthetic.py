"""Independent exhaustive verifier for the synthetic v3 dataset.

For every row it checks three orthogonal properties:

1. STRUCTURE  — the CoT has exactly one ``<think>…</think>`` block and exactly
   one non-empty ``\\boxed{…}`` after it.
2. COHERENCE  — the value inside ``\\boxed{…}`` equals the ``answer`` column
   (exact string match, except gravity/unit_conversion which use a 1% relative
   tolerance, matching the official metric). This is the property that the
   training notebook relies on, because it re-emits ``\\boxed{answer}`` and the
   surrounding reasoning must agree with it.
3. CORRECTNESS — re-solve the puzzle *from the prompt* with code that does NOT
   import the generators (so the check is genuinely independent), and confirm
   the ``answer`` column is a correct solution.

Run:  uv run python -m src.eval.verify_synthetic --csv <path>
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Boxed / structure extraction
# ---------------------------------------------------------------------------


def extract_last_boxed(text: str) -> str | None:
    """Return the brace-balanced payload of the last ``\\boxed{...}``."""
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
    return text[start : i - 1] if depth == 0 else text[start:]


def check_structure(cot: str) -> str | None:
    """Return None if OK, else a short reason string."""
    n_open = cot.count("<think>")
    n_close = cot.count("</think>")
    if n_open != 1 or n_close != 1:
        return f"think_tags({n_open},{n_close})"
    n_boxed = len(re.findall(r"\\boxed\{", cot))
    if n_boxed != 1:
        return f"boxed_count({n_boxed})"
    boxed = extract_last_boxed(cot)
    if boxed is None or boxed.strip() == "":
        return "empty_boxed"
    # boxed must come after </think>
    if cot.index("\\boxed{") < cot.index("</think>"):
        return "boxed_before_think_close"
    return None


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

FLOAT_FAMILIES = {"gravity", "unit_conversion"}


def _close(a: str, b: str) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        return False


def check_coherence(boxed: str, answer: str, family: str) -> str | None:
    boxed = (boxed or "").strip()
    answer = str(answer).strip()
    if family in FLOAT_FAMILIES:
        if _close(boxed, answer):
            return None
        return f"numeric_incoherent:{boxed}!={answer}"
    if boxed == answer:
        return None
    return f"incoherent:{boxed!r}!={answer!r}"


# ---------------------------------------------------------------------------
# Independent re-solvers (DO NOT import src.data.*)
# ---------------------------------------------------------------------------

_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _to_roman(n: int) -> str:
    out = []
    for v, s in _ROMAN:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def _to_base_k(n: int, k: int, alphabet: str = "0123456789ABCDEF") -> str:
    if n == 0:
        return alphabet[0]
    digits = []
    while n > 0:
        digits.append(alphabet[n % k])
        n //= k
    return "".join(reversed(digits))


def solve_gravity(prompt: str) -> str | None:
    pairs = re.findall(r"t = ([\d.]+)s, distance = ([\d.]+) m", prompt)
    q = re.search(r"for t = ([\d.]+)s given", prompt)
    if len(pairs) < 1 or not q:
        return None
    gs = [2 * float(d) / (float(t) ** 2) for t, d in pairs]
    g = sum(gs) / len(gs)
    tq = float(q.group(1))
    return f"{0.5 * g * tq * tq:.2f}"


def solve_unit(prompt: str) -> str | None:
    pairs = re.findall(r"([\d.]+) m becomes ([\d.]+)", prompt)
    q = re.search(r"convert the following measurement: ([\d.]+) m", prompt)
    if len(pairs) < 2 or not q:
        return None
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return f"{a * float(q.group(1)) + b:.2f}"


def solve_numeral(prompt: str) -> str | None:
    pairs = re.findall(r"^(\d+) -> (\S+)$", prompt, re.MULTILINE)
    q = re.search(r"write the number (\d+) in", prompt)
    if not pairs or not q:
        return None
    ip = [(int(n), s) for n, s in pairs]
    qn = int(q.group(1))
    if all(_to_roman(n) == s for n, s in ip):
        return _to_roman(qn)
    for k in range(2, 17):
        if all(_to_base_k(n, k) == s.upper() for n, s in ip):
            return _to_base_k(qn, k)
    return None  # unknown system -> can't verify


def solve_cipher(prompt: str) -> str | None:
    body = prompt.split("examples:\n", 1)[-1]
    lines = [ln for ln in body.splitlines() if " -> " in ln]
    q = re.search(r"decrypt the following text: (.+)$", prompt)
    if not lines or not q:
        return None
    mapping: dict[str, str] = {}
    for ln in lines:
        ct, pt = ln.split(" -> ", 1)
        cw, pw = ct.split(), pt.split()
        if len(cw) != len(pw):
            continue
        for a, b in zip(cw, pw):
            if len(a) != len(b):
                continue
            for c, p in zip(a, b):
                mapping.setdefault(c, p)
    query = q.group(1)
    out = []
    for c in query:
        if c == " ":
            out.append(" ")
        elif c in mapping:
            out.append(mapping[c])
        else:
            return None  # uncovered char -> can't verify
    return "".join(out)


def solve_bit(prompt: str) -> str | None:
    """Re-discover per-bit rules consistent with ALL shown examples, apply to query.

    Returns the predicted output, or None if no consistent per-bit rule exists
    (ambiguity-free puzzles should always resolve).
    """
    pairs = re.findall(r"^([01]{8}) -> ([01]{8})$", prompt, re.MULTILINE)
    q = re.search(r"determine the output for: ([01]{8})", prompt)
    if not pairs or not q:
        return None

    def bit(s: str, i: int) -> int:
        return int(s[7 - i])

    ex = [([bit(i, j) for j in range(8)], [bit(o, j) for j in range(8)]) for i, o in pairs]

    def candidates(ob: int):
        yield lambda b, j=ob: b[j]
        for j in range(8):
            if j != ob:
                yield lambda b, j=j: b[j]
        yield lambda b, j=ob: 1 - b[j]
        for j in range(8):
            if j != ob:
                yield lambda b, j=j: 1 - b[j]
        for j in range(8):
            for k in range(j + 1, 8):
                yield lambda b, j=j, k=k: b[j] & b[k]
                yield lambda b, j=j, k=k: b[j] | b[k]
                yield lambda b, j=j, k=k: b[j] ^ b[k]
                yield lambda b, j=j, k=k: 1 - (b[j] & b[k])
                yield lambda b, j=j, k=k: 1 - (b[j] | b[k])
                yield lambda b, j=j, k=k: 1 - (b[j] ^ b[k])

    qbits = [bit(q.group(1), j) for j in range(8)]
    out = [0] * 8
    for ob in range(8):
        col = [o[ob] for _, o in ex]
        found = None
        for fn in candidates(ob):
            if all(fn(ins) == t for (ins, _), t in zip(ex, col)):
                found = fn
                break
        if found is None:
            return None
        out[ob] = found(qbits)
    return "".join(str(b) for b in reversed(out))


def _arith(op: str, mode: str | None, sa: str, sb: str) -> str | None:
    """Independent reimplementation of the hidden-operator arithmetic."""
    if op == "concat":
        return sa + sb
    if op == "rconcat":
        return sb + sa
    a, b = int(sa), int(sb)
    if mode == "RR":
        a, b = int(sa[::-1]), int(sb[::-1])
    v = {"add": a + b, "sub": a - b, "absdiff": abs(a - b), "mult": a * b}.get(op)
    if v is None:
        return None
    s = str(v)
    if mode == "RR":
        neg = s.startswith("-")
        s = s[1:] if neg else s
        s = s[::-1]
        s = "-" + s if neg else s
    return s


_ARITH_CANDS = [
    ("add", "fwd"), ("sub", "fwd"), ("absdiff", "fwd"), ("mult", "fwd"),
    ("concat", None), ("rconcat", None),
    ("add", "RR"), ("absdiff", "RR"), ("mult", "RR"),
]


def solve_arith_transformation(prompt: str) -> str | None:
    """Independently solve a hidden-operator arithmetic transformation puzzle.

    Returns the predicted answer, or None if any operator is under-determined or
    the query operator was never demonstrated.
    """
    body = prompt.split("examples:\n", 1)[-1]
    lines = [ln for ln in body.splitlines() if " = " in ln]
    q = re.search(r"determine the result for: (.+)$", prompt)
    if not lines or not q:
        return None
    query = q.group(1)
    if len(query) != 5:
        return None
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for ln in lines:
        lhs, rhs = ln.split(" = ", 1)
        if len(lhs) != 5:
            return None
        groups.setdefault(lhs[2], []).append((lhs[:2], lhs[3:5], rhs))
    q_op = query[2]
    if q_op not in groups:
        return None
    rule = None
    for op, mode in _ARITH_CANDS:
        if all(_arith(op, mode, sa, sb) == rhs for sa, sb, rhs in groups[q_op]):
            rule = (op, mode)
            break
    if rule is None:
        return None
    return _arith(rule[0], rule[1], query[:2], query[3:5])


def solve_transformation(prompt: str) -> tuple[str | None, bool]:
    """Return (answer, ambiguous). ambiguous=True if >1 rule class fits the
    examples (then the puzzle is ill-posed and the gold may be arbitrary)."""
    body = prompt.split("examples:\n", 1)[-1]
    lines = [ln for ln in body.splitlines() if " = " in ln]
    q = re.search(r"determine the result for: (.+)$", prompt)
    if not lines or not q:
        return None, False
    pairs = [tuple(ln.split(" = ", 1)) for ln in lines]
    query = q.group(1)
    solutions = set()

    # deletion
    cands = []
    for lhs, rhs in pairs:
        lc, rc = Counter(lhs), Counter(rhs)
        cands.append({ch for ch, cnt in lc.items() if rc.get(ch, 0) < cnt})
    common = set.intersection(*cands) if cands and cands[0] else set()
    if common and all("".join(c for c in l if c not in common) == r for l, r in pairs):
        solutions.add("".join(c for c in query if c not in common))

    # substitution
    if all(len(l) == len(r) for l, r in pairs):
        m: dict[str, str] = {}
        ok = True
        for lhs, rhs in pairs:
            for a, b in zip(lhs, rhs):
                if m.setdefault(a, b) != b:
                    ok = False
                    break
            if not ok:
                break
        if ok and all("".join(m[c] for c in l) == r for l, r in pairs):
            try:
                solutions.add("".join(m[c] for c in query))
            except KeyError:
                pass

    # position filter
    if len({len(l) for l, _ in pairs}) == 1:
        n = len(pairs[0][0])
        cand_idx = []
        for r in range(1, n + 1):
            cand_idx.append(list(range(r)))
            cand_idx.append(list(range(n - r, n)))
        cand_idx.append([i for i in range(n) if i % 2 == 0])
        cand_idx.append([i for i in range(n) if i % 2 == 1])
        for idx in cand_idx:
            if all("".join(l[i] for i in idx) == r for l, r in pairs):
                if all(i < len(query) for i in idx):
                    solutions.add("".join(query[i] for i in idx))

    # affix trim
    diffs = []
    ok = True
    for l, r in pairs:
        if r not in l:
            ok = False
            break
        i = l.index(r)
        diffs.append((l[:i], l[i + len(r):]))
    if ok and diffs and len({d[0] for d in diffs}) == 1 and len({d[1] for d in diffs}) == 1:
        p, s = diffs[0]
        if (p or s) and query.startswith(p) and query.endswith(s):
            solutions.add(query[len(p): len(query) - len(s) if s else None])

    if not solutions:
        return None, False
    return next(iter(solutions)), len(solutions) > 1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def verify_row(row: dict) -> list[str]:
    """Return a list of problem tags for this row (empty = clean)."""
    fam = row["type"]
    cot = str(row["generated_cot"])
    answer = str(row["answer"]).strip()
    problems: list[str] = []

    s = check_structure(cot)
    if s:
        problems.append(f"struct:{s}")

    boxed = extract_last_boxed(cot)
    c = check_coherence(boxed or "", answer, fam)
    if c:
        problems.append(c)

    # correctness
    if fam == "gravity":
        pred = solve_gravity(row["prompt"])
        if pred and not _close(pred, answer):
            problems.append(f"wrong_answer:{pred}vs{answer}")
    elif fam == "unit_conversion":
        pred = solve_unit(row["prompt"])
        if pred and not _close(pred, answer):
            problems.append(f"wrong_answer:{pred}vs{answer}")
    elif fam == "numeral":
        pred = solve_numeral(row["prompt"])
        if pred is not None and pred != answer:
            problems.append(f"wrong_answer:{pred}vs{answer}")
    elif fam == "cipher":
        pred = solve_cipher(row["prompt"])
        if pred is not None and pred != answer:
            problems.append(f"wrong_answer:{pred}vs{answer}")
    elif fam == "bit_manipulation":
        pred = solve_bit(row["prompt"])
        if pred is not None and pred != answer:
            problems.append(f"wrong_answer:{pred}vs{answer}")
    elif fam == "transformation":  # hidden-operator arithmetic (real distribution)
        pred = solve_arith_transformation(row["prompt"])
        if pred is not None and pred != answer:
            problems.append(f"wrong_answer:{pred}vs{answer}")
    else:  # legacy transformation sub-types (cryptarithm_*/equation_numeric_*)
        pred, ambiguous = solve_transformation(row["prompt"])
        if ambiguous:
            problems.append("ambiguous_rule")
        elif pred is not None and pred != answer:
            problems.append(f"wrong_answer:{pred}vs{answer}")

    return problems


def main() -> None:
    import csv as _csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--show", type=int, default=8, help="examples per problem class")
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    per_family_total: Counter = Counter()
    per_family_bad: Counter = Counter()
    problem_counts: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for r in rows:
        fam = r["type"]
        per_family_total[fam] += 1
        probs = verify_row(r)
        if probs:
            per_family_bad[fam] += 1
        for p in probs:
            key = p.split(":", 1)[0]
            problem_counts[key] += 1
            if len(examples[key]) < args.show:
                examples[key].append(f"{fam}/{r['id']}: {p}")

    print(f"=== {args.csv} — {len(rows)} rows ===\n")
    print(f"{'family':<26} {'total':>6} {'bad':>6} {'bad%':>6}")
    for fam in sorted(per_family_total):
        t = per_family_total[fam]
        b = per_family_bad[fam]
        print(f"{fam:<26} {t:>6} {b:>6} {100*b/t:>5.1f}%")
    total = len(rows)
    bad = sum(per_family_bad.values())
    print(f"{'TOTAL':<26} {total:>6} {bad:>6} {100*bad/total:>5.1f}%\n")

    print("=== problem classes ===")
    for k, c in problem_counts.most_common():
        print(f"  {k:<22} {c}")
    print("\n=== examples ===")
    for k in problem_counts:
        print(f"\n[{k}]")
        for ex in examples[k]:
            print(f"  {ex}")


if __name__ == "__main__":
    main()
