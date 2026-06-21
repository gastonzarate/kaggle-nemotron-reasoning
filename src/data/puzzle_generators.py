"""Pure-Python random-puzzle generators — one per family.

Each `gen_<family>_puzzle(rng)` returns a dict with:
- `prompt`: the formatted puzzle as Nemotron will see it
- `answer`: the gold answer string (exactly what the official metric expects)
- `parsed`: the structured data ready to feed into the matching CoT generator

Determinism is via the `rng: random.Random` argument. Use a seed for reproducibility.
"""

from __future__ import annotations

import random
import string
from decimal import Decimal
from typing import Optional

from src.data.programmatic_cot import (
    ARITH_CANDIDATES,
    arith_result,
    bit_query_unique,
    solve_arith_operator,
    to_base_k,
    to_roman,
)


# ─────────────────────────────────────────────────────────────────────────────
# gravity:  d = 0.5 · g · t²  with g ∈ [5, 25] (Wonderland gravity)
# ─────────────────────────────────────────────────────────────────────────────


def gen_gravity_puzzle(rng: random.Random) -> dict:
    g = Decimal(str(round(rng.uniform(4.0, 25.0), 2)))
    n_examples = rng.randint(3, 5)

    pairs: list[tuple[Decimal, Decimal]] = []
    for _ in range(n_examples):
        t = Decimal(str(round(rng.uniform(0.5, 5.0), 2)))
        d = (Decimal("0.5") * g * t * t).quantize(Decimal("0.01"))
        pairs.append((t, d))

    query_t = Decimal(str(round(rng.uniform(0.5, 5.0), 2)))
    gold_d = (Decimal("0.5") * g * query_t * query_t).quantize(Decimal("0.01"))

    prompt = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        + "\n".join(f"For t = {p[0]}s, distance = {p[1]} m" for p in pairs)
        + f"\nNow, determine the falling distance for t = {query_t}s given d = 0.5*g*t^2."
    )
    return {
        "prompt": prompt,
        "answer": str(gold_d),
        "type": "gravity",
        "parsed": {"pairs": pairs, "query_t": query_t},
    }


# ─────────────────────────────────────────────────────────────────────────────
# unit_conversion:  Y = a·X + b
# ─────────────────────────────────────────────────────────────────────────────


def gen_unit_conversion_puzzle(rng: random.Random) -> dict:
    # Most train.csv examples look proportional (b≈0); inject some affine variety too
    a = Decimal(str(round(rng.uniform(0.3, 5.0), 4)))
    b = Decimal("0") if rng.random() < 0.7 else Decimal(str(round(rng.uniform(-5.0, 5.0), 2)))
    n_examples = rng.randint(3, 5)

    pairs: list[tuple[Decimal, Decimal]] = []
    used_x = set()
    while len(pairs) < n_examples:
        x = Decimal(str(round(rng.uniform(1.0, 100.0), 2)))
        if x in used_x:
            continue
        used_x.add(x)
        y = (a * x + b).quantize(Decimal("0.01"))
        pairs.append((x, y))

    query_x = Decimal(str(round(rng.uniform(1.0, 100.0), 2)))
    while query_x in used_x:
        query_x = Decimal(str(round(rng.uniform(1.0, 100.0), 2)))
    gold_y = (a * query_x + b).quantize(Decimal("0.01"))

    prompt = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
        + "\n".join(f"{p[0]} m becomes {p[1]}" for p in pairs)
        + f"\nNow, convert the following measurement: {query_x} m"
    )
    return {
        "prompt": prompt,
        "answer": str(gold_y),
        "type": "unit_conversion",
        "parsed": {"pairs": pairs, "query_x": query_x},
    }


# ─────────────────────────────────────────────────────────────────────────────
# numeral:  Roman (70%)  /  base-k (30%, k ∈ {2,3,7,8,16})
# ─────────────────────────────────────────────────────────────────────────────


def gen_numeral_puzzle(rng: random.Random) -> dict:
    use_roman = rng.random() < 0.7
    if use_roman:
        # Pick 3-5 example numbers in 1..3999 (Roman valid range)
        ns = sorted(rng.sample(range(1, 1000), rng.randint(3, 5)))
        pairs = [(n, to_roman(n)) for n in ns]
        query_n = rng.randint(1, 1000)
        while query_n in ns:
            query_n = rng.randint(1, 1000)
        gold = to_roman(query_n)
    else:
        k = rng.choice([2, 3, 7, 8, 16])
        ns = sorted(rng.sample(range(1, 200), rng.randint(3, 5)))
        pairs = [(n, to_base_k(n, k)) for n in ns]
        query_n = rng.randint(1, 200)
        while query_n in ns:
            query_n = rng.randint(1, 200)
        gold = to_base_k(query_n, k)

    prompt = (
        "In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
        "Some examples are given below:\n"
        + "\n".join(f"{p[0]} -> {p[1]}" for p in pairs)
        + f"\nNow, write the number {query_n} in the Wonderland numeral system."
    )
    return {
        "prompt": prompt,
        "answer": gold,
        "type": "numeral",
        "parsed": {"pairs": pairs, "query_n": query_n},
    }


# ─────────────────────────────────────────────────────────────────────────────
# cipher: monoalphabetic substitution with a Wonderland-themed vocabulary
# ─────────────────────────────────────────────────────────────────────────────

_VOCAB = [
    "king", "queen", "rabbit", "alice", "cheshire", "hatter", "wonder", "dragon",
    "knight", "princess", "wizard", "castle", "palace", "garden", "forest", "river",
    "sees", "watches", "finds", "follows", "discovers", "studies", "writes", "reads",
    "creates", "dreams", "sleeps", "speaks", "hears", "knows", "loves", "fears",
    "ancient", "clever", "golden", "magical", "secret", "hidden", "mysterious",
    "the", "a", "an", "and", "or", "but", "in", "on", "near", "inside", "behind",
    "before", "after", "school", "library", "mountain", "valley", "ocean", "sky",
    "moon", "sun", "star", "wind", "rain", "fire", "night", "day", "morning",
    "evening", "story", "tale", "song", "voice", "message", "letter", "book",
    "map", "treasure", "key", "door", "window",
]


def _build_cipher_map(rng: random.Random) -> dict[str, str]:
    """Random permutation of a-z."""
    letters = list(string.ascii_lowercase)
    shuffled = letters.copy()
    rng.shuffle(shuffled)
    # avoid the identity mapping (would be a trivial puzzle)
    if shuffled == letters:
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    return dict(zip(letters, shuffled))


def _encode_text(text: str, cmap: dict[str, str]) -> str:
    return "".join(cmap.get(c, c) for c in text)


def gen_cipher_puzzle(rng: random.Random) -> dict:
    """Build a cipher puzzle where every cipher letter in the query appears in some example.

    This guarantees the substitution map built from examples covers all query letters,
    so cipher_cot can decode the full query.
    """
    cmap = _build_cipher_map(rng)
    n_examples = rng.randint(4, 6)

    def random_sentence() -> str:
        n_words = rng.randint(3, 5)
        return " ".join(rng.choice(_VOCAB) for _ in range(n_words))

    example_plain = [random_sentence() for _ in range(n_examples)]
    pairs = [(_encode_text(p, cmap), p) for p in example_plain]  # (ciphertext, plaintext)

    # Letters that appear in examples (cipher space)
    seen_cipher_chars = set()
    for ct, _ in pairs:
        seen_cipher_chars.update(c for c in ct if c != " ")

    # Build query from words whose cipher form uses only seen letters
    def build_compatible_query():
        for _ in range(50):
            sentence = random_sentence()
            cipher = _encode_text(sentence, cmap)
            cipher_letters = {c for c in cipher if c != " "}
            if cipher_letters <= seen_cipher_chars:
                return sentence
        return None

    query_plain = build_compatible_query()
    if query_plain is None:
        # Fallback: add an extra example that covers missing letters
        all_alphabet = set(string.ascii_lowercase)
        missing = all_alphabet - seen_cipher_chars
        if missing:
            # Find a word using as many missing letters as possible (in plain space)
            for word in _VOCAB:
                if any(cmap[c] in missing for c in word):
                    extra_ct = _encode_text(word, cmap)
                    pairs.append((extra_ct, word))
                    seen_cipher_chars.update(c for c in extra_ct if c != " ")
        # Try again
        query_plain = build_compatible_query() or random_sentence()

    query_cipher = _encode_text(query_plain, cmap)

    prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
        + "\n".join(f"{c} -> {p}" for c, p in pairs)
        + f"\nNow, decrypt the following text: {query_cipher}"
    )
    return {
        "prompt": prompt,
        "answer": query_plain,
        "type": "cipher",
        "parsed": {"pairs": pairs, "query": query_cipher},
    }


# ─────────────────────────────────────────────────────────────────────────────
# bit_manipulation: per-bit rule the discover_bit_rules solver can find
# ─────────────────────────────────────────────────────────────────────────────


def _apply_rule(rule_def: dict, in_bits: list[int]) -> list[int]:
    """rule_def is a list of 8 dicts describing each output bit."""
    out_bits = []
    for spec in rule_def["per_bit"]:
        kind = spec["kind"]
        if kind == "copy":
            out_bits.append(in_bits[spec["src"]])
        elif kind == "not":
            out_bits.append(1 - in_bits[spec["src"]])
        elif kind in ("and", "or", "xor", "nand", "nor", "xnor"):
            a, b = in_bits[spec["src1"]], in_bits[spec["src2"]]
            if kind == "and": out_bits.append(a & b)
            elif kind == "or": out_bits.append(a | b)
            elif kind == "xor": out_bits.append(a ^ b)
            elif kind == "nand": out_bits.append(1 - (a & b))
            elif kind == "nor": out_bits.append(1 - (a | b))
            elif kind == "xnor": out_bits.append(1 - (a ^ b))
    return out_bits


def _make_random_rule(rng: random.Random) -> dict:
    """Build a random per-bit rule that the discoverer can find."""
    per_bit = []
    for out_bit in range(8):
        # 50% copy, 20% not, 30% 2-input op
        r = rng.random()
        if r < 0.5:
            per_bit.append({"kind": "copy", "src": rng.randint(0, 7)})
        elif r < 0.7:
            per_bit.append({"kind": "not", "src": rng.randint(0, 7)})
        else:
            j = rng.randint(0, 7)
            k = rng.randint(0, 7)
            while k == j:
                k = rng.randint(0, 7)
            kind = rng.choice(["and", "or", "xor", "nand", "nor", "xnor"])
            per_bit.append({"kind": kind, "src1": min(j, k), "src2": max(j, k)})
    return {"per_bit": per_bit}


def _bits_to_byte(bits: list[int]) -> str:
    """bits[0] is LSB. Returns MSB-first 8-char string."""
    return "".join(str(b) for b in reversed(bits))


def _byte_to_bits(s: str) -> list[int]:
    return [int(s[7 - i]) for i in range(8)]


def gen_bit_manipulation_puzzle(rng: random.Random) -> dict:
    """Generate a bit-manipulation puzzle the discoverer can uniquely solve.

    To uniquely disambiguate per-bit rules (especially 2-input ops like AND/OR/XOR/NAND/etc.),
    we need examples covering all relevant bit-pair combinations. Strategy:
    - 8 single-bit-set inputs (disambiguate single-input rules)
    - 16 random multi-bit inputs (statistically cover bit-pair combos)
    - Limit to 24 total — Kaggle template uses 8-12 visible examples; we use more under the hood.
    """
    rule = _make_random_rule(rng)
    inputs_int = [1 << i for i in range(8)]
    inputs_int.extend(rng.sample(range(0, 256), 16))
    rng.shuffle(inputs_int)
    inputs_int = list(dict.fromkeys(inputs_int))[:24]  # dedupe, keep order

    pairs = []
    for v in inputs_int:
        in_str = f"{v:08b}"
        in_bits = _byte_to_bits(in_str)
        out_bits = _apply_rule(rule, in_bits)
        pairs.append((in_str, _bits_to_byte(out_bits)))

    # Random query (avoid duplicates)
    used = {v for v in inputs_int}
    while True:
        q = rng.randint(0, 255)
        if q not in used:
            break
    query_str = f"{q:08b}"
    query_bits = _byte_to_bits(query_str)
    gold_bits = _apply_rule(rule, query_bits)
    gold = _bits_to_byte(gold_bits)

    # Well-posedness gate: every output bit must be uniquely determined by the
    # examples. If some bit is under-determined (≥2 consistent gates disagree on
    # the query), bail and let the orchestrator retry — otherwise the model is
    # being asked to guess a bit it cannot infer (the residual ambiguity the full
    # bit_manipulation sweep flagged).
    if not bit_query_unique(pairs, query_str):
        raise ValueError("bit puzzle under-determined on the query")

    prompt = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
        "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly "
        "majority or choice functions.\n\nHere are some examples of input -> output:\n"
        + "\n".join(f"{i} -> {o}" for i, o in pairs)
        + f"\n\nNow, determine the output for: {query_str}"
    )
    return {
        "prompt": prompt,
        "answer": gold,
        "type": "bit_manipulation",
        "parsed": {"pairs": pairs, "query": query_str},
    }


# ─────────────────────────────────────────────────────────────────────────────
# transformation (4 sub-types matching our 4 CoT classes)
# ─────────────────────────────────────────────────────────────────────────────

_TRANSFORM_ALPHABET = string.ascii_lowercase + string.digits + "$%&@#!?<>"


def _random_string(rng: random.Random, min_len: int, max_len: int) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(_TRANSFORM_ALPHABET) for _ in range(n))


def gen_transformation_deletion(rng: random.Random, type_label: str = "cryptarithm_deduce") -> dict:
    """Pick a delete-set, produce examples where EVERY char of delete_set appears in lhs."""
    pool_no_delete = [c for c in _TRANSFORM_ALPHABET if c not in "!@#?$&"]
    delete_chars = list(rng.sample("!@#?$&", rng.randint(1, 2)))
    delete_set = set(delete_chars)

    n = rng.randint(3, 5)
    pairs: list[tuple[str, str]] = []
    while len(pairs) < n:
        chars = [rng.choice(pool_no_delete) for _ in range(rng.randint(3, 6))]
        # Ensure ALL chars of delete_set appear at least once in lhs
        chars.extend(delete_chars)
        rng.shuffle(chars)
        lhs = "".join(chars)
        rhs = "".join(c for c in lhs if c not in delete_set)
        if rhs and (lhs, rhs) not in pairs:
            pairs.append((lhs, rhs))

    # Build query similarly: contains all delete chars at least once
    q_chars = [rng.choice(pool_no_delete) for _ in range(rng.randint(3, 6))]
    q_chars.extend(delete_chars)
    rng.shuffle(q_chars)
    query = "".join(q_chars)
    ans = "".join(c for c in query if c not in delete_set)

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        + "\n".join(f"{l} = {r}" for l, r in pairs)
        + f"\nNow, determine the result for: {query}"
    )
    return {
        "prompt": prompt,
        "answer": ans,
        "type": type_label,
        "parsed": {"pairs": pairs, "query": query},
    }


def gen_transformation_substitution(rng: random.Random, type_label: str = "cryptarithm_guess") -> dict:
    """Random char→char map. Use a SMALL active alphabet so examples cover all query chars."""
    # Use only 6-10 letters so 3-5 examples can cover the whole active alphabet
    active = rng.sample(string.ascii_lowercase, rng.randint(6, 10))
    targets = list(active)
    rng.shuffle(targets)
    while targets == active:
        rng.shuffle(targets)
    mapping = dict(zip(active, targets))

    n = rng.randint(3, 5)
    pairs: list[tuple[str, str]] = []
    # Make sure each pair uses the active alphabet
    for _ in range(n):
        lhs = "".join(rng.choice(active) for _ in range(rng.randint(3, 6)))
        rhs = "".join(mapping[c] for c in lhs)
        pairs.append((lhs, rhs))

    # Ensure every active letter appears in at least one example lhs
    seen = set("".join(l for l, _ in pairs))
    for letter in active:
        if letter not in seen:
            # Pad an extra pair containing the missing letter
            extra_lhs = letter + "".join(rng.choice(active) for _ in range(2))
            extra_rhs = "".join(mapping[c] for c in extra_lhs)
            pairs.append((extra_lhs, extra_rhs))
            seen.add(letter)

    # Query uses only active letters (guaranteed coverage)
    query = "".join(rng.choice(active) for _ in range(rng.randint(3, 6)))
    answer = "".join(mapping[c] for c in query)

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        + "\n".join(f"{l} = {r}" for l, r in pairs)
        + f"\nNow, determine the result for: {query}"
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "type": type_label,
        "parsed": {"pairs": pairs, "query": query},
    }


def gen_transformation_position_filter(rng: random.Random, type_label: str = "equation_numeric_deduce") -> dict:
    """Keep positions: first-k, last-k, even, or odd."""
    n_chars = rng.randint(5, 8)
    rules = [
        ("first_k", lambda n: list(range(rng.randint(2, n - 1)))),
        ("last_k", lambda n: list(range(n - rng.randint(2, n - 1), n))),
        ("even", lambda n: [i for i in range(n) if i % 2 == 0]),
        ("odd", lambda n: [i for i in range(n) if i % 2 == 1]),
    ]
    rule_name, rule_fn = rng.choice(rules)
    keep_idx = rule_fn(n_chars)

    n_ex = rng.randint(3, 5)
    pairs = []
    for _ in range(n_ex):
        lhs = "".join(rng.choice(_TRANSFORM_ALPHABET) for _ in range(n_chars))
        rhs = "".join(lhs[i] for i in keep_idx)
        pairs.append((lhs, rhs))

    query = "".join(rng.choice(_TRANSFORM_ALPHABET) for _ in range(n_chars))
    answer = "".join(query[i] for i in keep_idx)

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        + "\n".join(f"{l} = {r}" for l, r in pairs)
        + f"\nNow, determine the result for: {query}"
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "type": type_label,
        "parsed": {"pairs": pairs, "query": query},
    }


# ─────────────────────────────────────────────────────────────────────────────
# transformation (REAL distribution): hidden-operator 2-operand arithmetic.
# `AA <op> BB` with a puzzle-local operator glyph at index 2. Matches train.csv.
# See data/analysis/transformation_taxonomy.md and arithmetic_transformation_cot.
# ─────────────────────────────────────────────────────────────────────────────

_OP_GLYPHS = list("+-*/<>|#@$%^&!?~")
_FWD_OPS = ["add", "sub", "absdiff", "mult", "concat", "rconcat"]
_RR_OPS = ["add", "absdiff", "mult"]


def gen_transformation_arithmetic(rng: random.Random, type_label: str = "transformation") -> dict:
    """Generate a hidden-operator arithmetic puzzle.

    Guarantees: LHS is always 5 chars (`AAoBB`); the query's operator appears in
    at least 2 examples (so it's recoverable); the operator assignment is
    uniquely recovered by `solve_arith_operator` for every operator used (else we
    raise to let the orchestrator retry — this keeps the dataset well-posed).
    """
    n_ops = rng.randint(1, 2)
    glyphs = rng.sample(_OP_GLYPHS, n_ops)

    assign: dict[str, tuple[str, str | None]] = {}
    for g in glyphs:
        if rng.random() < 0.3:
            assign[g] = (rng.choice(_RR_OPS), "RR")
        else:
            op = rng.choice(_FWD_OPS)
            assign[g] = (op, None if op in ("concat", "rconcat") else "fwd")

    q_glyph = rng.choice(glyphs)

    # Example operator sequence: every glyph at least once, query glyph twice.
    seq = list(glyphs) + [q_glyph]
    while len(seq) < rng.randint(3, 5):
        seq.append(rng.choice(glyphs))
    rng.shuffle(seq)

    def rand_operand() -> str:
        return f"{rng.randint(0, 99):02d}"

    pairs: list[tuple[str, str]] = []
    for g in seq:
        op, mode = assign[g]
        sa, sb = rand_operand(), rand_operand()
        rhs = arith_result(op, mode, sa, sb)
        pairs.append((sa + g + sb, rhs))

    # Query (avoid duplicating an example LHS).
    for _ in range(50):
        qa, qb = rand_operand(), rand_operand()
        query = qa + q_glyph + qb
        if query not in {l for l, _ in pairs}:
            break
    op, mode = assign[q_glyph]
    gold = arith_result(op, mode, qa, qb)

    # Well-posedness. Two guarantees, else bail and let the orchestrator retry:
    #   1. Every operator used has at least one rule consistent with its examples
    #      (so the CoT can solve it).
    #   2. The query has a UNIQUE answer: all candidate rules consistent with the
    #      query operator's examples must agree on the query output. This rejects
    #      the under-determined puzzles QA flagged — e.g. sub vs abs_diff when all
    #      examples have a>=b (they only diverge when the query has a<b), or
    #      fwd vs RR when the examples are coincidentally palindromic.
    by_op: dict[str, list[tuple[str, str, str]]] = {}
    for lhs, rhs in pairs:
        a, opc, b = lhs[:2], lhs[2], lhs[3:5]
        by_op.setdefault(opc, []).append((a, b, rhs))
    for opc, rows in by_op.items():
        if solve_arith_operator(rows) is None:
            raise ValueError("under-determined operator")
    q_rows = by_op[q_glyph]
    query_preds = {
        arith_result(op, mode, qa, qb)
        for op, mode in ARITH_CANDIDATES
        if all(arith_result(op, mode, sa, sb) == rhs for sa, sb, rhs in q_rows)
    }
    if query_preds != {gold}:
        raise ValueError("query answer not unique among consistent rules")

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        + "\n".join(f"{l} = {r}" for l, r in pairs)
        + f"\nNow, determine the result for: {query}"
    )
    return {
        "prompt": prompt,
        "answer": gold,
        "type": type_label,
        "parsed": {"pairs": pairs, "query": query},
    }


def gen_transformation_affix_trim(rng: random.Random, type_label: str = "equation_numeric_guess") -> dict:
    """Strip a constant prefix/suffix.

    To avoid colliding with the deletion rule (which would also match if the prefix/suffix
    chars never appear in the core), we force at least one prefix-char AND one suffix-char
    to also appear INSIDE the core of every example. That way 'remove all X' fails because
    X appears in the rhs too, leaving only affix-trim as a consistent rule.
    """
    # Build a prefix and suffix using only a tiny private alphabet
    affix_pool = "PSXQZ"  # uppercase-rare so very unlikely to clash with core chars
    prefix = "".join(rng.choice(affix_pool) for _ in range(rng.randint(2, 4)))
    suffix = "".join(rng.choice(affix_pool) for _ in range(rng.randint(2, 4)))

    # Core uses a different alphabet
    core_pool = string.ascii_lowercase + string.digits

    n = rng.randint(3, 5)
    pairs = []
    # Force at least one prefix-char and one suffix-char to appear in EACH core
    affix_chars = set(prefix + suffix)
    for _ in range(n):
        # Build core that includes every affix char (so deletion can't remove them)
        core_chars = list(affix_chars)
        core_chars.extend(rng.choice(core_pool) for _ in range(rng.randint(3, 5)))
        rng.shuffle(core_chars)
        core = "".join(core_chars)
        lhs = prefix + core + suffix
        rhs = core
        pairs.append((lhs, rhs))

    # Query: also has affix chars in core
    query_core_chars = list(affix_chars)
    query_core_chars.extend(rng.choice(core_pool) for _ in range(rng.randint(3, 5)))
    rng.shuffle(query_core_chars)
    query_core = "".join(query_core_chars)
    query = prefix + query_core + suffix
    answer = query_core

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        + "\n".join(f"{l} = {r}" for l, r in pairs)
        + f"\nNow, determine the result for: {query}"
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "type": type_label,
        "parsed": {"pairs": pairs, "query": query},
    }
