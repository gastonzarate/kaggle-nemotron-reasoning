"""Parsers that extract structured data from each family's prompt.

Each function takes a raw prompt string and returns a dict with the relevant fields
(`examples`, `query`, etc.) plus any family-specific extras. Returns None if the
prompt doesn't match the family's grammar.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Optional


def parse_gravity(prompt: str) -> Optional[dict]:
    """Extracts (t, d) pairs and the query t.

    Returns: {"pairs": [(Decimal, Decimal), ...], "query_t": Decimal}
    """
    pairs = re.findall(r"t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)\s*m?", prompt, re.IGNORECASE)
    if not pairs:
        return None
    query_m = re.search(r"falling distance for\s*t\s*=\s*([\d.]+)", prompt, re.IGNORECASE)
    if not query_m:
        return None
    return {
        "pairs": [(Decimal(t), Decimal(d)) for t, d in pairs],
        "query_t": Decimal(query_m.group(1)),
    }


def parse_unit_conversion(prompt: str) -> Optional[dict]:
    """Extracts (X, Y) example pairs and the query X.

    Format: '18.75 m becomes 19.91' and 'Now, convert the following measurement: 15.19 m'
    """
    pairs = re.findall(r"([\d.]+)\s*m\s+becomes\s+([\d.]+)", prompt, re.IGNORECASE)
    if not pairs:
        return None
    query_m = re.search(r"convert the following measurement:\s*([\d.]+)", prompt, re.IGNORECASE)
    if not query_m:
        return None
    return {
        "pairs": [(Decimal(x), Decimal(y)) for x, y in pairs],
        "query_x": Decimal(query_m.group(1)),
    }


def parse_numeral(prompt: str) -> Optional[dict]:
    """Extracts (decimal_n, encoded_str) example pairs and the query decimal_n.

    Format: '3 -> III' and 'Now, write the number 43 in the Wonderland numeral system.'
    """
    pairs = re.findall(r"^\s*(\d+)\s*->\s*([A-Za-z0-9]+)\s*$", prompt, re.MULTILINE)
    if not pairs:
        return None
    query_m = re.search(r"write the number\s*(\d+)\s*in the Wonderland", prompt, re.IGNORECASE)
    if not query_m:
        return None
    return {
        "pairs": [(int(n), s) for n, s in pairs],
        "query_n": int(query_m.group(1)),
    }


def parse_cipher(prompt: str) -> Optional[dict]:
    """Extracts (ciphertext, plaintext) examples and the query ciphertext.

    Format: 'wcjz ivufex cjxciu xtsnno -> king dreams inside school'
    'Now, decrypt the following text: ysu fjtcujy qvcjtuxx knaji'
    """
    pairs = re.findall(r"^\s*([a-z\s]+?)\s*->\s*([a-z\s]+?)\s*$", prompt, re.MULTILINE | re.IGNORECASE)
    pairs = [(c.strip(), p.strip()) for c, p in pairs if c.strip() and p.strip()]
    if not pairs:
        return None
    query_m = re.search(r"decrypt the following text:\s*(.+?)\s*$", prompt, re.IGNORECASE | re.MULTILINE)
    if not query_m:
        return None
    return {
        "pairs": pairs,
        "query": query_m.group(1).strip(),
    }


def parse_bit_manipulation(prompt: str) -> Optional[dict]:
    """Extracts (input_byte, output_byte) example pairs and the query input.

    Format: '01010001 -> 11011101'
    'Now, determine the output for: 00110100'
    """
    pairs = re.findall(r"^\s*([01]{8})\s*->\s*([01]{8})\s*$", prompt, re.MULTILINE)
    if not pairs:
        return None
    query_m = re.search(r"determine the output for:\s*([01]{8})", prompt)
    if not query_m:
        return None
    return {
        "pairs": pairs,
        "query": query_m.group(1),
    }


def parse_transformation(prompt: str) -> Optional[dict]:
    """Extracts (lhs, rhs) example pairs and the query lhs.

    Generic format covering cryptarithm and equation_numeric (which share the
    'A = B' surface form in train.csv).

    Format: '$?:>` = $?>`' and 'Now, determine the result for: $>:>\\'
    """
    pairs = re.findall(r"^\s*(\S.*?)\s*=\s*(\S.*?)\s*$", prompt, re.MULTILINE)
    # filter intro lines like "Now, determine the result for:" by demanding both sides be present
    pairs = [(l, r) for l, r in pairs if l and r and "->" not in l and "->" not in r]
    if not pairs:
        return None
    query_m = re.search(r"determine the result for:\s*(.+?)\s*$", prompt, re.IGNORECASE | re.MULTILINE)
    if not query_m:
        return None
    return {
        "pairs": pairs,
        "query": query_m.group(1).strip(),
    }
