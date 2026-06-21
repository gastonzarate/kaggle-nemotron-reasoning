"""Family classifier — assigns each prompt to one of the 6 puzzle types.

Used ONLY for diagnostic per-family breakdowns. Kaggle's official metric is family-agnostic;
the breakdown helps us know where we're failing during iteration.

Classification is based on the opening sentence of each prompt, which follows a fixed template
("In Alice's Wonderland, ...").

The 6 raw families correspond to what we see in `data/raw/train.csv`. The downstream
`nemotron-cot-tong` dataset further splits two of these into 4 sub-families
(cryptarithm_deduce/guess, equation_numeric_deduce/guess) — but at the prompt level we
can't distinguish those sub-families, only at the augmented-CoT level.
"""

FAMILIES = (
    "bit_manipulation",
    "cipher",
    "gravity",
    "numeral",
    "transformation",
    "unit_conversion",
)


def classify_family(prompt: str) -> str:
    head = prompt[:200].lower()

    if "bit manipulation" in head:
        return "bit_manipulation"
    if "gravitational constant" in head:
        return "gravity"
    if "unit conversion" in head:
        return "unit_conversion"
    if "secret encryption" in head or "encryption rules are used on text" in head:
        return "cipher"
    if "numeral system" in head or "secretly converted into a different numeral" in head:
        return "numeral"
    if "transformation rules" in head:
        return "transformation"

    return "unknown"
