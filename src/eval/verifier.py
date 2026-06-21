"""Answer verifier — ported verbatim from the official submission demo.

Rule (single, family-agnostic):
- If both stored and predicted parse as floats → relative tolerance 1e-2 or absolute 1e-5.
- Otherwise → case-insensitive string equality after .strip().

This matches Kaggle's server-side metric. Do NOT add per-family logic here; that would
diverge from how submissions are actually scored.
"""

import math


def verify(stored_answer: str, predicted: str) -> bool:
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()
