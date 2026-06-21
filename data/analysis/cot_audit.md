# CoT-tong dataset bug audit

**Date**: 2026-05-29
**Source**: `dgxchen/nemotron-cot-tong` (Kaggle), downloaded `2026-05-29`.
**Files audited**:
- `data/raw/cot-tong/problem_ids_matched.csv` — 7 830 rows (the canonical training set the team trained the 0.83 baseline on).
- `data/raw/cot-tong/less_cot.csv` — 6 014 rows (a strict subset / further trimmed variant; same shape).

This is **already a cleaned descendant** of the original `huikang` CoT dataset that Ashutosh Kumar reported as having 50.5% bit-manipulation bugs in forum thread `/681745`. The 50.5% bug is **mostly gone in this version**. New bugs (precision, brace-balancing, leading-zero) are dominant instead.

Below numbers are from `problem_ids_matched.csv` (the larger, training-grade file). Numbers from `less_cot.csv` agree within ±1%.

---

## Summary table — bugs per family

| Family | n | `\boxed{}` vs gold answer mismatch | dominant root cause |
| --- | --- | --- | --- |
| bit_manipulation | 1 754 | **2 (0.1 %)** | residual; per-bit derivation in CoT also agrees |
| cipher | 1 656 | **0 (0.0 %)** | clean |
| unit_conversion | 1 070 | **1 065 (99.5 %)** | **precision**: CoT outputs 3 dp (`10.240`), gold is 2 dp (`10.25`) |
| gravity | 1 055 | **1 055 (100 %)** | **precision**: same as above |
| numeral | 730 | **0 (0.0 %)** | clean; cosmetic only (`Result:` line shows `L X` with space, but `\boxed{}` and answer both are `LX`) |
| equation_numeric_deduce | 658 | **0 (0.0 %)** | clean |
| cryptarithm_deduce | 627 | **93 (14.8 %)** | **brace bug**: `\boxed{$>>\}` — backslash escapes the closing brace, so the boxed payload reads as `$>>\}` instead of `$>>\` |
| cryptarithm_guess | 154 | **14 (9.1 %)** | same brace bug |
| equation_numeric_guess | 126 | **18 (14.3 %)** | **leading-zero strip**: CoT produces `3`, gold is `03` (2-char output convention) |
| **TOTAL** | **7 830** | **2 247 (28.7 %)** | |

Roughly **1 of every 3.5 training examples teaches the model an output that does not exact-match the gold answer.**

---

## Per-family root cause analysis

### bit_manipulation (1 754 rows)

**Bug rate: 0.1 %.** The CoT enumerates 9 boolean operators (Identity / NOT / Constant / AND / OR / XOR / AND-NOT / OR-NOT / XOR-NOT) across both column orderings (left and right), counts matches per output bit, then assembles the bit assignment. Tong's filtering already removed the broken huikang traces. Our cross-check (deriving the 8-bit output from the per-bit `Output\n0 ... = 1\n...` block at the end of the CoT) matches `\boxed{}` 100 % of the time and matches the gold answer 99.9 % of the time.

The only failing id is `ef2fe526`: the CoT correctly derives `01011011`, the gold says `01011010`. That looks like a noisy gold label (one bit flipped). It appears twice because the same id has a `-p0` duplicate row.

**Action**: keep these 1 754 rows as-is; optionally drop the 2 buggy.

### cipher (1 656 rows)

**Bug rate: 0.0 %.** Substitution cipher derivation; CoT extracts the letter map word-by-word, then applies to the query. Clean.

### unit_conversion (1 070 rows) — **MAJOR BUG**

**Bug rate: 99.5 %.** Inspection of 20 random rows: every single boxed value is rounded to **3 decimal places** while the gold answer is **2 decimal places**.

| gold | boxed |
| --- | --- |
| 10.25 | 10.240 |
| 66.97 | 66.935 |
| 37.80 | 37.793 |
| 17.75 | 17.747 |

Two compounding issues:
1. The CoT's long-division produces a 3 dp intermediate (`= 0.501`), uses that as the multiplier, gets a 3 dp final.
2. The result is **truncated**, not rounded. `10.240` should be `10.24`, but the gold says `10.25` — so the **gold itself** does proper rounding from the true 4-5 dp answer. The CoT's truncation drops information that rounding would have preserved.

This is the highest-impact bug: SFT on this teaches the model to output `10.240` when scored as `10.25` — guaranteed exact-match miss.

**Action**: regenerate with (a) higher-precision intermediate division, (b) bankers/half-up rounding at the very end, (c) **2 dp output format**.

### gravity (1 055 rows) — **MAJOR BUG**

**Bug rate: 100 %.** Same precision issue as unit_conversion. 3 dp boxed vs 2 dp gold. Worse because `d = k·t²` compounds two 3 dp truncations.

**Action**: same as unit_conversion.

### numeral (730 rows)

**Bug rate: 0.0 % on the answer.** The `Result:` line shows the digits with a stray space (`L X -> LX`) that we initially classified as a mismatch against `\boxed{LX}`. After accounting for `->` arrow and whitespace, the parsed value is identical. The model is being trained to output the right thing.

**Action**: no change required, but normalising the `Result:` line in the regeneration is nice-to-have for the parser.

### equation_numeric_deduce (658 rows)

**Bug rate: 0.0 %.** Clean. The "Result:" line and `\boxed{}` agree on `147`.

### cryptarithm_deduce (627 rows) — **brace bug**

**Bug rate: 14.8 % (93 / 627).** When the answer ends in `\` (backslash), the LaTeX output reads `\boxed{$>>\}` — but `\}` is an escaped brace, so the rendered token is `$>>\}` with the trailing `}` becoming part of the payload. The extractor that pairs braces by depth sees `boxed = "$>>\\}"` and compares against gold `$>>\`. Mismatch.

This is exactly the bug Ryan Holbrook fixed in the metric on 2026-05-17 (thread `/discussion/698106`). But the training data still teaches the model to emit the escape-prone form.

**Action**: in regeneration, wrap the answer payload in extra delimiters so the boxed content is unambiguous, e.g. `\boxed{ANSWER}` with the answer string never ending in `\`. Or post-process all answers ending in `\` by adding a trailing safe character that we strip during eval.

### cryptarithm_guess (154 rows)

**Bug rate: 9.1 %.** Same brace bug. Same action.

### equation_numeric_guess (126 rows) — **leading-zero bug**

**Bug rate: 14.3 % (18 / 126).** When the gold answer is a 2-digit number with a leading zero (e.g. `03`, `04`), the CoT computes the integer (`3`) and emits `\boxed{3}`. Gold expects `03`.

**Action**: regeneration must preserve the **output width** implied by the few-shot examples. E.g. if the examples show `40-30 = 4010`, outputs are 2-digit zero-padded; if `97-12 = 9712`, outputs are clean. The CoT must detect the convention and zero-pad.

---

## Implications for the new structured CoT

1. **bit_manipulation and cipher don't need regeneration for correctness** — but they will still benefit from the new structured-tag schema to make them easier for the model to imitate at lower step counts.
2. **unit_conversion and gravity are the highest-value targets**: ~2 100 examples currently sabotaging exact-match accuracy. A clean regeneration here should move the local eval score by several points.
3. **cryptarithm needs a brace-safe output convention** — both in the CoT and in the eval extractor.
4. **equation_numeric_guess needs zero-padding aware regeneration.**

Bug-fix priority for regeneration: `gravity` > `unit_conversion` > `cryptarithm_*` > `equation_numeric_guess` > `bit_manipulation` (drop 2) > rest (no change needed).
