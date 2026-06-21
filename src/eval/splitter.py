"""Deterministic train/holdout splitter — stratified by puzzle family.

Reads `data/raw/train.csv` and produces:
- `data/processed/train_split.csv`  — what we feed to training
- `data/processed/holdout.csv`      — what we hold back for local eval
- `data/processed/holdout_with_gold.csv` — same as above but keeps `answer` (only for local verifier; NEVER ship to Kaggle)

`holdout.csv` mirrors the schema of `test.csv` (id, prompt only) so it plugs into the official
submission notebook's inference path unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.classify_family import classify_family


def make_split(
    train_csv: Path,
    out_dir: Path,
    holdout_size: int = 1500,
    seed: int = 42,
) -> dict:
    df = pd.read_csv(train_csv)
    df["family"] = df["prompt"].apply(classify_family)

    per_family = holdout_size // df["family"].nunique()

    holdout_chunks = []
    for fam, group in df.groupby("family"):
        n = min(per_family, len(group))
        holdout_chunks.append(group.sample(n=n, random_state=seed))
    holdout = pd.concat(holdout_chunks).sort_values("id").reset_index(drop=True)

    train = df.drop(holdout.index).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train_split.csv", index=False)
    holdout[["id", "prompt"]].to_csv(out_dir / "holdout.csv", index=False)
    holdout.to_csv(out_dir / "holdout_with_gold.csv", index=False)

    return {
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "holdout_by_family": holdout["family"].value_counts().to_dict(),
        "seed": seed,
    }


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    stats = make_split(
        train_csv=root / "data" / "raw" / "train.csv",
        out_dir=root / "data" / "processed",
    )
    print(json.dumps(stats, indent=2))
