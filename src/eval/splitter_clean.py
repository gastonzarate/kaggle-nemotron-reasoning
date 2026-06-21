"""Build a CLEAN holdout — train.csv ids that were NOT in CoT-tong (so never seen by our LoRAs).

The original `splitter.py` carved a stratified-by-family 1500 holdout out of train.csv.
Problem: ~64% of those ids ARE in `dgxchen/nemotron-cot-tong`, which is what we
trained on. So that holdout is contaminated.

This script does the opposite: takes the 3 329 train.csv ids that the public teacher
LLM could NOT solve (and thus were filtered OUT of CoT-tong), and uses those as a
clean evaluation set. Caveat: this set is biased toward "hard" problems, so the
absolute accuracy will be lower than train.csv distribution.

Output:
    data/processed/holdout_clean.csv          (id, prompt)
    data/processed/holdout_clean_with_gold.csv (id, prompt, answer, family)
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.eval.classify_family import classify_family


def build_clean_holdout(
    train_csv: Path,
    cot_tong_csv: Path,
    out_dir: Path,
) -> dict:
    train = pd.read_csv(train_csv)

    cot_ids: set[str] = set()
    with open(cot_tong_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_key = next((k for k in row if k.lstrip("﻿") == "id"), None)
            id_val = row.get(id_key) if id_key else row.get("id")
            if id_val:
                cot_ids.add(id_val.strip())

    train["was_trained_on"] = train["id"].isin(cot_ids)
    clean = train[~train["was_trained_on"]].copy()
    clean["family"] = clean["prompt"].apply(classify_family)
    clean = clean.drop(columns=["was_trained_on"]).sort_values("id").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    clean[["id", "prompt"]].to_csv(out_dir / "holdout_clean.csv", index=False)
    clean.to_csv(out_dir / "holdout_clean_with_gold.csv", index=False)

    return {
        "train_csv_rows": len(train),
        "cot_tong_unique_ids": len(cot_ids),
        "clean_holdout_rows": len(clean),
        "by_family": clean["family"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    stats = build_clean_holdout(
        train_csv=root / "data" / "raw" / "train.csv",
        cot_tong_csv=root / "data" / "raw" / "cot-tong" / "problem_ids_matched.csv",
        out_dir=root / "data" / "processed",
    )
    print(json.dumps(stats, indent=2))
