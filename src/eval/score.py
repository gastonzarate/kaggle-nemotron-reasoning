"""End-to-end scorer for predictions.csv against holdout_with_gold.csv.

Inputs:
  - predictions.csv: columns `id,prediction` (or `id,raw_output` if the predictor returns raw text)
  - holdout_with_gold.csv: columns `id,prompt,answer,family`

Output:
  - dict with overall accuracy + per-family breakdown
  - if --debug-dir is passed, writes wrong.csv for inspection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval.classify_family import classify_family
from src.eval.extractor import extract_final_answer
from src.eval.verifier import verify


def score(
    predictions_csv: Path,
    holdout_with_gold_csv: Path,
    debug_dir: Path | None = None,
) -> dict:
    pred = pd.read_csv(predictions_csv)
    gold = pd.read_csv(holdout_with_gold_csv)

    if "family" not in gold.columns:
        gold["family"] = gold["prompt"].apply(classify_family)

    if "prediction" not in pred.columns:
        if "raw_output" in pred.columns:
            pred["prediction"] = pred["raw_output"].apply(extract_final_answer)
        else:
            raise ValueError(
                f"predictions.csv must have either `prediction` or `raw_output` column. "
                f"Got: {list(pred.columns)}"
            )

    merged = gold.merge(pred[["id", "prediction"]], on="id", how="inner")
    if len(merged) == 0:
        raise ValueError("No matching ids between predictions and holdout. Check id schema.")
    if len(merged) < len(gold):
        missing = len(gold) - len(merged)
        print(f"WARNING: {missing} holdout rows have no prediction (will count as wrong)")

    merged["correct"] = merged.apply(
        lambda r: verify(str(r["answer"]), str(r["prediction"])), axis=1
    )

    n_holdout = len(gold)
    n_correct = int(merged["correct"].sum())

    per_family = (
        merged.groupby("family")["correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy", "sum": "correct", "count": "predicted"})
        .reset_index()
    )

    result = {
        "overall_accuracy": n_correct / n_holdout,
        "n_holdout": n_holdout,
        "n_predicted": len(merged),
        "n_correct": n_correct,
        "per_family": per_family.to_dict(orient="records"),
    }

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        wrong = merged[~merged["correct"]].copy()
        wrong.to_csv(debug_dir / "wrong.csv", index=False)
        result["wrong_csv"] = str(debug_dir / "wrong.csv")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, default=None)
    args = parser.parse_args()

    result = score(args.predictions, args.gold, args.debug_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
