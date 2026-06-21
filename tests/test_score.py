import json
from pathlib import Path

import pandas as pd

from src.eval.score import score


def _setup_fake_data(tmp_path: Path):
    gold = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "prompt": [
                "In Alice's Wonderland, a secret bit manipulation rule...",
                "In Alice's Wonderland, the gravitational constant has been...",
                "In Alice's Wonderland, secret encryption rules are used on text...",
                "In Alice's Wonderland, a secret unit conversion is applied...",
            ],
            "answer": ["11010011", "134.5", "the queen sleeps", "16.13"],
            "family": ["bit_manipulation", "gravity", "cipher", "unit_conversion"],
        }
    )
    gold_path = tmp_path / "holdout_with_gold.csv"
    gold.to_csv(gold_path, index=False)
    return gold, gold_path


def test_score_all_correct(tmp_path):
    gold, gold_path = _setup_fake_data(tmp_path)
    pred = pd.DataFrame(
        {
            "id": gold["id"],
            "prediction": gold["answer"],
        }
    )
    pred_path = tmp_path / "preds.csv"
    pred.to_csv(pred_path, index=False)

    result = score(pred_path, gold_path)
    assert result["overall_accuracy"] == 1.0
    assert result["n_correct"] == 4
    assert result["n_holdout"] == 4
    assert len(result["per_family"]) == 4


def test_score_all_wrong(tmp_path):
    gold, gold_path = _setup_fake_data(tmp_path)
    pred = pd.DataFrame(
        {
            "id": gold["id"],
            "prediction": ["wrong"] * 4,
        }
    )
    pred_path = tmp_path / "preds.csv"
    pred.to_csv(pred_path, index=False)

    result = score(pred_path, gold_path)
    assert result["overall_accuracy"] == 0.0
    assert result["n_correct"] == 0


def test_score_extracts_from_raw_output(tmp_path):
    gold, gold_path = _setup_fake_data(tmp_path)
    pred = pd.DataFrame(
        {
            "id": gold["id"],
            "raw_output": [
                r"<think>analyzing bits</think> \boxed{11010011}",
                r"<think>g~9.8</think> \boxed{134.5}",
                r"<think>decoding</think> \boxed{the queen sleeps}",
                r"<think>linear fit</think> \boxed{16.13}",
            ],
        }
    )
    pred_path = tmp_path / "preds.csv"
    pred.to_csv(pred_path, index=False)

    result = score(pred_path, gold_path)
    assert result["overall_accuracy"] == 1.0


def test_score_per_family_breakdown(tmp_path):
    gold, gold_path = _setup_fake_data(tmp_path)
    pred = pd.DataFrame(
        {
            "id": gold["id"],
            "prediction": ["11010011", "wrong", "the queen sleeps", "wrong"],
        }
    )
    pred_path = tmp_path / "preds.csv"
    pred.to_csv(pred_path, index=False)

    result = score(pred_path, gold_path)
    assert result["overall_accuracy"] == 0.5

    per_family = {row["family"]: row["accuracy"] for row in result["per_family"]}
    assert per_family["bit_manipulation"] == 1.0
    assert per_family["gravity"] == 0.0
    assert per_family["cipher"] == 1.0
    assert per_family["unit_conversion"] == 0.0
