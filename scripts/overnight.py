#!/usr/bin/env python3
"""Overnight orchestrator.

Coordinates the three remaining concerns without human supervision:

1. Wait for train-structured-v1 v1 (wrong DATASET_PATH) to ERROR.
2. Push v2 of train-structured-v1 with fixed path.
3. Wait for eval-holdout-baseline-v1 to COMPLETE.
4. Download eval predictions, score locally, write per-family report.
5. Wait for train-structured-v1 v2 to COMPLETE.
6. Auto-submit to competition.
7. Wait for the new submission to score, log it.
8. Write OVERNIGHT_REPORT.md.

Everything logs to logs/overnight.log. Failures don't abort the script;
they get recorded and the next phase continues if possible.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "logs" / "overnight.log"
REPORT_FILE = ROOT / "OVERNIGHT_REPORT.md"

EVAL_KERNEL = "gastonz195/eval-holdout-baseline-v1"
TRAIN_KERNEL = "gastonz195/train-structured-v1"
TRAIN_KERNEL_DIR = ROOT / "notebooks" / "my_kernels" / "train-structured-v1"
COMPETITION = "nvidia-nemotron-model-reasoning-challenge"

EVENT_LOG: list[dict] = []


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [{level}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    EVENT_LOG.append({"ts": ts, "level": level, "msg": msg})


def run(cmd: list[str] | str, capture: bool = True, timeout: int = 600) -> tuple[int, str]:
    if isinstance(cmd, str):
        proc = subprocess.run(cmd, shell=True, capture_output=capture, text=True, timeout=timeout)
    else:
        proc = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def kernel_status(slug: str) -> str:
    """Returns 'RUNNING', 'COMPLETE', 'ERROR', 'QUEUED', or 'UNKNOWN'."""
    rc, out = run(["uv", "run", "kaggle", "kernels", "status", slug])
    if rc != 0:
        return "UNKNOWN"
    for status in ("RUNNING", "COMPLETE", "ERROR", "QUEUED"):
        if status in out:
            return status
    return "UNKNOWN"


def wait_for_status(slug: str, terminal_statuses: set[str], poll_sec: int = 90, max_minutes: int = 600) -> str:
    """Polls every poll_sec until status is in terminal_statuses or max_minutes elapses."""
    start = time.time()
    last = None
    while True:
        st = kernel_status(slug)
        if st != last:
            log(f"{slug}: status={st}")
            last = st
        if st in terminal_statuses:
            return st
        if (time.time() - start) > max_minutes * 60:
            log(f"{slug}: max wait exceeded ({max_minutes} min)", "WARN")
            return st
        time.sleep(poll_sec)


def phase_wait_for_train_v1_to_error() -> str:
    log("PHASE 1: waiting for train-structured-v1 v1 to fail (wrong DATASET_PATH)")
    st = wait_for_status(TRAIN_KERNEL, terminal_statuses={"ERROR", "COMPLETE"}, poll_sec=60, max_minutes=60)
    log(f"PHASE 1 done. train-structured-v1 v1 status: {st}")
    return st


def phase_push_train_v2() -> bool:
    log("PHASE 2: pushing train-structured-v1 v2 (with fixed DATASET_PATH)")
    for attempt in range(5):
        rc, out = run(["uv", "run", "kaggle", "kernels", "push", "-p", str(TRAIN_KERNEL_DIR)])
        log(f"push attempt {attempt + 1}: rc={rc}, out={out.strip()[-300:]}")
        if rc == 0 and "successfully pushed" in out:
            log("PHASE 2 ok: train-structured-v1 v2 pushed")
            return True
        if "Maximum batch GPU session count" in out:
            log("GPU quota still full, retry in 3 min", "WARN")
            time.sleep(180)
            continue
        log(f"push failed unexpectedly: {out}", "ERROR")
        time.sleep(60)
    log("PHASE 2 FAILED: could not push v2 after 5 attempts", "ERROR")
    return False


def phase_wait_for_eval() -> tuple[str, Path | None]:
    log("PHASE 3: waiting for eval-holdout-baseline-v1 to COMPLETE")
    st = wait_for_status(EVAL_KERNEL, terminal_statuses={"COMPLETE", "ERROR"}, poll_sec=90, max_minutes=180)
    log(f"PHASE 3 done. eval-holdout-baseline-v1 status: {st}")
    if st != "COMPLETE":
        return st, None
    out_dir = ROOT / "outputs" / "eval-holdout-baseline-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rc, out = run(["uv", "run", "kaggle", "kernels", "output", EVAL_KERNEL, "-p", str(out_dir)], timeout=1800)
    log(f"output download rc={rc}, last: {out.strip()[-200:]}")
    return st, out_dir


def phase_score_eval(out_dir: Path) -> dict | None:
    log("PHASE 4: scoring eval predictions locally")
    pred_csv = out_dir / "predictions.csv"
    if not pred_csv.exists():
        pred_csv = out_dir / "predictions_with_raw.csv"
    if not pred_csv.exists():
        log("predictions.csv not found in eval output", "ERROR")
        log(f"available files: {[p.name for p in out_dir.iterdir()]}", "ERROR")
        return None
    gold_csv = ROOT / "data" / "processed" / "holdout_with_gold.csv"
    debug_dir = ROOT / "outputs" / "eval-holdout-baseline-v1" / "debug"
    rc, out = run([
        "uv", "run", "python", "-m", "src.eval.score",
        "--predictions", str(pred_csv),
        "--gold", str(gold_csv),
        "--debug-dir", str(debug_dir),
    ])
    log(f"score rc={rc}")
    if rc != 0:
        log(f"score stderr: {out}", "ERROR")
        return None
    try:
        result = json.loads(out)
        log(f"PHASE 4 ok: overall={result['overall_accuracy']:.4f}")
        return result
    except Exception as e:
        log(f"score parse failed: {e}; raw output: {out}", "ERROR")
        return None


def phase_wait_for_train_v2() -> str:
    log("PHASE 5: waiting for train-structured-v1 v2 to COMPLETE")
    st = wait_for_status(TRAIN_KERNEL, terminal_statuses={"COMPLETE", "ERROR"}, poll_sec=180, max_minutes=600)
    log(f"PHASE 5 done. train-structured-v1 v2 status: {st}")
    return st


def phase_submit() -> dict | None:
    log("PHASE 6: submitting train-structured-v1 to competition")
    rc, out = run([
        "uv", "run", "kaggle", "competitions", "submit",
        COMPETITION,
        "-k", TRAIN_KERNEL,
        "-f", "submission.zip",
        "-m", "train-structured-v1: structured CoT + router (<r>/<m>/<c>/<l>) + 2dp half-up fix on gravity/unit_conversion",
    ])
    log(f"submit rc={rc}, out={out.strip()[-300:]}")
    if rc != 0:
        log("submission command failed", "ERROR")
        return None

    log("PHASE 7: waiting for new submission to score")
    last_status = None
    for _ in range(60):
        rc, out = run(["uv", "run", "kaggle", "competitions", "submissions", COMPETITION])
        first_line = next((l for l in out.splitlines() if "train-structured-v1" in l), None)
        if first_line:
            if "COMPLETE" in first_line:
                log(f"submission scored: {first_line.strip()}")
                return {"line": first_line.strip()}
            if last_status != first_line:
                last_status = first_line
                log(f"submission status: {first_line.strip()[-150:]}")
        time.sleep(60)
    log("submission did not finish scoring within 60 min", "WARN")
    return {"line": last_status}


def write_report(
    train_v1_status: str,
    train_v2_pushed: bool,
    eval_status: str,
    eval_result: dict | None,
    train_v2_status: str,
    submit_result: dict | None,
) -> None:
    lines = ["# Overnight run report", "", f"Generated: {datetime.now().isoformat()}", ""]
    lines.append("## Timeline summary")
    lines.append("")
    lines.append(f"- train-structured-v1 v1 (wrong path) → **{train_v1_status}**")
    lines.append(f"- train-structured-v1 v2 push → **{'OK' if train_v2_pushed else 'FAILED'}**")
    lines.append(f"- eval-holdout-baseline-v1 → **{eval_status}**")
    lines.append(f"- train-structured-v1 v2 final → **{train_v2_status}**")
    sub_line = submit_result.get("line") if submit_result else "skipped"
    lines.append(f"- submission to competition → {sub_line}")
    lines.append("")

    if eval_result:
        lines.append("## Baseline LoRA per-family accuracy (holdout 1500 rows)")
        lines.append("")
        lines.append(f"**Overall: {eval_result['overall_accuracy']:.4f}** ({eval_result['n_correct']}/{eval_result['n_holdout']})")
        lines.append("")
        lines.append("| Family | Accuracy | Correct | Total |")
        lines.append("|---|---:|---:|---:|")
        for row in eval_result["per_family"]:
            lines.append(f"| {row['family']} | {row['accuracy']:.4f} | {row['correct']} | {row['predicted']} |")
        lines.append("")

    lines.append("## Event log")
    lines.append("")
    for ev in EVENT_LOG[-200:]:
        lines.append(f"- {ev['ts']} [{ev['level']}] {ev['msg']}")
    REPORT_FILE.write_text("\n".join(lines))
    log(f"report written: {REPORT_FILE}")


def main() -> int:
    log("=== overnight orchestrator started ===")
    try:
        train_v1_status = phase_wait_for_train_v1_to_error()
        train_v2_pushed = phase_push_train_v2() if train_v1_status == "ERROR" else False
        eval_status, eval_out_dir = phase_wait_for_eval()
        eval_result = phase_score_eval(eval_out_dir) if eval_out_dir else None
        train_v2_status = phase_wait_for_train_v2() if train_v2_pushed else "skipped"
        submit_result = phase_submit() if train_v2_status == "COMPLETE" else None
        write_report(
            train_v1_status=train_v1_status,
            train_v2_pushed=train_v2_pushed,
            eval_status=eval_status,
            eval_result=eval_result,
            train_v2_status=train_v2_status,
            submit_result=submit_result,
        )
        log("=== overnight orchestrator finished ===")
        return 0
    except Exception as e:
        log(f"orchestrator crashed: {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        write_report(
            train_v1_status="?",
            train_v2_pushed=False,
            eval_status="?",
            eval_result=None,
            train_v2_status="?",
            submit_result=None,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
