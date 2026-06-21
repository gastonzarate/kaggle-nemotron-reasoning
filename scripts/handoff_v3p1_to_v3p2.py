#!/usr/bin/env python3
"""Handoff pipeline: when v3p1 training completes, automate the transition to v3p2.

Steps:
  1. Wait for v3p1 kernel to reach a terminal state.
  2. If COMPLETE: download its output (sft_adapter dir).
  3. Upload the adapter as a Kaggle Dataset (gastonz195/v3p1-adapter).
  4. Push the v3p2 kernel (already prepared in notebooks/my_kernels/train-structured-v3p2/).
  5. Submit v3p1 to the competition (so we have its LB score ASAP).
  6. Optionally wait for v3p2 to complete and submit it too.

All logs go to logs/handoff_v3p1_v3p2.log so the user can pick up later.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "logs" / "handoff_v3p1_v3p2.log"
V3P1_KERNEL = "gastonz195/train-structured-v3p1"
V3P2_KERNEL = "gastonz195/train-structured-v3p2"
V3P2_DIR = ROOT / "notebooks" / "my_kernels" / "train-structured-v3p2"
V3P1_ADAPTER_DATASET_DIR = ROOT / "data" / "kaggle_datasets" / "v3p1-adapter"
V3P1_OUTPUT_DIR = ROOT / "outputs" / "train-structured-v3p1"
COMPETITION = "nvidia-nemotron-model-reasoning-challenge"


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [{level}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run(cmd: list[str] | str, timeout: int = 1800) -> tuple[int, str]:
    is_shell = isinstance(cmd, str)
    proc = subprocess.run(cmd, shell=is_shell, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def kernel_status(slug: str) -> str:
    rc, out = run(["uv", "run", "kaggle", "kernels", "status", slug])
    if rc != 0:
        return "UNKNOWN"
    for s in ("RUNNING", "COMPLETE", "ERROR", "QUEUED", "CANCEL_ACKNOWLEDGED"):
        if s in out:
            return s
    return "UNKNOWN"


def wait_for_terminal(slug: str, poll_sec: int = 120, max_minutes: int = 720) -> str:
    """Wait until kernel reaches a terminal status."""
    start = time.time()
    last = None
    terminal = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}
    while True:
        st = kernel_status(slug)
        if st != last:
            log(f"{slug}: {st}")
            last = st
        if st in terminal:
            return st
        if (time.time() - start) > max_minutes * 60:
            log(f"max wait exceeded ({max_minutes}m) for {slug}", "WARN")
            return st
        time.sleep(poll_sec)


def step_download_v3p1() -> bool:
    log("Step 1: download v3p1 output")
    V3P1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rc, out = run(["uv", "run", "kaggle", "kernels", "output", V3P1_KERNEL, "-p", str(V3P1_OUTPUT_DIR)])
    log(f"download rc={rc}; last: {out.strip()[-300:]}")
    sft = V3P1_OUTPUT_DIR / "sft_adapter"
    sub_zip = V3P1_OUTPUT_DIR / "submission.zip"
    if sft.exists() and (sft / "adapter_config.json").exists():
        log("v3p1 sft_adapter present")
        return True
    if sub_zip.exists():
        # Sometimes the adapter is only in submission.zip
        log("sft_adapter missing but submission.zip present — extracting")
        import zipfile
        sft.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(sub_zip) as zf:
            zf.extractall(sft)
        if (sft / "adapter_config.json").exists():
            return True
    log("ERROR: no adapter found in v3p1 output", "ERROR")
    return False


def step_upload_v3p1_as_dataset() -> bool:
    log("Step 2: upload v3p1 sft_adapter as Kaggle dataset")
    V3P1_ADAPTER_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    target = V3P1_ADAPTER_DATASET_DIR / "sft_adapter"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(V3P1_OUTPUT_DIR / "sft_adapter", target)
    meta = {
        "title": "v3p1 LoRA adapter",
        "id": "gastonz195/v3p1-adapter",
        "licenses": [{"name": "CC0-1.0"}],
        "description": "LoRA adapter from gastonz195/train-structured-v3p1: 1 epoch over the 9000-row programmatic v3 dataset for Nemotron Reasoning Challenge. Used as starting point for v3p2 (second-epoch continuation).",
    }
    (V3P1_ADAPTER_DATASET_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    rc, out = run(["uv", "run", "kaggle", "datasets", "create", "-p", str(V3P1_ADAPTER_DATASET_DIR)])
    log(f"upload rc={rc}; last: {out.strip()[-300:]}")
    return rc == 0 or "already exists" in out


def step_push_v3p2() -> bool:
    log("Step 3: push v3p2 kernel")
    rc, out = run(["uv", "run", "kaggle", "kernels", "push", "-p", str(V3P2_DIR)])
    log(f"push rc={rc}; last: {out.strip()[-300:]}")
    return rc == 0 and "successfully pushed" in out


def step_submit_v3p1() -> bool:
    log("Step 4: submit v3p1 to competition")
    rc, out = run([
        "uv", "run", "kaggle", "competitions", "submit", COMPETITION,
        "-k", V3P1_KERNEL, "-v", "1", "-f", "submission.zip",
        "-m", "train-structured-v3p1: 1 epoch over programmatic v3 dataset (9000 rows, 100% synthetic)",
    ])
    log(f"submit rc={rc}; last: {out.strip()[-300:]}")
    return rc == 0


def main():
    log("=== handoff v3p1 → v3p2 started ===")
    st = wait_for_terminal(V3P1_KERNEL)
    if st != "COMPLETE":
        log(f"v3p1 ended with {st}, aborting handoff", "ERROR")
        return 1
    if not step_download_v3p1():
        return 1
    if not step_submit_v3p1():
        log("submit failed but continuing with v3p2 prep", "WARN")
    if not step_upload_v3p1_as_dataset():
        log("dataset upload failed, cannot push v3p2", "ERROR")
        return 1
    if not step_push_v3p2():
        log("v3p2 push failed", "ERROR")
        return 1
    log("v3p2 pushed; monitoring until terminal")
    st2 = wait_for_terminal(V3P2_KERNEL)
    log(f"v3p2 final status: {st2}")
    if st2 == "COMPLETE":
        log("Step 5: submit v3p2 to competition")
        rc, out = run([
            "uv", "run", "kaggle", "competitions", "submit", COMPETITION,
            "-k", V3P2_KERNEL, "-v", "1", "-f", "submission.zip",
            "-m", "train-structured-v3p2: continuation of v3p1 with second epoch (lr=1e-4)",
        ])
        log(f"v3p2 submit rc={rc}; last: {out.strip()[-300:]}")
    log("=== handoff finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
