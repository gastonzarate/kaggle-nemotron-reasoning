# Overnight run report

Generated: 2026-05-29T08:42:24.417783

## Timeline summary

- train-structured-v1 v1 (wrong path) → **ERROR**
- train-structured-v1 v2 push → **OK**
- eval-holdout-baseline-v1 → **ERROR**
- train-structured-v1 v2 final → **COMPLETE**
- submission to competition → skipped

## Event log

- 2026-05-29 01:32:28 [INFO] === overnight orchestrator started ===
- 2026-05-29 01:32:28 [INFO] PHASE 1: waiting for train-structured-v1 v1 to fail (wrong DATASET_PATH)
- 2026-05-29 01:32:29 [INFO] gastonz195/train-structured-v1: status=RUNNING
- 2026-05-29 01:36:33 [INFO] gastonz195/train-structured-v1: status=ERROR
- 2026-05-29 01:36:33 [INFO] PHASE 1 done. train-structured-v1 v1 status: ERROR
- 2026-05-29 01:36:33 [INFO] PHASE 2: pushing train-structured-v1 v2 (with fixed DATASET_PATH)
- 2026-05-29 01:36:36 [INFO] push attempt 1: rc=0, out=Kernel version 2 successfully pushed.  Please check progress at https://www.kaggle.com/code/gastonz195/train-structured-v1
- 2026-05-29 01:36:36 [INFO] PHASE 2 ok: train-structured-v1 v2 pushed
- 2026-05-29 01:36:36 [INFO] PHASE 3: waiting for eval-holdout-baseline-v1 to COMPLETE
- 2026-05-29 01:36:36 [INFO] gastonz195/eval-holdout-baseline-v1: status=ERROR
- 2026-05-29 01:36:36 [INFO] PHASE 3 done. eval-holdout-baseline-v1 status: ERROR
- 2026-05-29 01:36:36 [INFO] PHASE 5: waiting for train-structured-v1 v2 to COMPLETE
- 2026-05-29 01:36:37 [INFO] gastonz195/train-structured-v1: status=RUNNING
- 2026-05-29 08:42:23 [INFO] gastonz195/train-structured-v1: status=COMPLETE
- 2026-05-29 08:42:23 [INFO] PHASE 5 done. train-structured-v1 v2 status: COMPLETE
- 2026-05-29 08:42:23 [INFO] PHASE 6: submitting train-structured-v1 to competition
- 2026-05-29 08:42:24 [INFO] submit rc=1, out=Code competition submissions require both the output file name and the version number
- 2026-05-29 08:42:24 [ERROR] submission command failed