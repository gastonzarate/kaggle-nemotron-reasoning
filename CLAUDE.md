# Kaggle — NVIDIA Nemotron Model Reasoning Challenge

This repo is the workspace for the **NVIDIA Nemotron Model Reasoning Challenge** on Kaggle.

- Competition page: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- Owner: `gaston@sleakops.com`
- Today (project kickoff): **2026-05-02**

> The Kaggle pages are JS-rendered, so the details below were assembled from official secondary sources (NVIDIA dev blog, Kaggle/NVIDIA LinkedIn announcements, Luma event page). Anything marked **(verify)** should be re-checked against the live Kaggle Overview / Data / Rules tabs after authenticating with the Kaggle CLI.

---

## 1. Goal

Improve the **reasoning accuracy** of the open-weights `Nemotron-3-Nano-30B` base model on a private NVIDIA reasoning benchmark, *without* full pretraining. Submissions are **LoRA adapters** that are merged onto the base model at evaluation time.

We are NOT training a model from scratch. We are shipping a small set of weights (LoRA), or a prompting/data recipe that produces those weights, and being scored on the held-out reasoning benchmark.

## 2. Key facts

| Item | Value |
| --- | --- |
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B` (hybrid Mamba-Transformer MoE, 30B total / ~3B active, 1M ctx) |
| Submission artifact | LoRA adapter, **max rank 32** (verify) |
| Allowed techniques | prompting, data filtering, synthetic data generation, RL, lightweight SFT/LoRA |
| Frameworks called out | Hugging Face, Unsloth, Axolotl, TRL |
| Compute | Google Cloud G4 VMs / NVIDIA Blackwell (provided to participants) |
| Eval | Private NVIDIA-Research reasoning benchmark (held-out) |
| License | NVIDIA Open Model License (open weights, recipes, datasets) |

### Timeline (verify on Kaggle)
- **2026-03-16** — Competition start
- **2026-04-09** — Midpoint cutoff (Open Progress Prize judged here)
- **2026-06-15** — Final submission deadline
- → Today is 2026-05-02. ~**6 weeks remaining** to the final deadline.

### Prizes (≈ $106,388 total + DGX Sparks; verify exact split)
- 1st: $25,000 + 5× DGX Spark
- 2nd: $15,000 + 2× DGX Spark
- 3rd: $5,000 + 1× DGX Spark
- Open Progress Prize (mid-competition): $5,000 + 1× DGX Spark
- 3× Open Contribution Awards (Best Data / Best RL / Best Fine-tuning): 1× DGX Spark each
- Featured-tier Kaggle medal & points

> Prize-winning entries must publish a **public methodology notebook**.

## 3. Working strategy

The space of allowed techniques is wide. Pick a small set of orthogonal bets and iterate, instead of trying everything.

Initial bets, ordered by ROI vs. effort:

1. **Prompting + system instructions** — cheapest baseline. Establish a reproducible eval harness first; nothing else matters until we can score ourselves.
2. **Synthetic reasoning data + LoRA SFT** — distill chain-of-thought traces from a stronger teacher (e.g. another Nemotron-family or open reasoning model) on math/code/logic prompts. Train rank-32 LoRA on the base.
3. **Data curation / filtering** — filter public reasoning datasets (OpenMathInstruct, OpenCodeReasoning, NuminaMath, etc.) for hard, high-quality examples.
4. **RL on verifiable rewards** — only after SFT baseline beats vanilla. GRPO/RLOO via TRL on math/code with executable rewards.

Hard rules for ourselves:
- Every change is gated by an **internal eval score** before it touches the submission.
- No closed-source teachers in the synthetic-data pipeline (license risk for the LoRA artifact).
- Keep `adapters/` reproducible: each adapter dir contains the exact training config + data recipe used.

## 4. Repo layout

```
kaggle-nemotron/
├── CLAUDE.md                 # this file
├── pyproject.toml            # uv-managed deps
├── .gitignore
├── data/
│   ├── raw/                  # pulled from Kaggle / HF — never edit
│   ├── processed/            # tokenized / filtered datasets
│   └── synthetic/            # generated reasoning traces
├── notebooks/                # exploration only; nothing load-bearing
├── src/
│   ├── data/                 # dataset loading & filtering
│   ├── training/             # SFT / LoRA / RL entrypoints
│   ├── eval/                 # internal reasoning eval harness
│   └── inference/            # adapter merge + generation
├── configs/                  # YAML/JSON training configs
├── adapters/                 # trained LoRA artifacts (gitignored except recipe.md)
└── outputs/                  # logs, eval reports, predictions
```

## 5. Environment & tooling

- Python `3.14.3` is on PATH; managed via **`uv`** (`uv 0.10.9`).
- Use `uv add <pkg>` / `uv run <cmd>`. No `pip install` directly into a global env.
- Kaggle CLI lives inside the project venv (`uv add kaggle`); credentials at `~/.kaggle/kaggle.json` (mode 600).
- Hugging Face login required for gated Nemotron weights: `huggingface-cli login`.
- Heavy training does **not** run on this Mac. Local box is for data prep, eval harness, and small LoRA debug runs only. Real runs go to the Google Cloud G4 VM provided by the competition.

## 6. Conventions

- Spanish in conversation with the user (Gastón); English in code, configs, and committed docs.
- One concern per script; no god-modules.
- Every new training run writes a recipe file (`adapters/<run_id>/recipe.md`) describing: base model, dataset(s), hyperparams, eval score, git SHA. If the recipe is missing, the run does not exist.
- Never commit `data/raw/`, `data/synthetic/`, `adapters/*/checkpoints/`, or `~/.kaggle/kaggle.json`. See `.gitignore`.

## 7. Dataset (confirmed via Kaggle CLI on 2026-05-02)

Files in `data/raw/` (628 KB zip, 3.07 MB unzipped):

| File | Rows | Cols | Notes |
| --- | --- | --- | --- |
| `train.csv` | **9 500** | `id, prompt, answer` | the only real training signal |
| `test.csv` | **3** | `id, prompt` | dummy — all 3 IDs also exist in `train.csv` (format reference only) |

→ This is a **code/artifact competition**. The 3-row test is just a schema sample; the real eval set is private and run server-side. Submission is a LoRA adapter (or a notebook that produces predictions on the hidden test).

### Prompt taxonomy (≈evenly split across 6 families; all phrased as "In Alice's Wonderland…" few-shot induction puzzles)

| Family | n | Task | Answer type |
| --- | --- | --- | --- |
| Bit manipulation | 1 602 | 8-bit binary → 8-bit binary, hidden bitwise rule (XOR/AND/OR/NOT/shift/rotate/maj/choice) given 8–10 examples | binary string |
| Gravitational constant | 1 597 | Given (t, d) pairs and `d = 0.5·g·t²`, predict d for a new t (g is hidden) | float (2 dp) |
| Unit conversion | 1 594 | Linear `Y = a·X + b` from a few examples; predict for new X | float (2 dp) |
| Substitution cipher | 1 576 | Decrypt new text given 4–6 plaintext/ciphertext pairs | English phrase |
| Numeral system | 1 576 | Convert decimal N to a hidden numeral system (Roman, base-k, …) given 3–4 examples | short string |
| Generic transformation | 1 555 | Mixed ARC-like rules | mixed |

Prompt length: 177–510 chars (median 281). Answers are short (≤39 chars, median 5). Almost certainly **exact-match** scoring after light normalization.

### Leaderboard snapshot (2026-05-02)

- Top score: **0.87** (3 teams). Big plateau of ~15 teams at 0.86. **2 575 teams entered.**
- Score range looks like accuracy ∈ [0, 1]. Two-decimal granularity → likely thousands of held-out items.
- The 0.87 plateau suggests the easy gains (prompt+small SFT) are taken; meaningful movement probably requires either (a) per-family solvers / programmatic verification at inference, or (b) higher-quality synthetic SFT data.

## 8. Open questions still to resolve

- [ ] Exact submission mechanism — adapter upload vs. Kaggle Notebook with internet off (LoRA as dataset). Confirm from the *Submissions* / *Code Requirements* tab in the browser.
- [ ] Confirmed LoRA rank cap and adapter constraints (target modules, dtype, max param count).
- [ ] Daily submission limit and team-size cap.
- [ ] External-data policy: open datasets clearly OK; closed-source LLM outputs — confirm.
- [ ] Whether the private eval set covers the same 6 families or includes new ones (matters for OOD generalization strategy).
