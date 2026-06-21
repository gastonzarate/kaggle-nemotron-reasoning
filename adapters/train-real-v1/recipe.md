# train-real-v1 — corpus REAL (receta del Progress Prize winner)

**Pusheado**: 2026-06-10 · **Kernel**: https://www.kaggle.com/code/gastonz195/train-real-v1 (v1)

## Hipótesis
El gap 0.84↔0.56 era **prompts sintéticos vs reales** (+ estilo). Receta del ganador (huikang):
entrenar sobre los prompts REALES de train.csv con reasoning de solver verificado + completion masking.

## Dataset: `gastonz195/nemotron-real-corpus-v1` (8 429 rows)
Construido por `src/data/build_real_corpus.py`:
- **6 635 filas "solver"**: nuestros solvers programáticos sobre el prompt real; emitidas SOLO si
  predicted == gold real bajo la métrica oficial (rule_found). answer = valor derivado (coherente).
- **1 794 filas "teacher"**: fallback cot-tong (corpus del 0.84) para lo no resuelto. answer = gold.
- **1 071 skipped** (sin solver ni teacher): transformation 924, bit 142, unit 5.

| familia | total | solver% | teacher | skipped |
|---|---|---|---|---|
| gravity | 1597 | 100% | 0 | 0 |
| numeral | 1576 | 100% | 0 | 0 |
| unit_conversion | 1594 | 99.1% | 10 | 5 |
| bit_manipulation | 1602 | 69.7% | 344 | 142 |
| cipher | 1576 | 39.7% | 951 | 0 |
| transformation | 1555 | 9.1% | 489 | 924 |

## Cambios del notebook vs v3-curated
1. **Normalización bulletproof** de ambas fuentes → exactamente `<think>\n{body}\n</think>\n\boxed{answer}`
   (strip de think-tags existentes + strip del boxed final; teacher viene sin <think> y con \boxed propio).
2. **Completion masking**: loss SOLO sobre el turno assistant (DataCollatorForCompletionOnlyLM con
   response template `<|im_start|>assistant\n`; fallback inline si la versión de trl no lo trae).
   Mask-check assert en el primer batch (falla rápido si no funciona).
3. Dataset → real_corpus_v1.csv. Resto idéntico (rank 32/alpha 32, lr 2e-4, 1 epoch, eff batch 32,
   stratified por type, save_steps 50, Blackwell).

## Plan post-run
1. `kaggle kernels status gastonz195/train-real-v1` → COMPLETE (~7h).
2. Submit: `uv run kaggle competitions submit nvidia-nemotron-model-reasoning-challenge -k gastonz195/train-real-v1 -v 1 -f submission.zip -m "real-v1: solvers verificados sobre prompts reales + teacher fallback + completion masking"`
3. Run B en paralelo: extender solver transformation (ops extendidas + CSP símbolo→dígito) → corpus v2.

## Resultado (a completar)
- [ ] Kernel status: ____
- [ ] mask-check en log: ____
- [ ] LB público: ____
