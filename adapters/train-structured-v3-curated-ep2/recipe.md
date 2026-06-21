# train-structured-v3-curated-ep2 (epoch 2, continual)

**Pusheado**: 2026-06-09 (v2 con el adapter source válido)
**Kernel**: https://www.kaggle.com/code/gastonz195/train-structured-v3-curated-ep2

## Qué es
Continual training (1 epoch más) ARRANCANDO del adapter de `train-structured-v3-curated` (epoch 1). Equivale a "v3p2" pero con los dos fixes aplicados.

## Diferencias vs v3p2 (el continual previo, buggy)
| | v3p2 (previo) | ep2 (este) |
| --- | --- | --- |
| Adapter base | `gastonz195/v3p1-adapter` (entrenado con datos buggy + double-`</think>`) | **`gastonz195/v3-curated-adapter`** (epoch 1 sobre datos curados, single-`</think>`) |
| Dataset | v3 buggy | v3.1 curado (6 familias) |
| Target SFT | double-`</think>` | **un solo `</think>`** (arreglado) |

## Hyperparams (heredados de v3p2)
- Continual: carga el LoRA de epoch 1 vía Unsloth `from_pretrained` (auto-attach), NO `get_peft_model`.
- 1 epoch, micro-batch 1 × accum 32, **lr 1e-4** (más bajo para continual), max_seq 8192, save_strategy="steps" save_steps=50.
- Stratified batching por `type` (6 familias).

## Dependencias Kaggle
- `gastonz195/v3-curated-adapter` (dataset, 3GB) — el adapter de epoch 1, subido con `--dir-mode zip` → se monta extraído en `sft_adapter/`.
- `gastonz195/nemotron-structured-cot-v3` (dataset curado v3.1).

## Plan post-run (PENDIENTE)
1. Esperar COMPLETE (~6-7h). `kaggle kernels status gastonz195/train-structured-v3-curated-ep2`.
2. Submit:
   ```bash
   uv run kaggle competitions submit nvidia-nemotron-model-reasoning-challenge \
     -k gastonz195/train-structured-v3-curated-ep2 -v 2 -f submission.zip \
     -m "v3-curated-ep2: 2 epochs (continual) sobre datos curados + fix double-</think>"
   ```
3. Comparar LB vs epoch 1 para decidir si 2 epochs ayuda.

## Resultado (a completar)
- [ ] Kernel status: _______
- [ ] LB público: _______
