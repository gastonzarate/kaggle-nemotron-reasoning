# train-structured-v3-curated

**Pusheado**: 2026-06-08 (`kaggle kernels push`)
**Kernel**: https://www.kaggle.com/code/gastonz195/train-structured-v3-curated (v1)
**Objetivo**: re-hacer "v3 pt1" (LoRA fresco, 1 epoch) pero sobre el **dataset curado** y con el bug de double-`</think>` arreglado.

## Origen y diferencias vs train-structured-v3p1

Copia del notebook `train-structured-v3p1` (train-from-scratch, 1 epoch), con UN cambio en la celda de armado del target SFT:

| | v3p1 (previo) | v3-curated (este) |
| --- | --- | --- |
| Dataset | `nemotron-structured-cot-v3` v1 (9 familias, buggy) | **`nemotron-structured-cot-v3` v2 (curado: 6 familias ×1500)** |
| Target SFT | `cot_cleaned + "\n</think>\n\boxed{ans}"` → **DOBLE `</think>`** | `body + "\n\boxed{ans}"` → **un solo `</think>`** |
| Resto | — | idéntico |

### El bug arreglado (double-`</think>`)
Nuestros `generated_cot` ya son `<think>...</think>\boxed{X}`. El notebook v3p1/v3p2 quitaba el `\boxed` (conservando el `</think>` interno) y volvía a anexar `\n</think>\n\boxed{}`, produciendo **dos `</think>`** en el target. El chat template de Nemotron usa el content del assistant final **verbatim** (no inyecta `<think>` de apertura), así que el target quedaba malformado. Confirmado contra `chat_template.jinja` (línea 162). Es el mismo bug que el QA de v2 marcó como catastrófico — afectó a TODOS los rows de los submits v3 previos.

## Hyperparams (heredados de v3p1, sin cambios)

- Base: `metric/nemotron-3-nano-30b-a3b-bf16`, Unsloth, 16-bit LoRA.
- LoRA: rank 32, alpha 32, dropout 0; target `q/k/v/o_proj, in_proj/out_proj, up/down_proj`.
- 1 epoch, micro-batch 1 × grad-accum 32 (eff 32), lr 2e-4, linear, max_seq 8192.
- `save_strategy="no"` (heredado; el run de 1 epoch sobre 9000 completó en ~6-7h en v3p1).
- Stratified batching por `type` (6 familias).
- `machine_shape: NvidiaRtxPro6000`.

## Dataset curado (v3.1)

- 6 familias × 1500 = 9000. transformation = aritmética con operador oculto (reverse-engineered de train.csv).
- Verificado: boxed==answer exacto en 9000; 0 incoherencias; 0 ambigüedad en transformation y bit_manipulation; auditado por subagentes (20% + barridos completos de transformation y bit).
- Ver `data/analysis/qa_v3_corrected.md` y `data/analysis/transformation_taxonomy.md`.

## Plan post-run (PENDIENTE)

1. Esperar a que el kernel COMPLETE (~6-7h). Verificar status: `kaggle kernels status gastonz195/train-structured-v3-curated`.
2. (Opcional) bajar output: `kaggle kernels output gastonz195/train-structured-v3-curated -p outputs/train-structured-v3-curated`.
3. **Submit**:
   ```bash
   uv run kaggle competitions submit nvidia-nemotron-model-reasoning-challenge \
     -k gastonz195/train-structured-v3-curated -v 1 -f submission.zip \
     -m "v3-curated: LoRA fresco 1ep sobre dataset curado (6 familias, transformation aritmética) + fix double-</think>"
   ```
4. Anotar LB público acá y en `CHANGELOG.md`.

## Resultado (a completar)

- [ ] Kernel status: _________
- [ ] Training time: _________
- [ ] submission.zip generado: _________
- [ ] Submit enviado: _________
- [ ] LB público: _________
