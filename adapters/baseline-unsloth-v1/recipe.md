# baseline-unsloth-v1

**Lanzado**: 2026-05-26 ~22:40 -03 (vía `kaggle kernels push`)
**Estado al lanzar**: RUNNING en Kaggle (free tier, GPU asignada por Kaggle por defecto)
**Tiempo estimado**: ~7 horas
**URL del kernel**: https://www.kaggle.com/code/gastonz195/baseline-unsloth-v1

## Origen

Fork directo (sin modificaciones) de:
- `dgxchen/training-with-unsloth-to-achieve-0-85-lb` (versión clonada en `notebooks/community/`)

Receta probada por el autor a **LB público 0.85**. Es nuestra **safety net de Fase A** (ver `PLAN.md`).

## Hyperparams (heredados del notebook fuente, sin modificar)

- Base model: `metric/nemotron-3-nano-30b-a3b-bf16` (Kaggle Model hub)
- Framework: Unsloth (QLoRA backend)
- LoRA: rank 32, alpha 32, dropout 0
- Target modules: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, out_proj (sin lm_head — el autor dgxchen lo removió explícitamente del target_modules)
- Sequence length: 8 192
- Batch: micro-batch 1, gradient accumulation 32 → effective batch 32
- Steps: 1 000
- LR: 2e-4
- Training data: `dgxchen/nemotron-cot-tong` (los 6 558 ejemplos CoT-filtrados de Tong Hui Kang)

## Hipótesis bajo test

Reproducir 0.85 LB con cero cambios. Esto:
1. Valida que nuestro pipeline CLI funciona end-to-end.
2. Nos garantiza una submission con score público ≥0.83 (rango razonable considerando varianza de batching en GPU).
3. Si pasa, marcamos esta submission como **una de las 2 finales** por las dudas.

## Lo que NO esperamos

- Romper el plateau 0.87. Estamos reproduciendo, no innovando.
- Score >0.85 sin haber tocado nada.

## Caveats conocidos antes del run

- **Bug en training data** reportado por Ashutosh (50.5% mismatches en bit-manipulation, 49% en equation_transformation) — el dataset `dgxchen/nemotron-cot-tong` puede tenerlos. No vamos a corregirlos en esta corrida; los corregimos en Fase B.
- **Metric Update del 2026-05-17**: arregló extracción de `\boxed{...}` con `}` adentro. Las submissions nuevas usan el extractor corregido. No nos afecta porque es nuestra primera submission.

## Plan post-run

1. Cuando termine, bajar el output (`submission.zip`) con `kaggle kernels output`.
2. Subir como Kaggle Dataset privado: `gastonz195/baseline-unsloth-v1-adapter`.
3. Fork de `ryanholbrook/nvidia-nemotron-submission-demo` apuntando a ese dataset.
4. Submit via CLI o UI.
5. Anotar el LB score acá una vez disponible.

## Run log

### Attempt 1 (2026-05-26 22:40 → 2026-05-27 ~10:50, **FAILED**)

- **Duración**: 431.6 s (~7 min)
- **Fallo**: `AcceleratorError: CUDA error: no kernel image is available for execution on the device` al ejecutar `FastLanguageModel.from_pretrained(...)`.
- **Causa**: en `kernel-metadata.json` omití `machine_shape`, así Kaggle asignó GPU default (probablemente T4 sm_75). Los wheels de `causal_conv1d` y `mamba_ssm` están pre-compilados para Blackwell (sm_100). Mismatch de arquitectura CUDA.
- **Fix**: agregar `"machine_shape": "NvidiaRtxPro6000"` al metadata.

### Attempt 2 (2026-05-27 10:58 → 2026-05-27 ~22:50, **COMPLETE**)

- Versión 2 pusheada con `machine_shape: NvidiaRtxPro6000`.
- **GPU asignada**: NVIDIA RTX Pro 6000 Blackwell Server Edition, 94.97 GB VRAM.
- **Modo**: 16-bit LoRA (no QLoRA — base bf16 entró cómodo en 95 GB).
- **Trainable params**: 883.87 M sobre 32.46 B total = 2.72 %.
  - Unsloth detectó MoE 128 experts y extendió el LoRA a `mlp.experts.gate_up_proj` + `mlp.experts.down_proj`.
- **Dataset**: 7 830 rows con stratified batching por 9 tipos (más granular que las 6 familias visibles en `train.csv` original).
- **Effective batch**: 32 (micro 1 × grad accum 32).
- **Tiempo de training puro**: 406.5 min (6h 47min).
- **Tiempo total del kernel** (incl. setup + packaging): 25 192 s ≈ 7 h.
- **Output**:
  - `adapter_model.safetensors`: 3.37 GB (bf16, 12 008 keys)
  - `adapter_config.json` con `base_model_name_or_path: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `inference_mode: true`
  - `submission.zip`: 3.09 GB (deflated)
- **Naming convention**: Unsloth produjo keys con `base_model.model.backbone.*` (12 008 / 12 008), ya en formato Nemotron — **NO requirió pasar por el rename del submission demo**.

## Resultado

- [x] Run terminó OK (Attempt 2)
- [x] Output `submission.zip` bajado: `outputs/baseline-unsloth-v1/submission.zip` (3.09 GB)
- [ ] ~~Dataset subido~~ (no necesario — submitimos kernel directamente)
- [x] Submission enviada: 2026-05-28 03:00:12 UTC
- [x] **LB score público: 0.83** (scored 2026-05-28 ~04:20 UTC, ~80 min después del submit)
- [ ] LB score privado (post-deadline): N/A hasta 2026-06-15
- [x] Tiempo total de training: 406.5 min (6h 47min)
- [ ] Git SHA al cierre del experimento: pending (commit final)

### Comando exacto de submission

```bash
uv run kaggle competitions submit nvidia-nemotron-model-reasoning-challenge \
  -k gastonz195/baseline-unsloth-v1 \
  -v 2 \
  -f submission.zip \
  -m "baseline-unsloth-v1: reproducción receta dgxchen (Unsloth, 16-bit LoRA r32, ~7h en Pro 6000)"
```
