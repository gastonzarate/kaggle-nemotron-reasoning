# NVIDIA Nemotron Model Reasoning Challenge — workspace

Mi intento en el [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) de Kaggle. La tarea: entrenar un adapter LoRA (rank ≤ 32) sobre `Nemotron-3-Nano-30B` para resolver acertijos de razonamiento inductivo de 6 familias. La evaluación corre server-side con vLLM, `temperature=0`, y exact-match contra la respuesta dentro de `\boxed{}`.

## Resultado

Puesto ~2368 de 4183 (top 57%), private LB 0.832. Sin medalla (bronce requería 0.864). El mejor adapter terminó siendo la receta pública pura; ninguna de mis mejoras de datos la superó.

| Enfoque | Public LB |
|---|---|
| Receta teacher (SFT sobre cot-tong) | **0.84** |
| Re-seeds del base (varianza σ≈0.015) | 0.81–0.82 |
| Dataset 100% sintético | 0.56 |
| Prompts reales + CoT de solver | 0.64 |
| Corpus combinado (teacher + reasoning verificado) | 0.67 |
| + lm_head en targets / batch 64 | 0.59–0.77 |

## Qué aprendí

1. **El estilo del CoT pesa más que la corrección del contenido.** Trazas de solver verificadas al 100% rinden peor que las del teacher porque están fuera de la distribución nativa del modelo. El LoRA gasta su poco rank imitando un estilo ajeno en vez de afinar el razonamiento.
2. **En un plateau denso, repetir le gana a inventar.** 1790 equipos empataron en 0.86 corriendo la misma receta N veces y quedándose con el máximo. Yo gasté el compute inventando datos.
3. **El cuello de botella fue asignación de recursos, no ingeniería.** Faltó: farmear varianza, medir local rápido, y levantar la cuota gratis de 30h con una GPU alquilada.

Retrospectiva completa en [`data/analysis/lora_audit_v4.md`](data/analysis/lora_audit_v4.md) y el historial en [`CHANGELOG.md`](CHANGELOG.md).

## Lo reutilizable

- `src/data/transformation_extended.py` — solver gold-conditioned con CSP símbolo→dígito para la familia de transformación (resuelve ~97% como oráculo).
- `src/data/programmatic_cot.py`, `puzzle_generators.py` — generadores de puzzles + CoT por familia.
- `src/eval/` — verificadores y parser de la métrica oficial.
- `notebooks/my_kernels/` — los kernels de training (Unsloth + LoRA sobre el híbrido Mamba-Transformer).

## Estructura

```
src/        código de datos y evaluación
tests/      suite (≈650 tests)
notebooks/my_kernels/   kernels de Kaggle propios
data/analysis/          writeups y auditorías (sin datos de la competencia)
adapters/*/recipe.md    receta reproducible de cada run
```

Los datos de la competencia, los corpus derivados y los pesos de los adapters no se versionan (reglas de Kaggle + tamaño).
