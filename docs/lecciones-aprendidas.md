# Lecciones aprendidas — NVIDIA Nemotron Model Reasoning Challenge

Informe completo de la competencia: qué hicimos, qué funcionó, qué no, qué hicieron los ganadores, y los principios transferibles a otras competencias de Kaggle / fine-tuning de LLMs.

- **Resultado final**: private LB **0.832**, puesto **~2368 / 4183** (top 57%). Sin medalla (bronce requería 0.864).
- **Mejor adapter**: la receta pública pura (SFT teacher). Ninguna de nuestras mejoras de datos la superó.
- **Una frase**: en competencias de exact-match con plateau denso, *iterar rápido sobre lo que funciona* le gana a *inventar algo mejor sin poder medirlo*.

---

## 1. El problema en una pantalla

- **Tarea**: entrenar un adapter LoRA (rank ≤ 32) sobre `Nemotron-3-Nano-30B` (híbrido Mamba-Transformer MoE) para resolver acertijos de razonamiento inductivo de 6 familias (bit manipulation, gravedad `d=0.5·g·t²`, conversión lineal de unidades, cifrado por sustitución, sistemas numerales, transformaciones aritméticas con operador oculto).
- **Evaluación**: server-side con vLLM, `temperature=0`, `max_tokens=7680`, extracción de la respuesta dentro de `\boxed{}`, exact-match con normalización liviana. Métrica float con tolerancia relativa 1% en las familias numéricas.
- **Lo que se sube**: solo el adapter. Todo lo de "antes" (framework, batch, epochs, dónde entrenás) es libre.

## 2. Cronología de experimentos (public LB)

| # | Enfoque | LB | Δ |
|---|---|---|---|
| 1 | baseline: fork receta pública (Unsloth, teacher cot-tong) | 0.83 | — |
| 2 | structured CoT (tags `<r>/<m>/<c>/<l>`) + fix precisión | 0.84 | +0.01 |
| 3 | dataset 100% sintético (puzzles generados) | 0.54-0.56 | **-0.28** |
| 4 | prompts reales de train.csv + CoT de solver | 0.64 | -0.20 |
| 5 | + técnicas de auditoría (CSP, commit-verify, NEFTune, upsample) | 0.65 | -0.19 |
| 6 | anchor: receta 0.84 + 225 filas gap-fill verificadas | 0.81 | varianza |
| 7 | re-seeds de la receta base | 0.81-0.82 | varianza σ≈0.015 |
| 8 | winner-config: + lm_head en targets + batch 64 | 0.77 | -0.07 |
| 9 | corpus combinado (teacher + reasoning verificado en familias difíciles) | 0.67 | -0.17 |
| 10 | combinado + lm_head/batch64 | 0.59 | -0.25 |
| 11 | alpha-sweep post-hoc (α 28/32/36) | 0.81/0.84/0.84 | meseta |

**Privado**: mejor 0.832. Dato curioso: el corpus combinado subió en privado (0.676 → 0.720), generalizó algo mejor que en público pero seguía lejos del teacher.

## 3. Qué hicieron los ganadores

Top privado **0.912** (NullSira). Corte de oro ~0.876. Fuentes públicas tras el cierre:
- `taidoduc/nemotrone-writeup-taido` — medalla de oro (#14/4182, 0.87), "Best Data/Synthetic".
- `liauys/nemotron-reasoning-3rd-place-lora-submission` — 3er puesto (0.892).
- `tonghuikang/nemotron` — Progress Prize (Tinker + token weighting).
- `junkoda/a-bit-manipulation-algorithm` — solver de bit-manipulation.

La receta del ganador de oro (taido):
1. **Trace engineering**: las trazas enseñan una *metodología*, no la respuesta. Cada traza (a) identifica el subtipo, (b) declara el método, (c) lo ejecuta paso a paso sin saltarse nada, (d) incluye **fallback / auto-corrección** (ej. cipher con doble confirmación PASS/FAIL: si discrepa, re-lee y corrige). Eso llevó cipher a 100% local.
2. **Aritmética con sumas parciales** mostradas (`a+b=p; p+c=total`). El modelo se equivoca si no se muestra el cálculo, incluso en sumas simples.
3. **Estimar el factor desde TODOS los ejemplos** (`sum(out)/sum(in)`), robusto al ruido de redondeo (nosotros usábamos 2 puntos).
4. **2 epochs**: epoch 1 sobre 19.404 filas, epoch 2 con LR menor sobre un subset + sintéticos de la familia numérica.
5. **Selective token weighting** (`decision-weight 2`): los tokens de decisión difíciles pesan el doble en el loss.
6. **Gradient accumulation balanceado por categoría**.
7. **Eval LOCAL** (techo ~0.89) para iterar la calidad de las trazas rápido. Entrenó solo 7 modelos (mismo compute que nosotros) pero cada uno bueno.

## 4. Principios transferibles a otras competencias

### 4.1 Explotar antes que explorar
En un plateau denso, la receta probada es una variable aleatoria (acá σ≈0.015 entre seeds). 1478 equipos empataron en 0.86 simplemente corriéndola N veces y guardando el máximo. **Antes de invertir compute en ideas nuevas, gastá una parte fija en farmear la varianza de lo que ya funciona.** El máximo de 10 tiros sesgados al alza suele superar a 1 tiro de una idea brillante no probada.

### 4.2 El eval local rápido es la inversión #1
El verdadero diferenciador de los ganadores no fue una idea mágica: fue **poder iterar**. Ellos refinaban las trazas viendo qué fallaba en un holdout local que correlacionaba con el LB. Nosotros tirábamos runs de 7h a ciegas. **Construí el harness de evaluación ANTES que cualquier experimento de modelado.** Sin él, cada idea cuesta horas y no sabés por qué falló.

### 4.3 En distilación de razonamiento, el ESTILO pesa más que el contenido
Trazas verificadas al 100% (de un solver) rindieron peor que las del teacher, porque están fuera de la distribución nativa del modelo. El LoRA (poco rank) se "deforma" imitando un estilo ajeno en vez de afinar el razonamiento. La evidencia (papers de distilación) lo confirma: barajar el orden de los pasos hace caer la accuracy mucho más que meter números mal. **Para SFT de razonamiento: trazas en la distribución del modelo (self-distillation del propio base, filtrado por gold) > trazas "correctas" pero ajenas.**

### 4.4 No copiar técnicas del ganador en aislamiento
`lm_head` en los targets y `batch 64` nos restaron 0.07-0.25 sueltos, pero al ganador le servían **porque iban con** su token weighting y su pipeline (Tinker). Las ganancias suelen ser de la *combinación*, no de la pieza. Replicá el sistema completo o nada.

### 4.5 Levantá la restricción de compute si es barata
Toda la competencia estuvimos limitados por la cuota gratis de 30h/semana de Kaggle. Una H100 a ~$2/h por unas horas (=$30-60) habría dado 10-20× más experimentos: alcanzaba para el eval local, los re-seeds y el self-distillation. **Identificá temprano cuál es tu cuello de botella y cuánto cuesta sacarlo.**

### 4.6 Diagnosticá antes de arreglar
El derrumbe a 0.54 tenía dos causas concretas que encontramos con análisis, no con intuición: prompts sintéticos fuera de distribución + un bug que metía dos `</think>` en cada target (formato roto). Sin el diagnóstico habríamos seguido "mejorando datos" sobre una base rota.

### 4.7 Disciplina de verificación de datos
Un dataset puede importar limpio en pandas y estar corrupto (boxed ≠ answer, ambigüedades, doble tag). Cada cambio de datos se gatea contra la métrica oficial replicada localmente, fila por fila, antes de gastar GPU.

### 4.8 Diversificá las 2 submissions finales
Con un plateau enorme, el leaderboard privado reordena fuerte. Elegí 2 finales **diversas** (modelos distintos, no el mismo adapter reescalado) como hedge. Nosotros elegimos 0.84 + un re-seed 0.82 distinto.

## 5. Playbook reutilizable (checklist)

1. Leé el mecanismo de evaluación EXACTO (cómo se extrae y compara la respuesta) antes de modelar.
2. Replicá la métrica oficial localmente. Es tu fuente de verdad.
3. Construí un holdout local que correlacione con el LB público (gate < 0.02 de delta).
4. Reproducí la mejor receta pública. Ese es tu piso y tu baseline de ablación.
5. Mide el costo de tu cuello de botella (compute, datos, tiempo) y sacalo si es barato.
6. Separá presupuesto: X% a farmear varianza de lo que funciona, (1-X)% a explorar.
7. Cada idea nueva: una variable a la vez, gateada por el eval local antes de tocar GPU.
8. Para SFT de razonamiento: priorizá trazas en-distribución (self-distillation + filtro por gold) con metodología y auto-corrección.
9. Documentá cada run con su receta exacta (datos, hiperparámetros, seed, score). Si no hay recipe, el run no existió.
10. Finales: 2 submissions diversas, una segura y una con upside.

## 6. Plan para la próxima edición (o el revancha)

1. **Día 1**: harness de eval local + reproducir la receta 0.85. Nada de modelar hasta tener el eval.
2. **Self-distillation**: correr el propio Nemotron base sobre los prompts reales a temp~1.0, quedarse con las trazas correctas (gold-conditioned), entrenar sobre esas. Estilo en-distribución + correcto.
3. **Trace engineering** estilo taido: metodología + ejecución paso a paso + fallback/auto-corrección, sobre todo en cipher y bit.
4. **Token weighting** (decisión ×2) + 2 epochs (epoch 2, LR bajo, subset difícil).
5. **Farmear varianza**: 8-10 re-seeds de la mejor receta, guardar el máximo.
6. **H100 alquilada** para que (1)-(5) entren sin pelear la cuota.
7. Reusar nuestro **solver gold-conditioned de transformación** (CSP símbolo→dígito, ~97%) como oráculo para etiquetar/validar, no como estilo de traza.

## 7. Artefactos reutilizables de este repo

- `src/data/transformation_extended.py` — solver con CSP símbolo→dígito (oráculo ~97%).
- `src/data/programmatic_cot.py`, `puzzle_generators.py` — generadores por familia.
- `src/eval/` — verificadores y parser de la métrica oficial.
- `data/analysis/` — auditorías técnicas (cot_audit, qa, transformation_taxonomy, lora_audit_v4).
- `notebooks/my_kernels/` — kernels de training.
