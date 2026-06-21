# Changelog — NVIDIA Nemotron Reasoning Challenge

Registro de submissions, cambios aplicados y movimientos en el LB.
**Top público actual**: 0.87 · **Equipos**: 3 526 · **Deadline**: 2026-06-15 23:59 UTC.

---

## v4.0 — PIVOT a corpus REAL (receta del Progress Prize winner) — 2026-06-10

**Investigación:** el ganador del Progress Prize (huikang, LB 0.85) entrena sobre los prompts REALES
de train.csv con reasoning de solvers programáticos por familia + completion masking, y saltea lo que
no resuelve. Ablacionó la aritmética dígito-por-dígito: NO mejora. Fuentes: repo tonghuikang/nemotron,
writeup /discussion/689915, notebook end-to-end-finetuning-for-lb-0-85.

**Corpus real v1** (`src/data/build_real_corpus.py` → `gastonz195/nemotron-real-corpus-v1`, 8 429 rows):
- 6 635 solver-verified (gate: predicted==gold oficial) + 1 794 teacher cot-tong fallback + 1 071 skipped.
- Solve rates: gravity/numeral 100%, unit 99.1%, bit 69.7%, cipher 39.7%, transformation 9.1%.

**Notebook train-real-v1** (3 cambios): normalización bulletproof del CoT (ambas fuentes → un solo
`<think>…</think>\boxed{answer}`), **completion masking** (loss solo en el turno assistant, con
fallback inline + mask-check assert), dataset nuevo. Resto idéntico al baseline probado.

**Run A**: kernel `train-real-v1` v1 pusheado y corriendo (~7h). Expectativa: clase 0.84+.
**Run B (en prep)**: solver extendido de transformation (`transformation_extended.py`: ops ±1,
floordiv/mod, glifo-como-signo, CSP símbolo→dígito) para recuperar parte de las 924 skipped → corpus v2.

### Resultado
- **LB train-real-v1: 0.64** · **train-real-v2 (audit techniques): 0.65**.
- Lectura: prompts reales +0.08-0.09 sobre sintético; técnicas de auditoría +0.01; pero el
  **estilo del reasoning domina**: teacher (probablemente self-generated por el base) = 0.83-0.84
  vs nuestro programático = 0.54-0.65 en cualquier distribución de prompts. Hipótesis: el texto
  programático es OOD para el base; el LoRA r32 se deforma para imitarlo y degrada el razonamiento nativo.
- Transformation extendido (gold-cond + CSP): 9.1% → 38.9% solver-solved (605/1555).

## v4.6 — FASE 2 resultados + pivot continual→from-scratch (2026-06-13)

- **winner-cfg COMPLETE → submitido** (lm_head + batch 64). Score PENDING.
- **stage2-v2 ERROR**: la línea "continual" (v3p2/ep2/stage2 cargando adapter previo) **falló 3 veces**;
  la línea "from-scratch" (anchor, winner-cfg) **siempre completa**. Decisión: abandonar continual.
- **PIVOT**: corpus combinado `nemotron-combined-corpus-v1` (8 777 rows) — teacher en familias
  buenas (3 344: gravity/numeral/unit) + **reemplazo con reasoning VERIFICADO en las difíciles**
  (bit 1 728 jahyee global-expr, transformation 1 429 CSP/jahyee, cipher 1 329) + 947 IDs nuevos.
  Estrategia de REEMPLAZO (no aditiva): los CoT divagantes/incorrectos del teacher en transformation
  se sustituyen por los verificados gold-conditioned.
- **train-combined-winner**: corpus combinado + config ganadora (lm_head+b64) = mejor tiro único.
  RUNNING. **train-reseed2** (SEED 43) RUNNING en paralelo.
- Slug-fantasma recurrente: cada push que choca quota deja el slug inutilizable → recrear con slug nuevo
  (train-stage2-best→v2, train-anchor-reseed→reseed2).

## v4.5 — FASE 2 disparada con quota fresca (2026-06-13 ~00:15 UTC)

- Quota semanal reseteada. Pusheados: **train-stage2-v2** (stage2 desde adapter 0.84, corpus
  jahyee+CSP 7 632 rows, LoRA+ lr 5e-5) y **train-winner-cfg** (receta base + lm_head en targets +
  batch eff 64) — ambos **RUNNING en paralelo**.
- Límite descubierto: **máx 2 sesiones GPU batch simultáneas** → train-anchor-reseed (SEED 43,
  kernel listo) encolado; se pushea al liberarse un slot (stage2 ~3-4h).
- Slug `train-stage2-best` quedó corrupto en Kaggle ("Notebook not found" tras chocar con quota);
  re-creado como `train-stage2-v2`. Patrón: un push que falla por quota puede dejar el slug inutilizable.
- Al completar cada kernel → submit inmediato. Monitor orquestador armado.

## v4.4 — scores: anchor 0.81, alpha-sweep mapeado (2026-06-11)

- **anchor-v3: LB 0.81** (vs 0.83/0.84 históricos de la misma receta) → la lotería de la receta
  base tiene σ~0.015 con muestras {0.81, 0.83, 0.84}. Refuerza estrategia max-of-N.
  Consecuencia: stage2 y alpha-sweep re-apuntados al adapter del 0.84 (`v1-best-adapter`).
- **Alpha-sweep post-hoc del 0.84** (kernel CPU `package-alpha`, 0 GPU): α28→0.81, α32→0.84,
  α36→0.84. Meseta 32-36, caída a 28 → el scale no es palanca. Vía cerrada gratis.
- En cola sábado (quota reset): stage2 (jahyee+CSP desde 0.84) · winner-cfg (lm_head+b64) · re-seed.

## v4.3 — anchor COMPLETE + intel jahyee + quota gate (2026-06-10/11)

- **anchor-v3 COMPLETE y submitido** (score pendiente). Adapter bajado y subido como
  `gastonz195/anchor-v3-adapter` (READY).
- **Deep research (103 agentes)**: VERIFICADO que el winner usa Tinker + batch 64 + **train_unembed
  (lm_head)** + per-token loss weighting. Kernel `train-winner-cfg` (lm_head + accum 64) listo p/ sábado.
- **Corpus oracle jahyee** (kaggle: jahyee/nemotron-oracle-reasoning-traces, 2026-06-02): 9 056 traces,
  gateados 9 056/9 056 contra gold real. bit con expresiones GLOBALES (shifts/rotations) 1 440;
  cipher 100% (incl. letras no vistas). **stage2 v3 = 7 632 rows** (bit 1 440 jahyee + transformation
  1 425 ours∪jahyee + cipher 951 jahyee + replay 3 816).
- **Quota GPU semanal agotada (30h)** → stage2 push bloqueado hasta el reset del sábado 00:00 UTC.
  Mientras: alpha-sweep (kernel CPU `package-alpha`, sin quota) sobre el anchor apenas scoree.
- Sábado: stage2 (~3h) + winner-cfg (~7h) + epoch-2/re-seed (~14h) ≈ 24h ≤ 30h nuevas. Deadline lunes.

## v4.2 — Plan "Operación 0.87" + moonshot lkevincc (2026-06-10)

**Leaderboard real** (4 151 equipos): 0.89×1, 0.88×3, 0.87×29, **0.86×1 478 (mega-empate)**, 0.85×509.
Top 10% = puesto ≤415 → con desempate por fecha, en la práctica exige **0.87** (o un 0.86 fuerte que
suba en el reshuffle privado). Diff de la receta pública actual vs nuestro fork: IDÉNTICA → el plateau
es max-of-N de la misma lotería (σ≈0.015), no una receta mejor.

**Moonshot integrado**: solver público de lkevincc (97.2% gold-conditioned, foro /698293, repo
lkevincc0/kaggle-nemotron-equation-symbolic) — su `solver_results.parquet` trae mapa símbolo→dígito +
op por glifo + modo para 800/823 puzzles simbólicos. Generamos reasoning desde sus reglas para
**740 filas nuevas, 100% verificadas contra gold** → cobertura transformation **38.9% → 86.5%** (1 345/1 555).
Dataset stage2 v2: 2 918 rows (1 459 hard + 1 459 replay teacher). Op library de referencia mucho más
rica (gcd/lcm/min/max/xor/±2/mul_half/squared_diff) — anotada para futuras extensiones propias.

**Plan de cierre (seguro → arriesgado)**: F0 gratis (alpha-sweep post-hoc del adapter_config — el
scale α/r es editable sin re-entrenar; LoRA-soup CPU de adapters compatibles) · F1 stage2 (hard+replay,
LoRA+) · F2 sábado quota fresca (epoch-2 continual + anchor re-seed paralelo) · F3 domingo (alpha-sweep
y soup sobre el mejor; tiro arriesgado: soup ponderado con real-v2) · F4 lunes selección 2 finales
(mejor público + diversificada). Corrección de auditoría: vLLM SÍ soporta use_rslora/use_dora
(peft_helper docs); se mantienen fuera por riesgo de re-tuneo de LR (paper 2602.04998: variantes ≈
iguales con LR tuneado). LoRA+ (train-only, artefacto vainilla) integrado en stage2.

## v4.1 — Run C "anchor" (2026-06-10)
**Estrategia conservadora**: receta EXACTA del 0.84 (notebook train-structured-v1, sin masking)
con su dataset teacher + **gap-fill quirúrgico**: solo 225 filas solver para ids que el teacher
no cubre, SOLO familias difíciles (bit 114, transformation 111) = 2.8% del dataset. Piso ≈0.84,
upside en cobertura. Dataset `nemotron-anchor-corpus-v3` (8 055 rows). Kernel `train-anchor-v3` corriendo.
- [ ] LB train-anchor-v3: ____

---

## v3.3 — CoTs enriquecidos "show the work" (2026-06-09, sin entrenar aún)

**Diagnóstico del 0.56 (comparación v1 0.84 vs v3):** el formato NO era el problema — el v1 (0.84) tenía formato MÁS roto (doble `</think>`, 0 `\boxed`). La diferencia real es la **profundidad del razonamiento**: el teacher v1 muestra cada long-division y multiplicación dígito por dígito (~5457 chars); nuestro v3 **afirmaba** los resultados aritméticos (`g = 127.88/24.4 = 5.24`, ~650 chars) sin derivarlos. El modelo aprendió a afirmar, no a calcular.

**Fix (este):** enriquecí los CoTs de las 6 familias para MOSTRAR toda la aritmética/lógica paso a paso usando `math_primitives` (`ColumnMultiply`/`LongDivision`/`ColumnSubtract` con walkthrough), + framing conceptual + paso de verificación (que el teacher NO tenía). Longitudes nuevas: gravity 4068, unit_conv 2128, cipher 1741, bit 1453, transformation 993, numeral 955 (media 1890 vs ~650). Estructura limpia, 0 incoherencias, 0 ambigüedad, 644 tests verdes.

### Pendiente
- [ ] Re-subir dataset enriquecido + entrenar (espera slot GPU; ep2 terso todavía corriendo).
- [ ] LB con CoTs enriquecidos: _______ (test de la hipótesis "profundidad de razonamiento").

---

## v3.2 — submit con dataset curado + fix double-</think> (2026-06-09)

**Contexto que cierra el diagnóstico:** la submission previa `train-structured-v3p1` (datos v3 buggy) scoreó **LB 0.54** — ESE fue el drop (0.84 → 0.54), no un cambio menor. Causas combinadas: (a) distribución de transformation 0%-match (44% del dataset), (b) **target SFT malformado con doble `</think>`** en TODOS los rows.

**El bug double-`</think>` (nuevo hallazgo, 2026-06-09):** los notebooks v3p1/v3p2 hacían `cot_cleaned` (que conserva el `</think>` interno de nuestros CoTs) y le anexaban otro `\n</think>\n\boxed{}` → dos `</think>`. Confirmado contra `chat_template.jinja` (el template usa el content del assistant final verbatim). Es el bug catastrófico que el QA de v2 ya había marcado. **Arreglado** en el kernel nuevo: strip solo del boxed final (un solo `</think>`).

**Submit enviado:** kernel `gastonz195/train-structured-v3-curated` v1 (LoRA fresco, 1 epoch, lr 2e-4, dataset curado v3.1 + fix double-`</think>`). Estado: **PENDING** (2026-06-09 ~03:47 UTC).
**Epoch 2 (continual):** kernel `train-structured-v3-curated-ep2` (continual sobre el adapter de v3-curated, 1 epoch más, lr 1e-4, mismos fixes) — en preparación/push.

### Resultado
- **LB v3-curated (1 epoch): 0.56** (vs v3p1 buggy 0.54). El fix de datos + double-`</think>` dio solo **+0.02**. Verificado en el log que entrenó con el dataset curado correcto (6 familias ×1500) → NO fue error mecánico.
- [ ] LB v3-curated-ep2 (2 epochs): _______ (corriendo)

### Conclusión estratégica (importante)
El gap real NO es limpieza de datos. Entre v0.2.0 (0.84) y v3-curated (0.56) cambiaron **DOS variables a la vez**, sin poder aislarlas todavía:
1. **Prompts**: reales (cot-tong, sobre `train.csv`) → sintéticos (puzzles que generamos). Nuestra distribución sintética es angosta (solo digit-operand en transformation, solo per-bit en bit, vocab limitado en cipher).
2. **Estilo del CoT**: teacher largo/rico (media **5457 chars**) → programmatic corto/plantillado (media **~650 chars**, ~8× más corto). El base es un modelo de razonamiento; un CoT terso y repetitivo probablemente le enseñó a "pensar poco" → razona superficial en inferencia. **Sospecha principal: este factor pesa más que los prompts.**

**Experimento para aislar:** correr nuestros solvers verificados sobre los prompts REALES de `train.csv` (mantiene prompts reales, CoT terso nuestro). Si sube a ~0.84 → el problema eran los prompts. Si queda en ~0.56 → el problema es el CoT corto/plantillado, y hay que generar razonamiento RICO (teacher o CoT mucho más verboso) sobre prompts reales.
**Base segura conocida:** cot-tong = 0.84 (prompts reales + CoT teacher largo).

---

## v3.1 — dataset corregido + rebalanceado (2026-06-08, sin entrenar aún)

Investigación del drop del v3 (9 000 sintéticos). Dos defectos, dos fixes:

### Defecto 1 — coherencia/correctness (ARREGLADO)
- `build_synthetic_dataset.py::_match` usaba `float()` + tol 1% para TODAS las familias → (a) parseaba binarios como decimales y dejaba pasar **12 reglas erróneas** en bit_manipulation; (b) toleraba CoTs numéricos redondeados que diferían del gold → **1 376 filas incoherentes** (gravity 87.6%, unit_conv 48.8%) donde el texto concluía un valor y el target re-boxeaba otro.
- **Fix:** `_solves_puzzle` (exact-match salvo gravity/unit), emitir `answer = valor que el CoT deriva`, y assert `boxed==answer` byte-exacto. Verificador independiente nuevo: `src/eval/verify_synthetic.py`.

### Defecto 2 — distribución (causa principal del drop, ARREGLADO)
- Las 4 familias "transformation" sintéticas (deletion/substitution/position/affix) eran **44% del dataset** y matcheaban **~0%** de los prompts reales de transformation.
- Caracterización (8 subagentes, 200 prompts reales, reglas verificadas contra gold): la familia real es **aritmética con operador oculto** (`AA op BB`, 5 chars; ops add/sub/absdiff/mult/concat/reverse_concat; modos fwd y little-endian "RR"). Ver `data/analysis/transformation_taxonomy.md`.
- **Fix:** nuevo `gen_transformation_arithmetic` + `arithmetic_transformation_cot`; build rebalanceado a **6 familias × 1 500 = 9 000** (~1/6 c/u, igual que el real).

### Estado del dataset v3.1
- Verificador exhaustivo: **0 incoherencias**, **0 errores en transformation**, 3/9000 gravity/unit marginalmente >1% (redondeo, despreciable).
- **Auditoría manual del 20% (1 800 filas, 12 subagentes con solvers independientes):** 0 errores de estructura/coherencia/respuesta. Surgió una clase de ambigüedad real en transformation (sub vs abs_diff cuando todos los ejemplos tienen a≥b; fwd vs RR con ejemplos simétricos) → ~3.7% de transformation. **Arreglado**: el generador exige respuesta ÚNICA de la query entre todas las reglas consistentes.
- **Barrido completo de las 1 500 transformation (12 subagentes):** tras el fix anterior quedaba una ambigüedad residual `sub-RR` vs `absdiff-RR` (17/1500) que el gate no veía porque `sub-RR` no estaba en su espacio de candidatos. **Arreglado**: se agregó `("sub","RR")` al espacio de candidatos del gate (como hipótesis-fantasma; no se genera como regla real) → completa el espacio {add,sub,absdiff,mult}×{fwd,RR}+concat+rconcat. Ambigüedad transformation: **17 → 0/1500** (verificado independientemente).
- **Barrido completo de las 1 500 bit_manipulation (12 subagentes):** 26/1500 (~1.7%) con ambigüedad por-bit (un bit de salida con ≥2 compuertas consistentes que difieren en la query → señal ruidosa para SFT). 0 estructura/coherencia/wrong. **Arreglado**: gate de unicidad por-bit en el generador (`bit_query_unique`) que rechaza puzzles sub-determinados en cualquier bit de la query. Ambigüedad bit: **26 → 0/1500** (verificado independientemente sobre el espacio completo copy/NOT + AND/OR/XOR/NAND/NOR/XNOR).
- Suite: **645 tests verdes**. Backups del buggy en `*.BUGGY.csv`.
- **Alcance v1:** transformation cubre operandos-dígito; los reales con operandos-símbolo (cifrado símbolo→dígito) y "operador-como-prefijo" quedan como siguiente incremento.
- **Pendiente:** entrenar LoRA con v3.1 y medir LB (no se entrenó todavía).

---

## v0.2.0 — `train-structured-v1` v2

**Submitted**: 2026-05-29 13:36 UTC | **Scored**: ~14:45 UTC
**Public LB**: **0.84** (Δ +0.01 vs v0.1.0) · **Private LB**: pending
**Kernel**: `gastonz195/train-structured-v1` v2 · **Recipe**: `adapters/baseline-unsloth-v1/recipe.md` (con cambios abajo)

### Qué cambió respecto a v0.1.0

| Cambio | Justificación |
| --- | --- |
| **Structured CoT con sub-tags** `<r>` (router) + `<m>` / `<c>` / `<l>` (mode tags) | Hipótesis: forzar estructura de razonamiento mejora el SFT signal y la consistencia. Cada family tiene un tag plan determinístico (bit_manipulation→c, cipher→l, resto→m). |
| **Fix de precisión 2dp half-up** en gravity + unit_conversion | El agente auditor reportó que ~27% del dataset CoT-tong (2 120 rows) tenía mismatches `Result: 3dp` vs `\boxed{2dp}`. Aplicamos rounding half-up dentro del CoT body durante el rewriter heurístico. |
| **Heuristic rewriter** (`src/data/rewriter.py`) | Wrappea el `generated_cot` existente con la nueva estructura tag-based. **Sin teacher LLM** — solo regex + heurísticas. 0 drops sobre 7 830 ejemplos. |
| Hyperparams idénticos al baseline | Rank 32, alpha 32, dropout 0, lr 2e-4, batch eff. 32, 1 epoch. Solo cambia la data. Mantener fijo para aislar la señal. |

### Métricas del run

- Training: **6h 50min** en RTX Pro 6000 Blackwell (95 GB VRAM, 16-bit LoRA, batch 1 × accum 32)
- Adapter size: 3.37 GB (fp32) — sin cambio
- Trainable params: 883.87 M (2.72%) — sin cambio
- Dataset usado: `gastonz195/nemotron-structured-cot-v1` (7 830 rows, derivado de `dgxchen/nemotron-cot-tong`)

### Conclusiones honestas

1. **El +0.01 es real pero modesto**. Está al borde de la varianza esperada por batching GPU (~0.005-0.010 entre runs idénticos). Necesitamos otra señal independiente para confirmar.
2. **No podemos atribuir la mejora a un cambio específico** todavía — la estructura y el fix de precisión se aplicaron juntos. Próxima iteración debería ablacionar uno a la vez.
3. **El plateau público 0.86-0.87 sigue lejos** (3 puntos). El approach actual no rompe el techo per se — solo mejora consistencia.
4. **El eval local NO funcionó** todavía (errores en ambos kernels, pendiente de fixear). Sin per-family breakdown estamos parcialmente a ciegas sobre DÓNDE mejoró.

### Lo que aprendimos en el proceso (no estrictamente del score)

- **Pipeline CLI end-to-end funciona**: push training kernel → adapter como output → submit a competition. Sin tocar UI.
- **El dataset CoT-tong NO tiene los bugs de Ashutosh** que vimos en el foro (esos ya estaban corregidos). En su lugar tiene otros bugs (precisión, LaTeX escape, leading zeros) — documentados en `data/analysis/cot_audit.md`.
- **Adapter format**: nuestro LoRA salido de Unsloth ya viene con keys `backbone` (Mamba naming). No requiere pasar por el "rename" del submission demo.
- **Convención de paths Kaggle**:
  - Competition data: `/kaggle/input/<competition-slug>/`
  - User datasets: `/kaggle/input/datasets/<owner>/<slug>/` (con prefijo `datasets/`)
  - User kernel outputs: `/kaggle/usr/lib/notebooks/<owner>/<slug>/`
- **vLLM custom para esta comp**: hay que hacer `uv pip uninstall torch; tar metric/nvidia_metric_utility_script → /tmp/; sys.path.insert(0,'/tmp')` ANTES de importar vllm. Olvidarlo = `undefined symbol` import error.

---

## v0.2.x — Diagnóstico per-familia (eval limpio sobre 3329 holdout, 2026-05-29)

Después de submitir v0.2.0 al LB (0.84), corrimos eval local sobre el **holdout limpio** (3 329 prompts de train.csv NO incluidos en el dataset CoT-tong, así que el modelo NUNCA los vio durante training). Resultados:

| Family | Baseline v0.1.0 | Structured v0.2.0 | Δ | n holdout |
|---|---:|---:|---:|---:|
| numeral | **1.000** | **1.000** | 0 | 926 |
| unit_conversion | 1.000 | 0.980 | **-0.020** ⚠️ | 604 |
| gravity | 0.979 | 0.989 | +0.010 ✅ | 622 |
| bit_manipulation | 0.577 | 0.565 | -0.012 | 248 |
| **transformation** | **0.004** | **0.003** | ~0 | 929 |
| cipher | (ausente — teacher resolvió todos) | (ausente) | — | 0 |
| **Overall** | **0.687** | **0.684** | **-0.003** | 3 329 |

### Hallazgos no obvios

1. **El +0.01 LB de v0.2.0 NO se refleja en el holdout limpio**.
   - Posibles explicaciones: (a) cipher (ausente de holdout-clean) mejoró en el privado; (b) varianza de batching GPU; (c) el holdout-clean tiene distribución sesgada (solo los puzzles "difíciles" que el teacher no resolvió).

2. **El precision-fix funcionó en gravity** (979 → 989, +1.0 punto) pero **regresó en unit_conversion** (1.000 → 0.980, -2.0 puntos).
   - Diagnóstico: en unit_conversion, el redondeo a 2dp introduce ruido al fittear `Y = a·X + b` con puntos ya redondeados. El modelo deriva un `a` ligeramente distinto del verdadero.
   - Ej: prompt con factor ~0.5147, modelo predice 22.27 vs gold 21.95 (factor ~0.5223 — usa el primer par redondeado en lugar de un fit robusto).

3. **Diff comparativo (donde v0.2.0 ganó/perdió vs v0.1.0)**:
   - **Fixed por v0.2.0** (21 casos): 12 gravity + 9 bit_manipulation
   - **Broken por v0.2.0** (31 casos): 12 unit_conversion + 12 bit_manipulation + 6 gravity + 1 transformation
   - **Neto: -10** en el holdout limpio

4. **Bit_manipulation tiene ceiling alrededor de 0.57** en el holdout difícil. El razonamiento heurístico del teacher no transfiere bien a los "hard" inputs.

5. **Transformation es el gran agujero**: <0.4% accuracy en AMBOS modelos. Mirando las wrong predictions:
   - `34/44 = 1, 41/32 = 9` ← operaciones aritméticas encriptadas (/ resta, | concatena, \ multiplica)
   - `!*[{ = '"[`` ← reglas char-level opacas
   - **Nuestros 4 generadores (deletion/substitution/position/affix) NO cubren estos casos.**
   - El test privado probablemente tiene una mezcla amplia de reglas que NINGÚN approach actual resuelve bien.

### Implicancia para v3

- **Easy families ya saturadas** (numeral 100%, unit_conv ~100%, gravity ~99%). Cero margen para mejorar ahí.
- **Bit_manipulation**: 1 000 ejemplos sintéticos limpios podrían subir el ceiling.
- **Cipher**: 1 000 ejemplos sintéticos podrían ayudar (el LB +0.01 probablemente vino de ahí).
- **Transformation**: nuestro v3 cubre solo 4 sub-tipos. El test privado tiene MÁS variedad. **v3 va a mejorar EN NUESTROS 4 sub-tipos pero seguirá fallando en los otros.**

### Predicción honesta para v3 LB

```
Mejora esperada por familia:
  numeral, unit_conv, gravity: ±0 (saturadas)
  bit_manipulation: +0.05 a +0.10 (1000 ejemplos clean ayudan)
  cipher: +0.05 (programmatic > heuristic)
  transformation (nuestros 4 sub-tipos): +0.50 a +0.90 (de 0.004 a 0.5+ es plausible)
  transformation (otros tipos): SIN CAMBIO (no los cubrimos)

LB privado estimado:
  Optimista: 0.84 → 0.87 (rompe plateau público)
  Realista:  0.84 → 0.85-0.86
  Pesimista: 0.84 → 0.84 (transformation no es la mayoría del test)
```

### Próximos pasos (v0.4.0+)

1. **Agregar más rule classes a transformation**: arithmetic_op (encriptado como /,|,\ + operaciones), char_pair_swap, complex_permutation. Probablemente 6-10 clases más.
2. **Investigar unit_conversion regression**: ¿alpha del precision fix mal calibrado? O dejar el original sin tocar (la teacher's CoT funcionaba mejor para esta familia).
3. **MoE-tied LoRA**: lo dejamos pendiente. Probarlo aislado en v0.5.0.

---

## v0.1.0 — `baseline-unsloth-v1`

**Submitted**: 2026-05-28 03:00 UTC | **Scored**: ~04:20 UTC
**Public LB**: **0.83** (primer submission, baseline)
**Kernel**: `gastonz195/baseline-unsloth-v1` v2 · **Recipe**: `adapters/baseline-unsloth-v1/recipe.md`

### Qué hicimos

- Fork **literal** de `dgxchen/training-with-unsloth-to-achieve-0-85-lb` (notebook público con 392 votos).
- Dataset: `dgxchen/nemotron-cot-tong` tal cual (7 830 rows con `generated_cot` filtrado).
- Hyperparams idénticos: rank 32, alpha 32, lr 2e-4, batch eff. 32, 1 epoch.
- **Sin modificación alguna en la receta.**
- Único bug en el camino: omití `machine_shape: NvidiaRtxPro6000` en el metadata del primer intento → la default GPU (T4 sm_75) no soporta los kernels Blackwell de Mamba → `CUDA error: no kernel image is available`. Fix: agregar el `machine_shape`. Segundo intento corrió OK.

### Resultado

- Training: 6h 47min en RTX Pro 6000.
- Adapter: 3.37 GB.
- LB público: 0.83. (El autor original reporta 0.85; nuestro 0.83 está dentro del rango de varianza por batching/seed.)
- Equipos en LB top: 0.87 (3 equipos cuando submitimos; ahora 8). Plateau denso en 0.86.
- Posición estimada: top 20% (no la cheque exacta).

### Conclusión

**Pipeline end-to-end validado.** Tenemos UNA submission al LB que sirve como safety net mínima (en peor caso, esta queda como una de las 2 finales). Receta pública reproducible.

---

## Apuestas / preguntas abiertas

- [ ] **El +0.01 de v0.2.0 es señal o ruido?** Para confirmar: ablation con (estructura sola) vs (precision fix sola) en runs separados. Cada uno cuesta ~7h Kaggle.
- [ ] **Eval local roto** — sin él no sabemos per-familia dónde estamos perdiendo. Prioridad alta arreglar.
- [ ] **Romper plateau 0.86-0.87**: hipótesis pendiente — MoE-tied LoRA (`moe_tie_weights=True`) podría ser la palanca grande que falta probar.
- [ ] **Teacher LLM para CoT regeneration**: la heurística actual solo wrappea — no regenera reasoning. Para subir más fuerte habría que correr un teacher (Nemotron mismo en Kaggle) sobre los 7 830 ejemplos.

## Próximos pasos sugeridos (orden de prioridad)

1. **Fixear el eval kernel** (3er intento) — sin diagnóstico per-familia estamos ciegos.
2. **Ablation run**: regenerar el dataset con SOLO el precision fix (sin la estructura) y entrenar otra LoRA. Comparar.
3. **MoE-tied LoRA**: cambiar `target_parameters` para usar shared B across experts. Adapter pesaría 30-50 MB en vez de 3.4 GB y podría generalizar mejor.
4. **Teacher-based CoT regeneration**: si los dos pasos previos no rompen plateau, montar pipeline de teacher LLM (Nemotron base sobre Kaggle).

---

## v0.3.0 — INCIDENTE: training v3 (2 epochs) killed por límite 12h Kaggle

**Pushed**: 2026-05-29 17:50
**Killed**: 2026-05-30 ~05:50 (al cumplirse 12h exactas)
**Adapter recuperado**: NINGUNO. `save_strategy="no"` significa que sin completar, no hay checkpoint.

### Lo que falló

Estimé el training de v3 con 2 epochs en 9-10h, pero el ratio real:
- v0.2.0 baseline: 7 830 rows × 1 epoch = 6h 47min puro training
- v3 con 2 epochs: 9 000 rows × 2 = 18 000 examples
- Aunque las CoTs son 5× más cortas (avg 691 chars), el overhead de model load + grad accum + offload domina
- **Real**: probablemente necesitaba ~13-14 h, excedió las 12h hard limit

### Lección

- **Nunca asumir que CoTs cortos = training proporcionalmente más rápido** sin medir. Grad accumulation y model overhead dominan la wall clock.
- **save_strategy="no" es peligroso** cerca del límite. Próxima versión: `save_strategy="steps", save_steps=50` para tener checkpoints intermedios.
- **2 epochs es ambicioso** para Kaggle free tier en este modelo. Default conservador = 1 epoch.

### Plan de contingencia ejecutado → v3p1

Pushed v3.1 (kernel `gastonz195/train-structured-v3p1`) con un único cambio respecto a v3:
- `num_train_epochs=2` → `num_train_epochs=1`
- ETA: ~6-7h → completa ~17:30-18:30 del 2026-05-30
- Resto idéntico (dataset v3, hyperparams baseline, mismo machine_shape)

---

## v0.3.0 (en training, ETA 2026-05-30 03:00-04:00 — KILLED, ver INCIDENTE arriba)

**Kernel**: `gastonz195/train-structured-v3` v1 · **Dataset**: `gastonz195/nemotron-structured-cot-v3`

### Qué cambió respecto a v0.2.0

| Cambio | Detalle |
| --- | --- |
| **Dataset 100% sintético** | 9 000 rows generados por `src/data/puzzle_generators.py` (9 generators) + `src/data/programmatic_cot.py`. Cero rows del CoT-tong original. |
| **Balanceado**: 1 000 por familia | Antes: muy desbalanceado (126 a 1 754). Ahora: 1 000 exacto por cada una de las 9 familias. |
| **CoTs cortos**: avg 691 chars (vs 3 400 de v0.2.0) | 5× más compactos. Razonamiento real estilo escuela (column_multiply, long_division, per-bit search, etc.), sin la "long division bizarra" del teacher. |
| **100% match contra gold POR CONSTRUCCIÓN** | Cada CoT termina en `\boxed{X}` donde X es el resultado del cómputo programático, garantizado correcto. |
| **2 epochs** (vs 1 en v0.2.0) | Justificado por LIMA + Olmo3-7B paper (Data Repetition Beats Data Scaling in Long-CoT SFT, 2026). |
| Mismos hyperparams en todo lo demás | rank=32, alpha=32, lr=2e-4, batch eff=32, max_seq=8192. Atribución limpia: cualquier cambio vs v0.2.0 viene de la data. |

### Infraestructura nueva

- `src/data/math_primitives.py` (300 LOC, 134 tests): ColumnAdd/Sub/Mul + LongDivision con walkthrough escolar
- `src/data/prompt_parsers.py` (130 LOC, 30 tests): parser por familia
- `src/data/programmatic_cot.py` (650 LOC, 60+ tests): 9 generators de CoT con razonamiento real
- `src/data/puzzle_generators.py` (450 LOC, 180 tests): 9 generators de puzzles random (prompts + gold)
- `src/data/build_synthetic_dataset.py` (orquestador): produce 9000 rows balanceadas con yield 98-100%
- `tests/test_dataset_integration.py` (18 tests): valida invariantes sobre el dataset producido
- `tests/test_kernel_lint.py`: static validator para metadata Kaggle (no más push-and-discover bugs)

**Suite total: 468 tests, todos pasan**.

### Lo que NO cubre v0.3.0 (gap identificado en eval)

- Transformations con reglas más allá de nuestros 4 sub-tipos (deletion/substitution/position/affix). El holdout limpio mostró que el test real tiene **arithmetic operations encriptadas** y **char-level rules opacas** que NO modelamos. v0.4.0 debería extender.
- Sin teacher LLM regeneration (decisión consciente — programmatic > teacher por construcción).
- Sin MoE-tied LoRA (palanca arquitectónica diferida a v0.5.0).

### Predicciones (a verificar)

```
LB privado:
  Optimista (cubre buena parte del test): 0.84 → 0.87
  Realista:                                0.84 → 0.85-0.86
  Pesimista (transformations dominan):    0.84 → 0.84 / -0.005
```

### Resultado (a completar cuando termine training)

- [ ] Training v3 status: _________
- [ ] Adapter generado: _________
- [ ] Submission: _________
- [ ] LB score público: _________
- [ ] Eval local per-familia: _________

---

## Referencia: leaderboard

| Fecha | Equipos | Top público | Plateau denso | Nuestro mejor |
| --- | ---: | ---: | ---: | ---: |
| 2026-05-02 | 2 575 | 0.87 (3 equipos) | 0.86 (~15 equipos) | — (sin submission) |
| 2026-05-26 | 3 526 | 0.87 (8 equipos) | 0.86 (~25 equipos) | — |
| 2026-05-28 | 3 526 | 0.87 | 0.86 | **0.83** (v0.1.0) |
| 2026-05-29 | 3 526 | 0.87 | 0.86 | **0.84** (v0.2.0) |
| 2026-05-30 (ETA) | — | — | — | **TBD** (v0.3.0) |
