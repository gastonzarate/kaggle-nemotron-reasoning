# QA — dataset v3 corregido (post-fix de coherencia)

**Fecha:** 2026-06-07
**Archivo:** `data/kaggle_datasets/structured-cot-v3/structured_cot_v3.csv` (9 000 filas, regenerado)
**Backup del buggy:** `*.BUGGY.csv` en las mismas carpetas.

## TL;DR

Dos hallazgos separados:

1. **Correctness/coherencia (ARREGLADO).** El v3 original tenía 1 376 filas (15.3%) donde el texto del CoT concluía un valor y el target re-boxeado decía otro (gravity 87.6%, unit_conversion 48.8%), más 12 filas de bit_manipulation con la respuesta lisa y llanamente equivocada. Causa raíz: `_match` en el build usaba `float()` + tolerancia 1% para TODAS las familias, lo que (a) parseaba strings binarios como decimales y dejaba pasar reglas erróneas, y (b) toleraba CoTs numéricos redondeados que diferían del gold oculto. **Fix:** exact-match salvo gravity/unit, y emitir `answer = valor que el CoT realmente deriva`. Resultado: **0 incoherencias, 0 respuestas mal, 453 tests verdes.**

2. **Distribución/cobertura (NO arreglado — causa probable del drop).** La revisión por muestra (1 subagente por tipo, resolviendo cada puzzle de forma independiente y comparando contra prompts reales de `train.csv`) muestra que el dataset sintético entrena una distribución que **casi no se solapa** con el test real en varias familias. Esto explica un drop que la corrección de coherencia NO va a recuperar.

## Detalle por familia (muestra de 6 sintéticas + 6 reales c/u)

| Familia | Correctness interna | Match de distribución vs train.csv real |
| --- | --- | --- |
| gravity | 6/6 dentro de ~1% y coherentes. CoT redondea intermedios; un caso con error literal `9.99/2 = 5`. | Fraseo/longitud OK. g sintético hasta ~24 vs real ≤~19. Sintético incluye prompts de 3 ejemplos (real 4-5). |
| unit_conversion | 5/6 exactas, 1 en el borde de redondeo. CoT usa fit de 2 puntos (no LSQ) y reporta `b` a ojo. | Fraseo/nº ejemplos OK. Pendientes sintéticas más altas; un caso con `b≠0` fuerte ausente en la muestra real (real 6/6 proporcional puro). |
| numeral | 6/6 correctas, CoT sólido. | Real (muestra) 6/6 **Roman con N≤100**. Sintético 70/30 Roman/base-k y Roman sesgado a N grande. Posible sobre-gasto en base-k. |
| cipher | 6/6 correctas, CoT completo y sólido. | **Vocabulario divergente** (solo 26/68 palabras compartidas). Sintético siempre cubre toda la query (puro lookup); ~1/3 de los reales tienen chars no vistos que exigen generalizar. |
| bit_manipulation | 6/6 correctas, **sin ambigüedad** (fix OK). CoT tiene reversal LSB/MSB no explicado en la presentación. | **GAP grande**: sintético muestra 21-24 ejemplos vs real **7-10**; sintético solo modela reglas **por-bit independientes**, pero ~3/6 reales usan **acoplamiento entre bits** (shifts/rotations/majority/choice) que NINGUNA regla por-bit puede representar. |
| **transformation** (cryptarithm_deduce/guess, equation_numeric_deduce/guess) | 6/6 c/u correctas internamente (1 caso ambiguo en equation_numeric_deduce). | **GAP SEVERO Y UNÁNIME.** Los 4 sub-tipos (deleción/sustitución/posición/afijo) solo conservan/borran/mueven chars existentes. Los 6 prompts reales NO matchean NINGUNO: son **aritmética con operador oculto** (`84-33=51`, `92+76→7692`) y **transformaciones char-level con cambio de longitud** cuyo output contiene **símbolos nuevos ausentes del input** — imposible bajo nuestras 4 clases. El sub-tipo de afijos hasta usa un alfabeto artificial `PSXQZ` que se ve obviamente sintético. |

## La causa más probable del drop

- **transformation = 44% del dataset (4 000/9 000)** pero **0% de cobertura** de la distribución real de esa familia (real ≈17%). Entrenamos al modelo a aplicar con confianza deleción/sustitución/posición/afijo a puzzles reales que necesitan aritmética/reescritura → respuestas confiadamente equivocadas, y envenena una fracción enorme del entrenamiento.
- **bit_manipulation**: distribución más fácil (más ejemplos) y faltan clases de reglas (shift/rotate/majority/choice).
- **cipher/numeral**: sesgos de vocabulario/rango y ausencia del caso "char no visto".

## Qué NO arregla la corrección de coherencia

Nada de lo de arriba. La corrección hace que el dataset sea internamente correcto, pero **no cambia qué distribución enseña**. Para recuperar el drop hay que **rediseñar los generadores** (sobre todo transformation: agregar aritmética-con-operador-oculto y reescritura char-level con longitud variable; rebalancear pesos por familia para acercarse a ~1/6 cada una; ampliar vocabulario de cipher; agregar shifts/rotations/majority/choice a bit; reducir nº de ejemplos a 7-10).

## Acciones aplicadas en esta corrección

- `src/data/build_synthetic_dataset.py`: `_solves_puzzle` (exact salvo float-families) + emit `answer=predicted` + assert `boxed==answer` byte-exacto.
- `src/eval/verify_synthetic.py`: verificador independiente exhaustivo (estructura + coherencia + correctness por re-solución).
- Dataset regenerado en las rutas canónicas; backups `*.BUGGY.csv`.
