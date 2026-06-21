# Taxonomía real de la familia "transformation" (train.csv)

**Fecha:** 2026-06-08
**Método:** análisis estructural de las 1 555 filas + ingeniería inversa de una muestra de 200 (8 subagentes en paralelo, cada regla verificada con código contra el `gold_answer` de la query).

## Conclusión (unánime entre los 8 batches)

**La familia "transformation" NO es char-rewrite genérico. Es aritmética de dos operandos con operador oculto.** Nuestros 4 generadores actuales (deletion / substitution / position / affix) cubren **~0%** de esta distribución. Esto explica por qué transformation (44% del dataset v3) hundió el score.

## Estructura universal

- **Todo LHS tiene exactamente 5 caracteres**: `AA op BB` → dos operandos de 2 chars con un **operador en el índice 2**.
- 3–5 ejemplos por prompt. RHS de longitud variable (1–4).
- Dos sub-tipos:
  - **ARITH_DIGITS**: operandos son dígitos visibles (`21-75`, `96$54`).
  - **ARITH_SYMBOL**: operandos son símbolos que codifican dígitos vía una biyección símbolo→dígito por puzzle (`("-]]`, `)(-)(`).

## Mecánicas confirmadas (verificadas contra gold)

1. **El operador es por-puzzle, no global.** El mismo glifo (`+`, `-`, `*`, `$`, `<`, etc.) significa operaciones distintas en puzzles distintos. Dentro de un puzzle, cada char-operador distinto mapea a su propia operación.
2. **Operaciones observadas** (frecuencia agregada aprox., contando por operador-char):
   - `mult` (a·b) — muy frecuente
   - `add` (a+b) — muy frecuente
   - `sub` / `abs_diff` (a−b o |a−b|) — frecuente
   - `concat` (escribir AABB pegados = "borrar el operador") — frecuente
   - `reverse_concat` (BBAA)
   - `floordiv` (a//b), `mod` (a%b) — raros
3. **Patrón dominante "little-endian / reverse-op-reverse" (RR.op.R):** invertir los dígitos de cada operando, aplicar la operación, e invertir el string del resultado. Aparece en la mayoría de los ARITH_DIGITS de varios batches.
4. **Ruido ±1 sistemático:** varias operaciones tienen un offset constante (`mult+1`, `sum−1`, etc.), reproducible y consistente por símbolo.
5. **Signo de negativos:** cuando el resultado es negativo, se antepone `-` literal o el propio glifo del operador como signo (`@25`, `)2`).

## Por qué muchas filas quedaron "no verificadas" (~40–60% verified=YES)

No porque la teoría falle, sino por límites de los datos/cómputo:
- **El operador de la query no aparece en los ejemplos** → la regla de ese operador no es derivable (puzzle subdeterminado).
- **Mapa símbolo→dígito subdeterminado** (>9 símbolos distintos, o pocos ejemplos) — los batches que usaron z3/CSP confirmaron que SÍ son aritmética-cifrada; los que no, las dejaron como UNKNOWN/CHARMAP_VARLEN honestamente.
- **Filas de ejemplo corruptas**: un batch notó que las filas con operador `*` no matchean ningún producto real (`83*13=9711`) — posible bug de datos en el train original.

Esto implica además que **algunos puzzles reales son intrínsecamente ambiguos/irresolubles** desde los ejemplos dados — útil saberlo para no sobre-optimizar.

## Spec para el rediseño del generador de transformation

Reemplazar los 4 sub-generadores por UN generador de **aritmética con operador oculto**:

- LHS de 5 chars: `AA op BB`, operador en índice 2.
- Operandos: 2 dígitos (sub-tipo digits) o 2 símbolos con cifrado símbolo→dígito por-puzzle (sub-tipo symbol).
- Sortear 1–3 operadores distintos por puzzle, cada uno mapeado a una operación de: {add, sub, abs_diff, mult, concat, reverse_concat, floordiv, mod}.
- Variantes de rendering: big-endian directo vs RR.op.R (reverse-operands→op→reverse-result); offset ±1 ocasional; signo como `-` o glifo.
- **Garantizar que el operador de la query aparezca en los ejemplos** (si no, el puzzle es irresoluble — no generar esos).
- CoT que: identifica el operador del medio, deriva la operación de cada ejemplo, la aplica a la query con la convención correcta, y boxea el resultado (longitud variable).
- Rebalancear pesos por familia → ~1/6 cada una de las 6 familias reales (no 44% transformation).

## Otras familias (de la revisión por tipo previa, `qa_v3_corrected.md`)

- **bit_manipulation**: bajar a 7–10 ejemplos; agregar shift/rotate/majority/choice (reglas con acoplamiento entre bits), no solo per-bit.
- **cipher**: ampliar vocabulario para cubrir el del train real; incluir casos con chars de la query NO vistos en ejemplos.
- **numeral**: rebalancear Roman hacia N≤100; medir el split real Roman vs base-k con más muestra.
- **gravity/unit_conversion**: ya coherentes; opcional limpiar la aritmética redondeada del CoT (un caso con `9.99/2=5`).
