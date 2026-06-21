# Auditoría completa del stack LoRA — técnicas avanzadas (v4, 2026-06-10)

Auditoría end-to-end: mecánica del eval → targets a nivel token → arquitectura del adapter →
config de entrenamiento. Cada técnica está verificada contra cómo el eval REALMENTE corre
(vLLM + LoRA, temp=0, max_tokens=7680, extracción `\boxed{}`); lo incompatible se descarta
explícitamente — un train/eval mismatch silencioso es peor que no optimizar.

## Hallazgos de la auditoría

### A1. El extractor oficial toma el ÚLTIMO `\boxed{}` no-vacío — y maneja boxed sin cerrar
`src/eval/extractor.py` (port verbatim del demo oficial post metric-fix): `re.findall(r"\\boxed\{([^}]*)(?:\}|$)")`
→ último match no-vacío; el `(?:\}|$)` rescata un boxed truncado al final del texto.
**Implicancia explotable**: se puede emitir un boxed provisional ANTES de verificar; si la
generación muere en la verificación, el provisional gana. → Técnica T1.

### A2. Presupuesto de tokens: los teacher rows ROZAN el límite del eval
Con el tokenizer real de Nemotron: teacher rows = mediana 5712, **p95 7334, max 7579 tokens**
(prompt+target) vs `max_tokens=7680` de generación en el eval. El modelo aprende longitudes de
traza que apenas caben → en problemas difíciles nuevos puede pasarse del límite y nunca boxear
(= cero automático). Los solver rows están cómodos (mediana 858, max 2705). → refuerza T1.

### A3. `<think>`/`</think>` son tokens ÚNICOS del vocabulario
Formato barato y nítido; el armazón cuesta 2 tokens. El mask-template `<|im_start|>assistant\n`
tokeniza estable (4 tokens) → el completion masking por subsecuencia es confiable.

### A4. Guardrail vLLM (técnicas DESCARTADAS a propósito)
El eval carga el adapter con vLLM `enable_lora` (max_lora_rank=32). Por eso:
- **DoRA: NO** — vLLM no soporta adapters DoRA; fallaría o degradaría silencioso.
- **rsLoRA: NO** — cambia el factor de escala (α/√r); si vLLM ignora el flag, el scale de
  inferencia difiere del de training → mismatch silencioso.
- **lm_head en target_modules: NO** — la comunidad (dgxchen) midió que empeora LB.
- Se queda LoRA vainilla r32/α32 sobre q,k,v,o,in_proj,out_proj,up,down — lo único con
  semántica idéntica train↔eval garantizada.

## Técnicas aplicadas (Run B)

### T1. Commit-then-verify + seguro anti-truncación (original de esta auditoría)
Reestructura del CoT en gravity y bit (solver rows): derivar → **`Answer: \boxed{X}` DENTRO
del think** → recién después la verificación → `</think>` → boxed final. Doble efecto:
1. **Conductual**: el modelo aprende a comprometerse con una respuesta y luego chequear, en
   vez de divagar sin boxear (la causa de los ceros por truncación en temp=0).
2. **Mecánico**: si la generación trunca durante la verificación, el extractor (último boxed,
   A1) rescata el provisional. Win-win sin costo.
Implementado como `commit_early=True` (opt-in del corpus real; el pipeline sintético no cambia).

### T2. Completion masking (ya en Run A)
Loss SOLO sobre el turno assistant. Lo usa el ganador; el plateau (forks de dgxchen) entrena
sobre la secuencia completa — gastan capacidad modelando los prompts.

### T3. Gold-conditioned rule disambiguation (transformation extendido)
Entre las reglas consistentes con los ejemplos, elegir la que reproduce el gold (esa ES la regla
latente; misma idea que el "97.2% Gold-Conditioned Symbolic Solver" del foro, aplicado a
generación de training data, que es donde es legítimo). + CSP símbolo→dígito con backtracking
y ops extendidas (±1, floordiv, mod, glifo-como-signo, little-endian).

### T4. Upsampling de familias difíciles (×2 bit + transformation)
"Repetition beats scaling" para long-CoT SFT (Olmo3): 2-epochs locales exactamente donde el
eval pierde puntos, sin pagar 2 epochs del corpus entero (límite 12h de Kaggle).

### T5. NEFTune (noise α=5)
Ruido en embeddings durante SFT; mejora documentada de generalización, 1 línea, sin efecto en
inferencia (el adapter resultante es LoRA vainilla → A4 se respeta).

### T6. Cobertura bit 3-input (MAJ/CHOICE)
El espacio per-bit no expresa majority/choice que los puzzles reales usan; agregado al
discoverer (+10 filas reales resueltas; gate de unicidad sintético más estricto de regalo).

## Evaluadas y rechazadas (con motivo)
- Aritmética dígito-por-dígito más profunda: **el ganador la ablacionó — no mejora**.
- Loss-weighting sobre los tokens del answer: requiere tocar el CE fusionado de Unsloth —
  riesgo de romper un run de 7h por una mejora incierta. No con 5 días de deadline.
- 2 epochs del corpus completo: ~9-10h, demasiado cerca del límite de 12h (lección v3).
  T4 da el efecto donde importa.
- Curriculum por dificultad: evidencia mixta en SFT; el stratified batching ya balancea.
- Ensembling / self-consistency en inferencia: imposible — el eval es greedy single-pass y
  solo se submite el adapter.

## Secuencia de runs
- **Run A** (`train-real-v1`, corriendo): corpus real v1 + masking. Baseline fuerte esperado.
- **Run B** (`train-real-v2`, listo): corpus v2 = v1 + T1 + T3 + T4 + T6, notebook + T5.
- Selección final: 2 submissions = mejor de {Run A, Run B} + safety (0.84 v0.2.0 ya scoreada).
