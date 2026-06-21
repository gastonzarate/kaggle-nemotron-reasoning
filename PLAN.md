# Plan de competencia — NVIDIA Nemotron Reasoning Challenge

**Última actualización**: 2026-05-26
**Deadline final**: 2026-06-15 23:59 UTC → quedan ~20 días (~3 semanas)
**Estado del repo**: data bajada, notebooks públicos clonados, nada de código propio escrito todavía.

---

## 0. Recordatorio rápido

- **Modelo base**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (30B total / 3B active, MoE hybrid Mamba-Transformer)
- **Submission**: LoRA adapter (rank ≤ 32, target modules incluyendo `in_proj` Mamba), empaquetado en `submission.zip`
- **Eval**: vLLM server-side corre `base + LoRA`, `temperature=0`, `max_seq=8192`, `max_tokens=7680`, extrae `\boxed{respuesta}`, exact-match con normalización
- **Leaderboard hoy**: 8 equipos al techo público 0.87, ~25 equipos en 0.86. **3 526 equipos inscriptos**.

---

## 1. Hallazgos del foro (sesión 2026-05-26)

### 1.1 Cupo de Google Cloud G4 — NO EXISTE

NVIDIA no ofrece compute gratuito a participantes. El "powered by Google Cloud G4 VMs" del marketing se refiere a que **Kaggle corre el eval** en G4 VMs server-side.

Evidencia:
- Thread anclado **"How to Get Started + Resources"** (Jamil C Semaan, host, 58 votos): cero mención a form / cupo.
- Thread **"New Contestant - Urgent Compute Resources"** (Adesina, 3 días atrás): 0 respuestas a pesar de taggear a los hosts.
- Lo único que mencionan oficialmente es un **Brev launchable** (`https://brev.nvidia.com/launchable/deploy?launchableID=env-32kC34ErT9wsqTcJyaKMxBEuhr2`) que requiere comprar créditos.

**Opciones de compute reales**:

| Opción | Costo | Suficiente para... |
|---|---|---|
| Kaggle Notebooks gratis (T4 16 GB, 30 h/sem) | $0 | QLoRA en Nemotron 30B, ~7 h/run de 1000 steps |
| Brev launchable (créditos NVIDIA) | $$ | Similar a otros clouds |
| RunPod / Lambda 1× H100 80GB | ~$2/hr ($5-15/run) | Training holgado, iteración rápida |
| 3090 24 GB local (prestada) | $0 | QLoRA con dolor, batch=1, ~10 h/run |
| 3070 8 GB / Mac M1 | — | NO sirve para training |

### 1.2 BUG en training data (palanca grande)

**Ashutosh Kumar** reportó (en el thread anclado de Jamil):

> **Bit manipulation**: de 1 513 ejemplos con línea `Result`, **764 (50.5 %) tienen MISMATCH** entre `Result` y la respuesta `\boxed{}`. La mitad enseña respuestas incorrectas.
>
> **Equation transformation**: 49 % tienen length mismatches I/O que rompen el approach char_map. Otro 44 % tienen `?` (chars desconocidos).

Script disponible: `https://storage.googleapis.com/kaggle-forum-message-attachments/3428335/39613/check_bugs.py`

**Importante**: el data crudo de Kaggle (`train.csv`) solo tiene `(prompt, answer)`. El bug está en los **CoT-augmented datasets sintéticos públicos** que todos forkean (huikang, konbu17). Limpiar/regenerar CoT desde cero es la palanca más obvia para romper el plateau 0.87.

### 1.3 Metric Update del host (2026-05-17 aprox)

**Ryan Holbrook** (Kaggle staff host) deployó un fix:

> *"Deployed a fix to the metric that should address the issue of answers with the `}` character not being correctly extracted... since this fix should only improve scores, I won't be doing a rescore. Instead, just resubmit anything you want scored again."*

**Implicancias**:
- Si tu respuesta contiene `}` adentro de `\boxed{...}`, antes se truncaba mal.
- Submissions previas no se reescore automáticamente.
- Nuestro eval harness local debe replicar la lógica NUEVA del extractor. Sacar la implementación directamente del notebook oficial `notebooks/community/nvidia-nemotron-submission-demo.ipynb` ya clonado.

### 1.4 Recursos oficiales NVIDIA (subexplotados)

Del thread anclado:

| Recurso | Para qué |
|---|---|
| **NeMo Data Designer** — `github.com/NVIDIA-NeMo/DataDesigner` | Generación de datos sintéticos por dominio |
| **NeMo Curator** — `github.com/NVIDIA-NeMo/Curator` | Filtrado/curation multi-modal |
| **NeMo RL** — `github.com/NVIDIA-NeMo/RL` | RLVR + GRPO a escala |
| **NeMo Gym** — `github.com/NVIDIA-NeMo/gym` | Entornos RL custom |
| **Nemotron 3 Nano RL guide** — `docs.nvidia.com/nemo/rl/nightly/guides/nemotron-3-nano.html` | Receta oficial RLHF para el base de este concurso |
| **GRPO RLVR notebook** (Nemotron-3-Super) | Ejemplo RL con rewards verificables |
| **Discord oficial** — `discord.gg/udbkJyWk` | Conversación más rápida que foro Kaggle |
| **Email** — `community@nvidia.com` | Preguntas directas a NVIDIA |
| Datasets HF: post-training-v3, RL-instruction_following, Agentic-v1, RAG, Personas | Material SFT/RL |

> **Insight competitivo**: el thread público "GRPO Training guide needed" confirma que **casi nadie del top está usando NeMo RL**, a pesar de que NVIDIA literalmente publicó el playbook. Hay ventana ahí.

### 1.5 Threads útiles para leer

| Thread | URL fragment | Por qué |
|---|---|---|
| Metric Update (host) | `/discussion/698106` | Reglas del extractor de respuesta |
| Rescore After Metric Update (host) | `/discussion/687798` | Política de rescoring |
| How to Get Started + Resources | `/discussion/681745` | Catálogo oficial de recursos |
| How to break 0.86 ceiling | `/discussion/702447` | Discusión activa del plateau |
| 97.2% Gold-Conditioned Symbolic Solver | `/discussion/698293` | **NO es leak** — research oracle de lkevincc, recupera estructura simbólica latente de `equation_symbolic` usando la answer como constraint. Código: `github.com/lkevincc0/kaggle-nemotron-equation-symbolic` |
| Are all symbolic puzzles uniquely solvable? | `/discussion/702304` | Ambigüedad estructural en algunos prompts |
| Pattern-Matching for Symbolic Arithmetic | `/discussion/701981` | Approach algorítmico no-LLM |
| Clarification on Final Evaluation Settings | `/discussion/702473` | Detalles de submission |
| Score 0.87 started queuing at the top | `/discussion/701761` | Meta del plateau |

---

## 2. Plan de las próximas 3 semanas

Tres fases. Cada fase tiene un **deliverable concreto** y un **criterio de salida**.

### Fase A — Baseline + safety net (días 1-5, 2026-05-27 → 2026-05-31)

**Objetivo**: tener UNA submission al menos en 0.85 LB guardada. Es nuestra red de seguridad para una medalla de bronce.

Pasos:
1. Fork de `dgxchen/training-with-unsloth-to-achieve-0-85-lb` en Kaggle Notebooks.
2. Correr el training tal cual (T4 gratis, ~7 h).
3. Empaquetar el adapter via `ryanholbrook/nvidia-nemotron-submission-demo` (fork).
4. Submit a la competencia. Verificar que LB score ≥ 0.83.
5. **Selección final**: marcar esta submission como una de las 2 finales por las dudas.

**Criterio de salida**: tenemos `submission.zip` con score público ≥ 0.83 ya hecha y registrada como "final".

**Deliverable local**:
- `adapters/baseline-unsloth/` con `recipe.md`, `adapter_config.json`, `adapter_model.safetensors`, y el log del run.

---

### Fase B — Data cleanup + eval harness (días 6-12, 2026-06-01 → 2026-06-07)

**Objetivo**: dejar de pelear a ciegas. Construir cómo medimos.

Pasos:
1. **Eval harness local** (`src/eval/harness.py`):
   - Split determinístico de `train.csv` en train (8000) / val (1500).
   - Verifier por familia replicando la métrica oficial post-fix:
     - bit_manipulation, cifrado, numeral: exact match con normalización (lower, strip).
     - gravitational, unit_conversion: tolerancia `±0.05` abs o `±0.5%` rel.
     - equation_transformation: pendiente — confirmar la lógica viendo el demo notebook.
   - Predictor genérico: recibe un modelo (vLLM o transformers) + LoRA path opcional, devuelve `predictions.csv`.
   - Output: accuracy global + por familia.

2. **Data cleanup**:
   - Bajar `check_bugs.py` del thread de Ashutosh.
   - Correr sobre los datasets sintéticos que estamos usando (los de huikang/konbu17).
   - Para bit_manipulation: regenerar CoT con verificación programática (Python solver enumera funciones booleanas candidatas por bit).
   - Para equation_symbolic: usar el solver de lkevincc como oráculo para generar CoT que SÍ refleje la regla latente.
   - Output: `data/synthetic/clean/` con un dataset auditado.

3. **Validar la limpieza**: entrenar mismo recipe que en Fase A pero con datos limpios. Medir delta en eval harness local + 1 submission para confirmar correlación local↔LB.

**Criterio de salida**: eval harness reporta accuracy local que correlaciona con LB (delta < 0.02 entre ambas mediciones). Tenemos al menos un adapter entrenado sobre data limpia.

**Deliverables locales**:
- `src/eval/harness.py` — runnable end-to-end con un modelo HF.
- `src/data/clean_cot.py` — pipeline de limpieza.
- `data/synthetic/clean/v1/train.parquet`.
- `adapters/clean-data-sft/` con recipe.md.

---

### Fase C — Una palanca diferencial (días 13-20, 2026-06-08 → 2026-06-15)

**Objetivo**: romper 0.86. Elegimos UNA bala — no las tres.

Tres candidatos, ordenados por mi recomendación:

#### C1 (recomendado): GRPO sobre bit-manipulation + unit-conversion
- **Por qué**: estas dos familias tienen **rewards 100 % verificables programáticamente**. NeMo RL tiene la receta oficial. Casi nadie público lo hace.
- Pasos: implementar reward function para ambas, usar `nvidia/nemotron-3-nano-rl-guide` como base, entrenar LoRA sobre SFT-checkpoint de Fase B con GRPO 200-500 steps.
- Riesgo: instability del RL, lr scheduling sensible, requiere alquilar H100 (~$30-60 USD una tarde).

#### C2: Per-family preprocessing tipo lkevincc
- Para equation_symbolic, usar el solver simbólico para extraer la regla latente y embed eso en el prompt como CoT scaffold.
- Riesgo: solo afecta a ~16 % de los puzzles. Ceiling de mejora limitado.

#### C3: MoE-tied LoRA puro con datos limpios
- Continuar SFT pero con `moe_tie_weights=True` agresivo (todos los lados tied) sobre el dataset auditado.
- Riesgo: es la continuación natural del meta público; puede no diferenciar.

**Criterio de salida**: una submission con LB ≥ 0.86 (idealmente ≥ 0.88). Ambas finales seleccionadas.

**Deliverable local**:
- `adapters/final-{nombre}/` con recipe completa.
- `outputs/final_report.md` con el writeup interno (insumo para el writeup público si gana algo).

---

## 3. Decisiones pendientes (necesito tu input)

1. **Presupuesto compute**: ¿hasta cuánto USD estás dispuesto a invertir en alquilar H100 si lo necesitamos en Fase C? Cero (puro free tier) / $20-50 / $50-150 / abierto.
2. **Discord oficial**: ¿entrás vos? Te paso el link: `https://discord.gg/udbkJyWk`.
3. **Submission temprana**: ¿priorizamos asegurar una submission ya esta semana (Fase A "speedrun") o vamos ordenado?

---

## 4. Reglas autoimpuestas

- **No submitir nada sin pasarlo por el eval harness primero** (una vez que esté construido).
- **Todo adapter entrenado tiene `recipe.md`** con base model, dataset hash, hyperparams, eval local, git SHA.
- **`data/synthetic/` y `adapters/*/checkpoints/` no se commitean** (ver `.gitignore`).
- **Trust your CV, not the LB**: si local sube y LB no, hay un problema de distribución que investigar antes de seguir.
- **Selección final**: las 2 submissions finales son: (a) la más segura (Fase A o B) + (b) la más arriesgada (Fase C).

---

## 5. Estado actual del repo

```
kaggle-nemotron/
├── CLAUDE.md                       # contexto general del proyecto
├── PLAN.md                         # ← este archivo
├── .gitignore
├── pyproject.toml                  # uv 0.10.9, kaggle 2.1.0
├── data/
│   ├── raw/
│   │   ├── train.csv               # 9500 rows (prompt, answer)
│   │   ├── test.csv                # 3 dummy rows
│   │   └── nvidia-nemotron-model-reasoning-challenge.zip
│   ├── processed/                  # (vacío)
│   └── synthetic/                  # (vacío)
├── notebooks/
│   └── community/                  # 6 notebooks top clonados
│       ├── nvidia-nemotron-submission-demo.ipynb  ← official submission
│       ├── dgxchen_training-with-unsloth-to-achieve-0-85-lb/
│       ├── dennisfong_nvidia-nemotron-sfttrainer-training/
│       ├── huikang_end-to-end-finetuning-for-lb-0-85/
│       ├── huikang_tinker-submission-notebook/
│       ├── konbu17_nemotron-sft-lora-with-cot/
│       └── kienngx_nvidia-nemotron-training-cot-labels/
├── src/                            # (estructura vacía)
├── configs/
├── adapters/
└── outputs/
```

Sin commits aún. Sin código propio escrito.
