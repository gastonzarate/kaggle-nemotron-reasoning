# train-real-v2 — corpus real + técnicas de la auditoría (Run B)

**Pusheado**: 2026-06-10 · **Kernel**: https://www.kaggle.com/code/gastonz195/train-real-v2 (v1)
**Auditoría**: `data/analysis/lora_audit_v4.md`

## Dataset: `gastonz195/nemotron-real-corpus-v2` (10 748 rows = 8 543 base + 2 205 upsample)
`build_real_corpus.py --upsample-hard` con el solver extendido:
- **transformation: 9.1% → 38.9% solver-solved** (605/1555; +463 vs v1 vía gold-conditioning,
  ops extendidas y CSP símbolo→dígito — 128 filas con mapa simbólico completo).
- bit 70.3% (+MAJ/CHOICE 3-input), resto igual a v1.
- sources: solver 7 913 · solver_ext 926 · teacher 1 909.

## Técnicas (vs train-real-v1)
1. **Commit-then-verify** en gravity/bit: `Answer: \boxed{X}` dentro del think antes de la
   verificación (seguro anti-truncación: el extractor oficial toma el ÚLTIMO boxed).
2. **Gold-conditioned disambiguation** + CSP en transformation.
3. **Upsample ×2** bit + transformation.
4. **NEFTune α=5** (SFTConfig).
5. Mismo masking/normalización/hyperparams que real-v1.

## Plan post-run
Submit: `uv run kaggle competitions submit nvidia-nemotron-model-reasoning-challenge -k gastonz195/train-real-v2 -v 1 -f submission.zip -m "real-v2: corpus real + audit techniques (gold-cond CSP, commit-verify, upsample, NEFTune)"`

## Resultado (a completar)
- [ ] Kernel status: ____
- [ ] LB público: ____ (comparar vs train-real-v1)
