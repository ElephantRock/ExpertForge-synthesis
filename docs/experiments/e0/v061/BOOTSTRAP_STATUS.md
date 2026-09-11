# E0 v0.6.1 Mechanism Bootstrap — Lane Status

**Lane branch:** `e0/v061-mechanism-bootstrap` (from diagnostics head `118c14f`)
**Authorization:** `V06_MECHANISM_BOOTSTRAP_AUTHORIZED` — record `E0_v0.6.1_V06_MECHANISM_BOOTSTRAP_AUTHORIZATION.json` (SHA-256 `937ff955...`), digests verified against the issued values.
**Scientific specification:** FIXED at contract SHA-256 `a70a910e...` (v0.6.1-candidate). Registry FROZEN_FOR_V06_MECHANISM_BOOTSTRAP at `a3e4444c...`.
**Decision program:** authoritative = `e0_v061_mechanism_decision_r1.py` (`ab2fb6f6...`, 245-state oracle + targeted regression PASS, operator-recomputed 2026-09-11); defective original preserved for provenance (`8fef40ef...`, REJECTED_FOR_AUTHORITY). **Independent recomputation from the authoritative files remains mandatory before MECHANISM_SELECTION_EXECUTION_RELEASED** (authorization verification_note).

## Authorized scope (contract §16 only)
generator/verifier/StructSig, structural-support audit, leakage/shortcut audits, P0 snapshot binding, metric/sampler/decision implementation, token/resource smoke tests, replay validators, reporting schemas/manifests.

## NOT authorized
R/P scoring, comparison-seed scoring, v0.6 student qualification, Q3/source qualification, final E0 corpus, A0/downstream evidence, architecture experiments, Engram/HCM, contract-science/threshold/seed changes.

## Bootstrap progress log

- 2026-09-11: lane created; governance artifacts bound (9 files, digests verified vs issued values; original design manifest `b707a125...` arrived and verified); r1 decision program operator-recomputed (245-state oracle + regression PASS, digests unchanged). Commit `3c28c50`.
- 2026-09-11: **P0 snapshot captured and verified** (`P0_SNAPSHOT_VERIFICATION.json`): all six file digests bound; `model.safetensors` local SHA-256 **matches the observed upstream value exactly** (`ebfa4e2f...`, 166,029,852 bytes); config architecture matches the registry on all 12 fields; classification form instantiated — backbone without LM head **44,670,976** (exact registry match), classifier 1,539, total **44,672,515** (exact match); LM head genuinely untied (checkpoint `embed_out.weight` → runtime `lm_head`, distinct values). Open item: no standalone LICENSE file at the pinned revision — README.md (`9a572054...`) bound as the license record pending authority confirmation.

## Remaining Section-17 bootstrap items (next work, in planned order)
1. CMDR-MSEL-v0 generator extension + corpus materialization (402k examples, namespace `ExpertForge-E0-v061-msel`) + independent proof verifier
2. CMDR-StructSig-v1 implementation + structural-support/depletion audit
3. Leakage audits (family/rendered-string/StructSig) + shortcut audits (Q_shortcut ≤ 0.38)
4. Metric (§9) + MSEL-ExampleStream-v1 + decision-binding implementations, with deterministic replay validators
5. R1/R4/R16 nested-selection digests + corpus split digests
6. Token-length audits under both tokenizer families (CMDR-Lex-v1 and frozen P0 native)
7. 400-update resource smoke projections (R16, P-FT@R1, P-RANDOM@R1) — burned smoke corpus only
8. Frozen manifests for all implementations; independent recomputation of the authoritative r1 decision-program digest

