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

## Structural-support preflight — COMPLETE: **FAIL (fail-closed)**

Run from clean implementation commit `cfe0840` (two-commit discipline; evidence in `STRUCTURAL_PREFLIGHT.json`). All correctness gates PASSED: 16,000 pilot families (2,000/depth/surface × 8 cells) built and accepted; **zero** verifier/invariance errors; deterministic regeneration (0 mismatches); renaming stability holds on all tested families both surfaces; zero cross-cell signature ambiguity. 

**The failure is the structural-space size.** Within-cell signature saturation at n=2,000: only 744–1,323 unique signatures per cell (37–66% collision rates; multiplicities up to 26). Saturation fits give effective per-cell structural spaces of only **~800–2,200 signatures** — while `CMDR-MSEL-v0` requires 32,000 train_pool families/depth and `eval_STRUCT` requires 2,000 families structurally disjoint from ~134,000 burned signatures. The v0.5-inherited family construction's structural diversity — after the contract's mandated name abstraction and slot-order removal — is one to two orders of magnitude too small for the v0.6.1 corpus at the requested scale.

Mechanism (why the space is small): most of the generator's randomness is structurally inert under `CMDR-StructSig-v1`. The 40-pred draws (from 1024), the entity draw (from 512), and the 12-rule order permutation all abstract away by design (name abstraction + sorting). Distractor predicates are chosen disjoint from every other predicate, so each distractor rule is an isolated component distinguishable only by sign pattern and arity; the surviving structural variety is essentially chain shape (fixed per cell) × distractor sign/connectivity patterns (~thousands).

**402k materialization NOT run** (held per the preflight gate). Per the authority's fail-closed instruction, the generator was NOT modified after observing this result — this is reported for contract review. Candidate directions for the authority (not implemented, not authorized): (a) enrich the generator's structurally-effective variation (e.g., cross-connected distractor topology, variable fact/rule counts, distractor entity-sharing patterns) as a v0.6.1 generator extension under a new contract revision; (b) redefine the structural-isolation scope (e.g., isolate eval_STRUCT at coarser granularity); (c) reduce the corpus scale. Option (a) alone preserves the contract's current SS4.3/SS4.4 semantics.

