# E0 v0.6.1 Mechanism Bootstrap — Lane Status

## Structural-support preflight r1 — COMPLETE: **PASS** (implementation-validating); support measurement delivered for authority sufficiency ruling

**SUPERSESSION NOTICE:** the r0 preflight (`74a80fd`, `STRUCTURAL_PREFLIGHT.json`) was ruled `STRUCTURAL_PREFLIGHT_INVALID_FOR_SUPPORT_CONCLUSION`: the original signature was non-injective w.r.t. the required family structure (min-of-orbit collapse, set-destroyed incidence multiplicity, WL colors treated as exact canonical labels, truncated internal hashes). Its structural-exhaustion interpretation — including the ~800–2,200 space estimates and the "generator expansion already required" conclusion — is **superseded and withdrawn**. The r0 evidence is preserved unmodified as the incident record; the r0 fail-closed governance behavior (402k withheld) was accepted and remains in force.

### V06-STRUCTSIG-REMEDIATION-1 (code commits `e433329`, `53fe79a`, `a67defe` — generator and proof verifier from `cfe0840` byte-identical throughout)

**Corrected `structsig_r1`:** per-variant exact canonical form via individualization-refinement (full-multiset WL with full-length SHA-256 colors to a stable class count; IR branching on smallest non-singleton cell with a 5,000-branch cap and a 10s per-variant wall guard, both fail-closed); exact bulk shortcut for interchangeable cells (identical masked-edge multisets — sound generalization of mutual-hyperedge-freedom and co-twin cases); family signature = SHA-256 over the canonical JSON of the **sorted triple** of canonical variant representations (lossless over the orbit).

**Discrimination tests (all PASS, from the full run):** adversarial multiplicity discrimination (distinct); renaming stability (all); fact/rule/premise order-permutation stability (all); orbit-shared-member tests — two families sharing one orbit member but differing in others receive **different** signatures in all 24 tested cases (the r0 defect class); repeated-signature groups — every group's members have byte-identical full canonical serializations (0 mismatches); deterministic regeneration (0 mismatches); cross-cell signature ambiguity 0; verifier/counterfactual-invariance errors 0 across all 16,000 pilot families; zero IR-cap and zero wall-guard exclusions (every family canonicalized exactly within budget).

### Support measurement (descriptive only; per authority ruling no collision-based failure)

Pilot: 2,000 families/depth/surface × 8 cells, burned namespace `ExpertForge-E0-v061-msel-preflight-r1`. Unique-family-signature counts at n=2,000: ID 1,139/1,010/887/744; STRUCT 1,395/1,152/1,033/802 (d1→d4). Occupancy fits (u = M(1−e^(−n/M))): ID 1,593/1,276/1,038/814; STRUCT 2,599/1,630/1,327/899. Chao1 at n=2,000: ID 2,232/1,756/1,490/1,548; STRUCT 5,781/3,108/2,798/2,036. Estimator formulas and assumptions are recorded in the evidence; both estimators assume iid uniform sampling and are **descriptive** — no sufficiency threshold was applied and none was invented.

**Factual reading (no sufficiency ruling made locally):** even under the corrected exact signature, unique-structure counts at n=2,000 (744–1,395 per cell) are far below the 32,000 families/depth the MSEL train pool specifies per surface. Under the contract's current semantics — which **permit** train-pool structural reuse but require `eval_STRUCT` to be signature-disjoint from the other three splits — the open question for the authority is whether the measured support can sustain a 2,000-family structurally-disjoint eval_STRUCT after a 32,000×2-family train-pool burn (both estimators suggest per-(surface,depth) signature populations on the order of 10³, and eval_STRUCT needs disjointness only from the burned signatures, not within itself). Materialization of the authoritative 402k corpus remains **HELD** pending the authority's SS4.4 sufficiency ruling on these numbers.

## Prior segment records


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

## Structural-support preflight r0 — SUPERSEDED INCIDENT RECORD (was `74a80fd`)

> **RULING: `STRUCTURAL_PREFLIGHT_INVALID_FOR_SUPPORT_CONCLUSION`.** The authority identified four defects in the r0 `CMDR-StructSig-v1` (min-of-orbit family collapse; `set()`-destroyed incidence multiplicity; 1-WL colors serialized as an exact canonical labeling; 64-bit truncated internal hash channel) plus two interpretation errors (post-hoc saturation estimates not computed by the committed runner; "any collision" treated as an exhaustion criterion although the contract permits train-pool structural reuse). **The r0 exhaustion interpretation below is superseded and withdrawn**; the record is preserved verbatim as the incident that exposed the faulty support metric. Accepted from r0: pilot ran only in a burned namespace; independent proof verification preliminary PASS; deterministic regeneration preliminary PASS; 402k correctly withheld; fail-closed governance behavior.

Run from clean implementation commit `cfe0840` (two-commit discipline; evidence in `STRUCTURAL_PREFLIGHT.json`). All correctness gates PASSED: 16,000 pilot families (2,000/depth/surface × 8 cells) built and accepted; **zero** verifier/invariance errors; deterministic regeneration (0 mismatches); renaming stability holds on all tested families both surfaces; zero cross-cell signature ambiguity. 

**The failure is the structural-space size.** Within-cell signature saturation at n=2,000: only 744–1,323 unique signatures per cell (37–66% collision rates; multiplicities up to 26). Saturation fits give effective per-cell structural spaces of only **~800–2,200 signatures** — while `CMDR-MSEL-v0` requires 32,000 train_pool families/depth and `eval_STRUCT` requires 2,000 families structurally disjoint from ~134,000 burned signatures. The v0.5-inherited family construction's structural diversity — after the contract's mandated name abstraction and slot-order removal — is one to two orders of magnitude too small for the v0.6.1 corpus at the requested scale.

Mechanism (why the space is small): most of the generator's randomness is structurally inert under `CMDR-StructSig-v1`. The 40-pred draws (from 1024), the entity draw (from 512), and the 12-rule order permutation all abstract away by design (name abstraction + sorting). Distractor predicates are chosen disjoint from every other predicate, so each distractor rule is an isolated component distinguishable only by sign pattern and arity; the surviving structural variety is essentially chain shape (fixed per cell) × distractor sign/connectivity patterns (~thousands).

**402k materialization NOT run** (held per the preflight gate). Per the authority's fail-closed instruction, the generator was NOT modified after observing this result — this is reported for contract review. Candidate directions for the authority (not implemented, not authorized): (a) enrich the generator's structurally-effective variation (e.g., cross-connected distractor topology, variable fact/rule counts, distractor entity-sharing patterns) as a v0.6.1 generator extension under a new contract revision; (b) redefine the structural-isolation scope (e.g., isolate eval_STRUCT at coarser granularity); (c) reduce the corpus scale. Option (a) alone preserves the contract's current SS4.3/SS4.4 semantics.

