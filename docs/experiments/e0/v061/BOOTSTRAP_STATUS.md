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

- 2026-09-11: lane created; governance artifacts bound (9 files, digests verified); P0 snapshot capture in progress.
