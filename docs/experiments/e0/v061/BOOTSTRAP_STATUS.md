# E0 v0.6.1 Mechanism Bootstrap — Lane Status

**Branch:** `e0/v061-mechanism-bootstrap` | **Head:** see evidence commits below

## Current state: V06-GPU-BOOTSTRAP-REMEDIATION-1 COMPLETE — awaiting authority final §17 audit (incl. r1 decision-program byte-digest recompute) and explicit MECHANISM_SELECTION_EXECUTION_RELEASED.

### V06-GPU-BOOTSTRAP-EXECUTION-AUTHORIZED (r0) — RULED NOT ADMITTED, PRESERVED AS DIAGNOSTIC
The r0 smokes/replays executed on a firewall weakened post-observation
(family-ID-only after the both-class gate failed) and under a drifted runtime
(system Python / transformers 4.50.0 instead of the frozen .venv 5.16.1
stack). Authority §17 audit ruled items 21/23 NOT ADMITTED, item 20 REOPENED.
See `incidents/STRUCTSIG_FIREWALL_GATE_WEAKENED_POST_OBSERVATION.md`,
`incidents/EXECUTION_RUNTIME_SNAPSHOT_DRIFT.md`, and the preserved artifacts
under `incidents/gpu_bootstrap_r0/` (r0 measurements: t400 5.33/3.62/3.62 min;
peaks 5.09/3.44/3.32 GiB; replay exact — diagnostic only).
Conceptual point recorded verbatim: **"duplicate StructSigs are allowed inside
MSEL" and "a StructSig already burned by MSEL may be reused in a later
bootstrap fixture" are not equivalent propositions** — the latter was
explicitly disallowed.

### V06-GPU-BOOTSTRAP-REMEDIATION-1 — COMPLETE: ALL GATES PASS
1. **Incidents + push** (`293b13f`): both incidents recorded; r0 evidence
   moved (not rewritten) to `incidents/gpu_bootstrap_r0/` with execution logs;
   full local chain pushed to the remote branch.
2. **GPU runtime freeze** (`e136ca7` code, `f23b41a` evidence —
   `GPU_RUNTIME_FREEZE.json`, ALL 7 GATES PASS): execution runtime = repo
   `.venv`, **field-identical to the frozen pre-execution snapshot**
   (Python 3.12.10 / transformers 5.16.1 / tokenizers 0.23.2 /
   torch 2.14.0+cu126 / driver 616.64 / pip-freeze SHA `ef6d062c…` /
   igraph `_igraph.pyd` SHA). Prior 5.16.1 record preserved as the frozen
   snapshot (not reclassified). P0 classification interface exact under this
   runtime (44,670,976 + 1,539, FP32, finite readout, both arms). Complete
   **402k P0 tokenizer audit reproduces the prior frozen audit exactly**
   (min 261 / med 283 / p95 286 / p99 353 / max 358 / max_token_id 49464;
   truncation 0; invalid 0). Supplementary binding: `sys.executable`,
   package RECORD SHA-256 for torch/transformers/tokenizers/numpy/scipy/
   scikit-learn/igraph/psutil, deterministic state QUERIED.
3. **Structurally fresh r2 fixtures** (`c1feed1`+`551507a` code, `b0f2604`
   evidence — `SMOKE_REPLAY_CORPORA_R2_EVIDENCE.json`, PASS): rejection
   sampling against the cumulative burn (MSEL 134,000 families / 28,138 sigs
   ∪ r0 fixtures 9,200 families / 5,576 sigs). **Both-class overlap = 0**
   (family IDs AND r3 StructSigs), verified twice — at generation and by an
   independent second pass over the WRITTEN files. Rejection rates recorded:
   smoke_r2 8,200 accepted / 42,731 candidates (19.2%; 34,531 sig
   rejections), replay_r2 1,000 / 6,359 (15.7%). fid rejections 0. Verifier
   0 errors. Within-fixture multiplicity permitted (production norm);
   replay additionally avoids smoke signatures (cumulative burn). No
   relaxations. New burned namespaces: `ExpertForge-E0-v061-smoke-burn-r2`,
   `ExpertForge-E0-v061-replay-burn-r2` (+`_dev` splits).
4. **Authoritative smokes + replay under the frozen runtime**
   (`4e1977e` harness binding — path-only diff, NO recipe changes;
   evidence bound `4e1977e`):
   | arm | t400 | projected 8k | peak VRAM | host RSS | gates |
   |---|---|---|---|---|---|
   | R16/C0 | 5.24 min | 131.0 min | 5.09 GiB | 1.7 GiB | PASS |
   | P-FT@R1 | 3.61 min | 90.2 min | 2.57 GiB | 2.1 GiB | PASS |
   | P-RANDOM@R1 | 3.67 min | 91.6 min | 2.44 GiB | 2.1 GiB | PASS |
   - r2-corpus P0 token audit: min 265 / med 310 / p95 357 / max 358 ≤ 384; trunc 0; unk 0.
   - Dev readouts (0.3333 at update 400) are resource/trajectory signals on
     a burned namespace, NOT capability evidence.
   - Deterministic model replay (`GPU_REPLAY_EVIDENCE.json`): C0/R and P0
     paths × 2 separate processes × 3 checkpoints (updates 40/80/120) —
     **exact match, 0 differences** (bitwise state digests, dev argmax
     vectors, float32 logits digests, metric scalars). r2 replay reproduces
     the r0 diagnostic result (also exact) on fresh fixtures under the
     frozen runtime.

### Remaining before execution release
1. Authority-side byte-level SHA-256 recompute of the r1 decision-program
   digest (§17 item 6; program `ab2fb6f6…` is operator-attested)
2. Authority inspection of the pushed harnesses/evidence
3. Explicit `MECHANISM_SELECTION_EXECUTION_RELEASED`

## Superseded state: r0 GPU bootstrap (NOT ADMITTED — diagnostic only)

### V06-GPU-BOOTSTRAP-EXECUTION-AUTHORIZED — r0 execution record (SUPERSEDED)

> SUPERSESSION NOTE: everything below describes the r0 execution, ruled NOT
> ADMITTED by the authority §17 audit (firewall weakened post-observation;
> runtime drift). Its "transformers 4.50.0 working-tree reality" claim was the
> drift symptom — root cause: r0 invoked the system Python instead of the
> frozen repo `.venv`; the frozen runtime was never absent. Preserved for the
> incident record; do not cite as §17 evidence.
- **Code commits (before any GPU execution):** `d4f0fae` (generator + smoke harness + replay harness), `ac23919` (CPU-validation fixes), `877cd41` (compare-evidence digest binding)
- **Burned fixture corpora** (`SMOKE_REPLAY_CORPORA_EVIDENCE.json`): smoke 8,000 families / 24,000 examples (`ExpertForge-E0-v061-smoke-burn`), replay 800 / 2,400 (`ExpertForge-E0-v061-replay-burn`); family-ID overlap with the 134,000-entry frozen MSEL burn index = **0 both**; verifier errors = 0. StructSig overlap with the burn is descriptive only (5,021 / 499) — skeleton recurrence is intrinsic to the CMDR template space (the burn itself averages ~4.8 families per unique signature).
- **Resource smokes** (`GPU_SMOKE_EVIDENCE.json`, seed 806915476, bound to `ac23919`):
  | arm | t400 | projected 8k | peak VRAM | host RSS | gates |
  |---|---|---|---|---|---|
  | R16/C0 | 5.33 min | 133.1 min | 5.09 GiB | 1.3 GiB | PASS |
  | P-FT@R1 | 3.62 min | 90.6 min | 3.44 GiB | 1.9 GiB | PASS |
  | P-RANDOM@R1 | 3.62 min | 90.6 min | 3.32 GiB | 1.9 GiB | PASS |
  - P0 native-token audit on the smoke corpus: min 263 / median 308 / p95 357 / max 358 ≤ 384; truncation 0; unknown 0.
  - Dev readouts at update 400 are resource/trajectory signals on a burned namespace, explicitly NOT capability evidence.
- **Deterministic model replay** (`GPU_REPLAY_EVIDENCE.json`): C0/R and P0 trainable paths, each trained twice in separate processes under the frozen deterministic preamble, 3 checkpoints each (updates 40/80/120) — **exact match, 0 differences**: bitwise state digests, full dev argmax vectors, float32 logits digests, metric scalars.
- **Environment facts verified during execution:** installed transformers is **4.50.0** (LM head attribute `embed_out`, `torch_dtype` kwarg) — the working-tree reality the earlier closure-corrigendum fixture also used. P-RANDOM `from_config` initially materialized FP16 from pythia's `config.json` `torch_dtype`; the harness's FP32 dtype assertion caught it and the construction now overrides to FP32 explicitly (authority constraint: no silent FP16 substitution).

### Next: authority §17 24-item blocking-matrix audit over this lane, then (only on explicit release) MECHANISM_SELECTION_EXECUTION_RELEASED work.

## Previous state: ALL CPU pre-execution gates closed. Awaiting 400-update GPU smokes + deterministic model replay.

### CMDR-MSEL-v0 corpus: FROZEN / IMMUTABLE / ACCEPTED
- 134,000 families / 402,000 examples / namespace `ExpertForge-E0-v061-msel` (permanently burned)
- 16 split JSONL byte identities frozen (`MSEL_MANIFEST.json` at `e63a116`)
- R1/R4/R16 membership frozen by contract ranking (`MSEL_RUNGS.json`, SHA `e83e6849...`)
- Burn index: 134,000 families, SHA `5d6dc175...`
- All audits PASS: verifier (0 errors), family-ID/render leakage (0), eval_STRUCT isolation (0 overlaps), shortcut (all ≤0.38 at chance)
- §4.4 structural support: PASS (contract-exact replay `a2b6be0`)

### Specification binding
- Candidate contract SHA-256: `a70a910e...`
- SPEC-BOUND addendum SHA-256: `eecccdc3...` (sidecar at `aef3f73`)
- Corrigendum (conservative-vs-exact burn distinction, frozen eval_STRUCT selector): `e335166`
- train_pool = ID surface only (addendum ruling)

### Canonicalization: structsig_r3 (BLISS via igraph 1.0.0) — ACCEPTED
- structsig_r3.py SHA: `fb8e4944...` | igraph wheel SHA: `faeff8ede0...` | _igraph.pyd SHA: `92d9e9e7...`
- Oracle: 266 structures, 8/8 gates PASS (clean replay `f0b54c7`)
- 4,000-family correctness pilot: 0 errors (`048420a`)
- Read-only frozen-corpus audit: all PASS (`f23f0dd`)

### §9 metrics: ACCEPTED (remediated)
- msel_metrics.py SHA: `dde9e4e9...` (frozen at `8b5a91f`)
- Hierarchical bootstrap: fail-closed (explicit family_ids_id + family_ids_struct, no fallback, 6-seed, triplet validation)
- 24/24 self-tests PASS including STRUCT-triplet, ±0.02 boundaries, and [0.02]*12 superiority boundary
- per_label_recall: 8 surface×depth cells

### MSEL-ExampleStream-v1: CANDIDATE ACCEPTED + full replay verified
- msel_stream.py: SHA-256 cycle ranking, 1,024,000 presentations, cross-boundary batches
- Full replay: all 18 arm×seed cases (R1/R4/R16 × 6 primary seeds) — implementation streaming SHA roots match independent oracle roots exactly (`CLOSURE_CORRIGENDUM_EVIDENCE.json` at `0b66816`)

### Pre-execution CPU gates: ALL PASS
| Gate | Evidence | Status |
|---|---|---|
| 16-file digest hard-stop | every runner | PASS |
| Independent proof verifier + cf_invariance | 134k families, 0 errors | PASS |
| Family-ID / rendered-string leakage | 0 across all pairs | PASS |
| eval_STRUCT StructSig isolation | 0 overlaps | PASS |
| Shortcut audits | all at 0.333333 (chance) | PASS |
| Full-corpus tokenizer audit | P0 max 358 ≤384, 0 trunc, IDs<50304; Lex max 194 ≤384 | PASS |
| P0 readout interface fixture | terminal=':', dim=512, logits [1,3] | PASS |
| 13 seeds + 48 substreams | all recomputed and match | PASS |
| Decision program r1 | SHA matches, 245-state oracle PASS | PASS |
| Report schema | SHA matches | PASS |
| Deterministic preamble | post-application queried and verified | PASS |
| Deterministic stream replay (§17.22) | 18/18 arm×seed full-stream root matches | PASS |

### Deterministic execution preamble: FROZEN
- `deterministic_preamble.py`: CUBLAS_WORKSPACE_CONFIG=:4096:8, are_deterministic_algorithms=True, TF32 off (matmul+cudnn), cudnn deterministic on, benchmark off
- Post-application queried and verified in corrigendum evidence

## Next authorized stages (pre-authorized after corrigendum PASS)
1. **400-update GPU resource smokes** (R16, P-FT@R1, P-RANDOM@R1) on fresh burned smoke-corpus namespaces — must include checkpoint/dev evaluation; observed wall ≤19.2 min; peak VRAM ≤12 GiB; host RSS ≤64 GiB
2. **Deterministic model replay** on a separately burned replay-corpus namespace — both C0/R-path and P0 trainable-path, twice under the frozen deterministic preamble; require exact selected-checkpoint ID, exact semantic predictions, exact recorded metric scalars

## Explicitly NOT AUTHORIZED
- R/P evidence scoring (requires MECHANISM_SELECTION_EXECUTION_RELEASED from project authority)
- Comparison-only seed scoring
- v0.6 student qualification
- Q3/source qualification
- Final E0 corpus generation
- A0 or downstream E0 evidence
- Architecture experiments, Engram/HCM
- Contract-science/threshold/seed changes

## Historical incidents (preserved, superseded)
- r0 preflight (`74a80fd`): invalid support conclusion (non-injective signature)
- r1 remediation (`9ec3603`): masked-edge bulk shortcut falsified by directed-cycle counterexample
- r2/r2A (`5344bbf`): custom IR closed — renaming-invariance failures under verified-automorphism pruning
- Pre-execution bundle (`95841d0`): partial PASS — bootstrap defect (family indices computed but never used), stream/tokenizer/runtime coverage gaps
- Remediation-1 (`c79479c`): partial PASS — STRUCT-family fallback, stream oracle coverage, tokenizer special/unknown fields, asserted (not queried) determinism
- All preserved under `incidents/` or as superseded evidence records
