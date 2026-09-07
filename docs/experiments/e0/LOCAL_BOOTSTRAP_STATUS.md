# E0 Windows Local Bootstrap Status

- Q1 local reproduction: **PASS** — all four frozen corpus hashes matched; pushed as `eca842fad54a52db638dd913189ef9ada1dc69a6` and verified on GitHub (issue #1 comment, 2026-09-07T20:14:09Z).
- Q2/Q3 pinned runtime: **INSTALLED** — torch 2.14.0+cu126, transformers 5.16.1, accelerate 1.14.0, safetensors 0.8.0, numpy 2.3.5, scikit-learn 1.8.0. CUDA 12.6 available; BF16 supported (RTX 3080 Ti, 12 GiB). See `q2_q3_runtime.snapshot.json`.
- Q2 harness (`m0_model.py`, `q2_m0_qualify.py`): **IMPLEMENTED, SMOKE PASS** — exact frozen parameter counts (C0/C1/C2), init audit, bit-exact repeatable forward/backward/step, deterministic data order, LR endpoints, metric sanity. See `q2_smoke_manifest.json`.
- Q3 harness (`q3_source_qualify.py`): **IMPLEMENTED, SMOKE PASS on all three registry candidates** — tokenizer A/B/C contextual single-token checks at pinned revisions for S0C0/S0C1/S0C2; full model interface checks: config matches registry, TPDS bound to `lm_head` with bit-exact replay and bit-exact two-pass determinism, BF16, resource-feasible within frozen ceilings. Peaks: S0C0 3.42 GiB CUDA (direct residency); S0C1 10.69 GiB CUDA / 10.1 GiB host (20 GPU / 12 CPU split); S0C2 10.41 GiB CUDA / 10.8 GiB host (25 GPU / 11 CPU split). See `q3_smoke_manifest.json`, `q3_smoke_s0c1.json`, `q3_smoke_s0c2.json`.
- Q2 training-path exercise: **PASS (non-scientific 16-update C0 run)** — periodic eval, best-checkpoint selection, reload, both-surface metrics, checkpoint digest all exercised; output under gitignored `local_data/`/`SMOKE_TRAIN/`.
- Q2 full 8,000-update qualification: **NOT STARTED — awaiting project-authority release of the harness/smoke commit.**
- Q3 capability/headroom: **BLOCKED on Q2 Arm-A pilot results** (per frozen release).
- Final 108,000-example corpus generation: **NOT AUTHORIZED until Q2/Q3 close.**

Interpretation questions — **RESOLVED by project authority** in `Q2_FULL_RUN_RELEASE.md` (commit `1070a61c4f48cef4d02bc6ac6305c31fe09b4117`), recorded on issue #1 and draft PR #2:
1. **R1 — weight decay: `exclude_norm_bias`** (frozen Decision-22 parameter-group semantics). Applied in Q2 scientific execution; scientific invocations requesting `--wd-scope all` fail closed. Q2 smoke and the 16-update path exercise were rerun under the corrected scope: PASS.
2. **R2 — permutation derivation: approved as implemented** — `sha256("ExpertForge-E0-Q2|M0|perm|<seed>|<epoch>")[:8]` big-endian, zero-based epochs. Frozen.
3. **R3 — Q3 answer context: exact leading-space continuations `" A"/" B"/" C"`, no trailing newline, no fallback forms.** Fallback removed; all three Q3 smokes rerun: PASS.
4. **R4 — replay subset: approved as implemented** — 20 per label×depth cell by ascending `sha256("ExpertForge-E0-Q3|replay|<sample_id>")`. Frozen.

Correction checkpoint status: the two narrow corrections (R1 enforcement, R3 fallback removal) are in this commit; all smoke manifests regenerated under the corrected rules and PASS. Full 8,000-update Q2 runs were **not** started from `f5d5a88`; C0 is authorized to begin from this checkpoint after GitHub-side validation.
