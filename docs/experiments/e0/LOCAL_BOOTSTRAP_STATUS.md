# E0 Windows Local Bootstrap Status

- Q1 local reproduction: **PASS** — all four frozen corpus hashes matched; pushed as `eca842fad54a52db638dd913189ef9ada1dc69a6` and verified on GitHub (issue #1 comment, 2026-09-07T20:14:09Z).
- Q2/Q3 pinned runtime: **INSTALLED** — torch 2.14.0+cu126, transformers 5.16.1, accelerate 1.14.0, safetensors 0.8.0, numpy 2.3.5, scikit-learn 1.8.0. CUDA 12.6 available; BF16 supported (RTX 3080 Ti, 12 GiB). See `q2_q3_runtime.snapshot.json`.
- Q2 harness (`m0_model.py`, `q2_m0_qualify.py`): **IMPLEMENTED, SMOKE PASS** — exact frozen parameter counts (C0/C1/C2), init audit, bit-exact repeatable forward/backward/step, deterministic data order, LR endpoints, metric sanity. See `q2_smoke_manifest.json`.
- Q3 harness (`q3_source_qualify.py`): **IMPLEMENTED, SMOKE PASS on all three registry candidates** — tokenizer A/B/C contextual single-token checks at pinned revisions for S0C0/S0C1/S0C2; full model interface checks: config matches registry, TPDS bound to `lm_head` with bit-exact replay and bit-exact two-pass determinism, BF16, resource-feasible within frozen ceilings. Peaks: S0C0 3.42 GiB CUDA (direct residency); S0C1 10.69 GiB CUDA / 10.1 GiB host (20 GPU / 12 CPU split); S0C2 10.41 GiB CUDA / 10.8 GiB host (25 GPU / 11 CPU split). See `q3_smoke_manifest.json`, `q3_smoke_s0c1.json`, `q3_smoke_s0c2.json`.
- Q2 training-path exercise: **PASS (non-scientific 16-update C0 run)** — periodic eval, best-checkpoint selection, reload, both-surface metrics, checkpoint digest all exercised; output under gitignored `local_data/`/`SMOKE_TRAIN/`.
- Q2 full 8,000-update qualification: **NOT STARTED — awaiting project-authority release of the harness/smoke commit.**
- Q3 capability/headroom: **BLOCKED on Q2 Arm-A pilot results** (per frozen release).
- Final 108,000-example corpus generation: **NOT AUTHORIZED until Q2/Q3 close.**

Open interpretation questions flagged to project authority (no frozen value altered):
1. Weight-decay scope: `weight decay 0.05` — applied to all trainable parameters by default (`--wd-scope all`, literal reading); `exclude_norm_bias` (common LLM convention) is implemented as a switch.
2. Training permutation seed derivation: `sha256("ExpertForge-E0-Q2|M0|perm|<seed>|<epoch>")[:8]` big-endian.
3. Source decision suffix appended with no trailing newline after `Answer:`; A/B/C contextual form chosen as `" A"/" B"/" C"` (all three single tokens for all registry tokenizers).
4. Q3 replay subset: within each label×depth cell of eval_ID, rank by `sha256("ExpertForge-E0-Q3|replay|<sample_id>")` ascending, take first 20 (240 total).
