# E0 Windows Local Bootstrap Status

## Q2-UNQUALIFIED diagnostics — D0–D2 complete (diagnostic branch e0/q2-unqualified-diagnostics)

All outputs under `docs/experiments/e0/diagnostics/`, flagged `diagnostic_only: true` / `non_scientific: true`. No v0.5 record was modified; no D4 perturbations, Q3 scoring, corpus generation, or long runs were performed.

- **D2 — transport audit: PASS (all 6 checks).** Label mapping identical to the Q1 generator order; `<DECIDE>` terminal-token/index/collate verified; labels reaching cross-entropy match an independent gold-string mapping; bit-exact logit replay; the classifier provably reads the post-final-RMSNorm state at exactly `decide_index`; counterfactual variants are pairwise-distinct sequences (first difference at positions 41–90) with exact unigram/bigram multiset invariance — the input-side distinguishing signal exists.
- **D1 — tiny-set memorization (key falsifier): MEMORIZED.** 24 complete counterfactual families (72 examples, exactly balanced 24/24/24 labels, 18 per depth), trained through the exact production collate/model/loss/optimizer path: training accuracy reached 1.0000 (CE 0.0000) by update ~100–200 and stayed there through update 2,000, per-label accuracy 1.0 on all three labels. The implementation/optimization path can learn and can represent all three labels of complete families.
- **D0 — production-scale dynamics (C0, frozen seed 1647674144, 2,000-update window): optimization alive, model trapped in constant-label basins.** Parameters move (per-update Δ-norm 0.01 → 0.7); gradients flow (global norm 33.8 → ~2–5; classifier-weight norm 4.6 → 1.5; embedding norm 19 → ~0.001); batch CE declines only 1.32 → 1.10 — pinned at the ln(3)=1.0986 uniform floor. Predictions are degenerate: after ~update 50 every recorded batch is a constant-label prediction (entropy 0.0), cycling over the run between all-ENTAILED, all-UNKNOWN, and occasionally mixed states (73 distinct histograms, almost all degenerate); eval_ID SMA 0.333333 at every checkpoint, matching production.

**Combined D0–D2 reading (per issue #3's discriminator):** D2 rules out signal-transport defects; D1 rules out cannot-learn/cannot-represent failures of the implementation or optimizer path; D0 localizes the production-scale failure to generalization beyond memorization — the recipe at 24k examples does not escape the constant-label basin within its budget. This is evidence toward architecture/representation or task/recipe mismatch rather than implementation defect. D3 (representation discriminability) and D4 (single-factor perturbations) remain unexecuted pending project-authority review of this checkpoint.


## C2 full Q2 qualification — COMPLETE; harness terminal conclusion: **Q2 UNQUALIFIED** (final; held for project-authority closure)

All three frozen seeds completed 8,000 updates each under `exclude_norm_bias` from launch head `12bbe9e888ba1c1450ea6720ad80da9b4a628718` (clean tree at start; identical runtime snapshot binding; frozen Q1 digests re-verified per seed; no `scientific_failure.json`; each `best_checkpoint_sha256` matches its local uncommitted `best.pt`; parameter count 73,686,019 verified). Per-seed Q: 0.333333 / 0.333333 / 0.333333. Harness-emitted gate decision: `BELOW_FLOOR_NEXT_CANDIDATE_OR_UNQUALIFIED`, followed by the deliberate terminal exit `Q2 UNQUALIFIED: C2 remains below floor.` after all evidence was written — the pre-registered scientific conclusion, not an execution failure. Evidence: `q2_c2_evidence/seed_*_result.json` (byte-identical copies) and the updated `q2_m0_qualification_summary.json` (C0/C1 sections preserved).

Ladder outcome across all three candidates: C0, C1, C2 all BELOW_FLOOR at exactly chance on both surfaces; C3 does not exist; no recipe alteration was made at any point. Q2 closure, Q3 capability/headroom disposition, and final corpus authorization rest with the project authority per the frozen contracts.


## C1 full Q2 qualification — COMPLETE (evidence pushed; ladder decision held for project authority)

All three frozen seeds completed 8,000 updates each under `exclude_norm_bias` from launch head `226b670a9fe59f94395a3db995c67a1a2e2145c2` (clean tree at start; identical runtime snapshot binding across seeds; frozen Q1 digests re-verified per seed; no `scientific_failure.json`; each `best_checkpoint_sha256` matches its local uncommitted `best.pt`; parameter count 63,459,331 verified). Per-seed Q: 0.333333 / 0.333333 / 0.333333. Harness-emitted gate decision: **`BELOW_FLOOR_NEXT_CANDIDATE_OR_UNQUALIFIED`**. Evidence: `q2_c1_evidence/seed_*_result.json` (byte-identical copies) and the updated `q2_m0_qualification_summary.json` (C0 section preserved). No ladder inference locally; **C2 is NOT started** — the project authority recomputes the C1 gate. Per the frozen ladder, C2 is the final candidate; if it also remains below floor, Q2 is UNQUALIFIED and no C3 may be invented.


## C0 full Q2 qualification — COMPLETE (evidence pushed; ladder decision held for project authority)

All three frozen seeds (1647674144, 1110194409, 335767543) completed 8,000 updates each under `exclude_norm_bias` from launch head `7556f756146e59940ada037385c81fdf5325bc1e` (clean tree verified at start; identical runtime snapshot binding across seeds; frozen Q1 digests re-verified at each seed's setup; no `scientific_failure.json`; each `best_checkpoint_sha256` matches its local `best.pt`, which remains uncommitted under gitignored `local_data/`). Harness-emitted gate decision: **`BELOW_FLOOR_NEXT_CANDIDATE_OR_UNQUALIFIED`** (median Q = 0.333333; all three seeds at chance). Evidence: `q2_c0_evidence/seed_*_result.json` (byte-identical copies of the local source records) and `q2_m0_qualification_summary.json`. No ladder inference is made locally; C1 is **not started** — the project authority recomputes the frozen Decision-6 gate from the pushed records.


## C0 launch incident 001 — INFRASTRUCTURE_INVALID_PRE_MODEL (closed by remediation)

The first C0 launch attempt from `48b2597f...` crashed pre-model with `NameError: subprocess` in the scientific lineage block; zero seeds, updates, checkpoints, or results were produced. Preserved logs are bound by SHA-256 in `q2_c0_launch_incident_001.json`. Remediation per `Q2_C0_INFRASTRUCTURE_RETRY_RELEASE.md`: module-level `import subprocess`, shared `scientific_setup()` used by both the full `--candidate` path and the new non-scientific `--preflight` rehearsal of the exact scientific setup branch, plus `q2_c0_scientific_preflight.json` evidence. C0 retry is HELD until Commits C and D validate on GitHub.

**Remediation evidence (this commit, D):** from clean Commit C `7d0ebd9ec437c4d86c962faba021ffb3b63a00b4` — `py_compile` OK; `--preflight --candidate C0 --wd-scope exclude_norm_bias` PASS (seed 1647674144, P1 digests verified, lineage assembled on the previously failing branch, parameter count 53,232,643, Decision-22 groups 53,216,768 decay / 15,875 no-decay, all finiteness guards true, no scientific artifacts emitted); Q2 `--smoke` PASS; 16-update non-scientific path PASS under `exclude_norm_bias`. Both evidence manifests bind to `7d0ebd9...`. Q3 code untouched — no Q3 rerun required per the release.

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

Prelaunch integrity audit (P1–P4, `Q2_PRELAUNCH_INTEGRITY_AUDIT.md`): **CLOSED.**
- P1: scientific Q2 re-hashes all four frozen Q1 inputs at startup and fails closed before model construction. Verified locally: a tampered `train_ID.jsonl` aborts with the digest mismatch, exit 1.
- P2: scientific runs fail closed on non-finite microbatch loss, accumulated gradients, global grad norm, post-step parameters, and eval logits/metrics; `scientific_failure.json` records candidate/seed/update/failure code/code SHA/verified digests/last finite checkpoint. Verified locally: injected NaN produces `nonfinite_microbatch_loss` and emits **no** result.json or best.pt — an earlier checkpoint cannot rescue a diverged run.
- P3: stale `wd_scope_default` metadata replaced by `wd_scope: exclude_norm_bias (frozen Decision-22 / R1)`; scientific seed results record `code_git_commit`, `working_tree_clean_at_start`, `verified_q1_input_digests`, `runtime_snapshot_sha256`, preserved into the candidate summary.
- P4: two-commit evidence protocol — code-fix commit `35c6a8f21169202dbc297fa5908f5f09d6c7d39c` first; all smoke manifests in this commit were regenerated from that clean revision and record `git_commit = 35c6a8f...` (Q2 smoke PASS, Q3 S0C0/S0C1/S0C2 PASS, 16-update non-scientific path PASS under `exclude_norm_bias`).

C0 full Q2 qualification (three frozen seeds × 8,000 updates) is released once this evidence-refresh commit is validated on GitHub, per the audit's release condition.
