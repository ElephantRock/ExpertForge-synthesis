# E0 Windows Local Bootstrap Status

## D4-C4 prototype-gradient coupling — COMPLETE: `COUPLING_NOT_BOTTLENECK` (lineage: code `400de59` = Commit W → evidence in this commit)

**Preflight (from literally clean W, output under gitignored `local_data` until this commit): PASS.** The state-space gradient decomposition on the frozen 128-example batch at initialization measured: |g_FULL|=0.7499, |g_SG|=0.3681, |g_PROTO|=0.3823 (prototype component comparable to the query component, ratio 1.04); cos(g_FULL,g_SG)=0.9993, cos(g_SG,g_PROTO)=0.9974; **signed projection of the prototype component onto the query component +0.3813 — prototype backprop REINFORCES the query-side signal at init rather than cancelling it**. Forward losses bit-identical (diff 0.0). SG direct-vs-two-pass equivalence holds (worst rel 4.47e-05); SG 128-example rehearsal clean (42 families).

**Paired result (800 updates each; FULL ran first and exactly reproduced D4-C2 at all four probes across five metric families — zero mismatches — before SG proceeded):**
- **FULL:** train FRA probe loss 1.1391, mean alignment −0.0052 → `FRA_SELF_OPTIMIZATION_FAILED` (D4-C2 replication).
- **SG:** train FRA probe loss 1.1150, mean alignment −0.0007 → `FRA_SELF_OPTIMIZATION_FAILED`. Pre-clip grad norms 1.35 (u=1) → 3.43 (u=800) — above the frozen clip of 1.0, so the applied gradient is ~unit-norm; removing the (reinforcing) prototype component roughly halved the raw gradient, exactly as the decomposition predicted, without changing the outcome.
- **New observables (both arms):** cosine-logit spread DOES widen from 0.171 at init to ~0.29–0.45 mid-run (correcting the earlier wording point — spread widened; it just never became label-aligned), while the target margin stays negative throughout (FULL −0.133, SG −0.109 at T800) and residual norms remain 0.04–0.08 against state norms of order 10.

**Per the frozen interpretation: `COUPLING_NOT_BOTTLENECK` — prototype-gradient coupling is not the bottleneck; the next target per the release is a persistent/fixed cross-batch target formulation.** The observables sharpen the picture: training widens logit spread without producing positive target margins or cross-family alignment — the residuals are learning *some* consistent structure that is not the label geometry. Q2 remains UNQUALIFIED; Q3, corpus generation, v0.6 execution, and all other perturbations remain **held** pending Commit X review.


## D4-C3 fixed-temperature FRA (τ=0.25) — COMPLETE: `FRA_SELF_OPTIMIZATION_FAILED` (lineage: code `211d163` = Commit U → evidence in this commit)

**Preflight (from literally clean U, output under gitignored `local_data` until this commit): PASS.** τ=0.25 direct-vs-VJP equivalence holds (worst rel 4.06e-05, min cosine 0.99999988, loss diff exactly 0). The frozen-batch τ comparison quantified what temperature actually changes at init: VJP norm amplification ×4.09 (largely normalized by the frozen grad clip — pre-clip norms 11.0 at u=1) with only mild rotation of the gradient direction (cosine 0.9946 between τ=1 and τ=0.25 VJPs). Tempered rehearsal: 42 families, all finite.

**D4-C3 result (single changed factor vs D4-C2: `auxiliary_logits = cosine_logits / 0.25`; 800 updates; canonical untempered probe loss preserved for endpoint comparability):** training tempered FRA loss fluctuates in the ln(3) band (1.16 → 1.24 at u=800; untempered 1.107 → 1.116); the VJP norm explodes (3.1 → 51.9 → 226.9 → 124.2) while the clipped parameter gradient stays bounded (~6–11) and the geometry never moves — train mean alignment −0.003 → +0.005 at T800; canonical train FRA probe loss **1.0742**; eval exactly 0.333333. Tempered and untempered probe losses converge to near-identity (1.0741 vs 1.0742): the cosine-logit spread never widened, i.e. sharpening did not extract more usable signal from these states.

**Endpoint per the frozen criteria (canonical untempered values): `FRA_SELF_OPTIMIZATION_FAILED` — τ=0.25 is insufficient.** Per the release, this rules out only τ=0.25 (not all temperatures); the next target, if pursued, is prototype-gradient coupling / objective formulation rather than λ, LR, optimizer, or architecture. The standing horizon caveat applies (800 updates). Q2 remains UNQUALIFIED; Q3, corpus generation, and v0.6 execution remain **held** pending Commit V review.


## D4-C2 effective-batch prototype support — COMPLETE: `FRA_SELF_OPTIMIZATION_FAILED` (lineage: code `aa7a62a` = Commit S → evidence in this commit)

**Preflight (from literally clean S, output under gitignored `local_data` until this commit): PASS.** The two-pass exact-VJP construction was validated before the arm: on a real 16-example microbatch, direct FRA-backward and proxy/VJP parameter gradients from identically initialized C0 models agree to worst relative L2 difference 4.1e-05 and minimum cosine 0.99999988 (float32 accumulation-order noise; tolerance recalibrated from absolute 1e-6 to relative 1e-4 / cosine 0.999999 with the measured rationale recorded). The working-tree rehearsal of the new preflight caught two pre-commit defects (None-grad comparison crash; miscalibrated tolerance) before Commit S existed. One real 128-example D4-C2 update rehearsed end-to-end: 42 complete families + 2 fragment examples, nonzero finite VJP (0.75), finite step/params.

**D4-C2 result (one C0 arm, 800 updates, L = L_FRA only, prototypes from ALL 42 complete families per 128-example effective batch via two-pass exact VJP; everything else identical to D4-C1):** training full-batch FRA loss never leaves the ln(3) band (1.107 → 1.101 → 1.119 → 1.119 at u=800); VJP norm grows (0.75 → 9.3) while the parameter grad norm stays ~2-3 — the objective pushes without producing coherent geometry; train mean alignment stays at baseline (−0.003 → +0.012 → −0.005 at T800); train FRA probe loss 1.139; unseen FRA 1.126; eval_ID exactly 0.333333 at every probe.

**Endpoint per the frozen criteria (unchanged from D4-C1): `FRA_SELF_OPTIMIZATION_FAILED`** — mean alignment −0.0052 < 0.25 AND train FRA loss 1.1391 ≥ 1.00. **Per the D4-C2 release rule, prototype-support noise is disfavored as the explanation; the next discriminator isolates temperature/scale of the cosine objective** rather than changing λ, architecture, or optimizer. Q2 remains UNQUALIFIED; Q3, corpus generation, v0.6 execution, and all other perturbations remain **held** pending Commit T review.


## D4-C1 FRA-only self-optimization — COMPLETE: `FRA_SELF_OPTIMIZATION_FAILED` (lineage: code `8aadbdc` = Commit Q2 → evidence in this commit)

**Incident d4c1_preflight_incident_001 (`DIAGNOSTIC_PREFLIGHT_OUTPUT_INVALID`)**: Commit Q (`10731d6...`) omitted the one-line `D4C1_PREFLIGHT_OUT` constant; the first preflight invocation crashed at the manifest-write step after the FRA-only rehearsal itself had completed (no manifest emitted, 800-update arm never started, no evidence). Remediated per the authority's release as Commit Q2 (`8aadbdced328721a008f63e5ff6658c92fd0232d`): constant added, working-tree preflight executed end-to-end (PASS, artifact discarded), incident JSON binds the crash log by SHA-256. **Authoritative preflight from literally-clean Q2: PASS**, bound to the full Q2 SHA with `working_tree_clean_at_start = true`; preflight output remained under gitignored `local_data` until this evidence commit.

**D4-C1 result (one C0 arm, 800 updates, L = L_FRA only, FC stream, otherwise unchanged recipe; telemetry now records means over all 8 accumulation microbatches):**
- Training microbatch FRA loss never leaves the ln(3) band: 1.106 (u=1) → 1.081 (u=100) → 1.079 (u=400) → 1.091 (u=800) — the objective does not optimize itself under the exact 4–5-family microbatch regime (mean 4.667 families/microbatch across 6,400 microbatches; min 4, never violating the ≥3 rule).
- Train-probe FRA loss: 1.117 → 1.132; train mean alignment over {E−C, E−U, C−U}: −0.0028 → **−0.0051** (never leaves the T0 baseline); unseen FRA 1.089–1.173; eval_ID exactly 0.333333 at every probe.
- Endpoint per the frozen criteria: mean alignment −0.0051 < 0.25 AND train FRA loss 1.132 ≥ 1.00 → **`FRA_SELF_OPTIMIZATION_FAILED`**.

**Per the D4-C1 release rule, λ tuning is explicitly NOT the next step; prototype support / cosine-temperature mechanics become the leading target** — with FRA unable to self-optimize even alone, the failure localizes to the objective's mechanics at this batch scale (leave-one-out prototypes over 4–5 families are noise-dominated targets; unit-normalized cosine logits without temperature give near-zero gradient in the symmetric basin) rather than to task/FRA gradient interference. Q2 remains UNQUALIFIED; Q3, corpus generation, v0.6 execution, and all λ/temperature/prototype-support/architecture/optimizer/LR perturbations remain **held** pending Commit R review.


## D4-C Family-Residual Alignment — COMPLETE after remediation: `FRA_OBJECTIVE_FAILED` (lineage: code `531c593` = Commit O2 → evidence in this commit)

**Incident d4c_launch_incident_001 (DIAGNOSTIC_INVALID_PRE_UPDATE)**: the first pair attempt from defective Commit O (`dc5a9b6...`) crashed at FC-FRA update 0 on a nested-tuple return in `_fra_loss_microbatch`; zero optimizer steps, no evidence; classified and remediated per the authority's release. Commit O2 (`531c5930d97530d9b2d20502b8e320bea1106fda`) contains only the flattening fix with a scalar-finite-tensor assertion, a shared `_fc_fra_forward` used verbatim by loop and preflight, and `run_d4c_preflight` (real FC-FRA branch, one effective update, fresh C0, discarded). A second external interruption (task killed mid-FC-CE ~T1600, no evidence) delayed the rerun; the aborted log is bound by SHA-256 in the incident JSON. The completed FC-CE arm from the aborted pair was NOT accepted as evidence per the release.

**Preflight from clean O2: PASS** (all checks; captured state autograd-attached; 4–5 complete families per microbatch across all 16,000 FC-FRA microbatches; losses/backward/grads/step/params finite).

**Paired result (2,000 updates each, frozen config λ=1.0, ε=1e-6, no temperature; probe digests `a870bfb5...` (D3 train) and `2288b187...` (unseen) recorded):**
- **FC-CE control:** train SMA 0.3335 → 0.6694, eval chance — third exact replication of the known FC trajectory; its probe-FRA(train) fell 1.117 → 0.680 and train-probe alignment rose to 0.3008 (memorization creates partial within-train residual alignment without any unseen transfer: FRA(unseen) 1.09–1.19, unseen alignment 0.0294).
- **FC-FRA:** the auxiliary objective **suppressed memorization without achieving alignment** — train SMA flat 0.3333, task CE pinned at ln(3), training microbatch FRA loss only drifted 1.134 → ~0.95–1.06 (vs ≈0.37 achievable with aligned residuals), probe FRA(train) 1.1475 ≈ ln(3), train-probe alignment **−0.0101** (below FC-CE's 0.3008), unseen alignment −0.0152, both eval surfaces exactly chance.

**Lineage caveat (flagged for authority disposition):** the pair manifest records `working_tree_clean_at_start = false` because the protocol-generated, O2-bound preflight JSON was present untracked when the pair launched; `code_git_commit = 531c593...` is exact and no uncommitted code existed. Per the preregistered D4-C table, `FRA_OBJECTIVE_FAILED` mandates **diagnosis of the FRA implementation/objective itself before architecture conclusions** — candidate factors on the record: leave-one-out prototypes averaged over only 4–5 families per microbatch (high-variance targets), λ=1.0 auxiliary gradient interacting with the task signal inside the symmetric basin, and unit-normalized cosine logits without temperature. Q2 remains UNQUALIFIED; all perturbations, Q3, corpus, and v0.6 execution remain **held**.


## D4-B3 / FC-8K full-horizon family-coherent extension — COMPLETE: `FIT_NO_TRANSFER` (two-commit lineage: code `e290fe7` → evidence below)

Single FC arm (8,000 updates, full 24k corpus, wall 1.81 h) from clean Commit M (`e290fe7197cdd2fedea2f16540b3633adacc4b58`; `parameter_count = 53,232,643`, clean tree at start, frozen Q1 digests, probe digest, full record in `diagnostics/d4b3_fc_8k.json`). The fail-closed prefix gate at T2000 **passed**: all 30 comparisons (6 probes × train-SMA / train-CE / eval-ID-SMA / eval-STRUCT-SMA / unseen-family E−C alignment) exactly reproduced the D4-B FC arm; first-2,000-batch SHA-256 `110a615b62dd3fa9...` recorded.

**Endpoint: `FIT_NO_TRANSFER`** (both eval ≤ 0.36, train ≥ 0.90):
- train_ID SMA rose monotonically 0.3335 → 0.6694 (T2000) → 0.9384 (T4000) → 0.9998 (T8000); train CE 1.358 → 0.0020 — near-perfect memorization of all 8,000 training families.
- eval_ID SMA stayed 0.331–0.339 at every one of 21 probes (final 0.334667); eval_STRUCT final 0.335667; diagnostic Q = 0.3352; Q₂:₄ 0.3338/0.3369; min label recall 0.326/0.281.
- Unseen-family E−C alignment never left its T0 baseline (final −0.0132) — the D3-style secondary mechanism check shows the memorized representation remains family-specific at the full horizon.

**Reading (per the preregistered endpoint table):** family-coherent batching at the full production horizon produces complete training-corpus memorization with zero transfer to held-out surfaces or unseen families. Per the D4-B3 release, this does not reopen v0.5 Q2 (which remains UNQUALIFIED); it is diagnostic evidence for any v0.6 recipe proposal — the student can fit the corpus under co-location but extracts no generalizing relational rule under the frozen recipe. D4-C, Q3, corpus generation, new qualification, and all other perturbations remain **held** pending project-authority review of Commit N.


## D4-B2 matched-permutation family-disjoint control — COMPLETE: `COLOCATION_CAUSES_ENGAGEMENT_CONFIRMED` (two-commit lineage: code `95dbeaa` → evidence below)

Single LB-MP arm (2,000 updates, full 24k corpus) from clean Commit K (`95dbeaa5b81d048cf71a081402af35cf51ebc3b2`; `parameter_count = 53,232,643`, clean tree at start, frozen Q1 digests, runtime digest, full lineage in `diagnostics/d4b2_matched_permutation_control.json`). LB-MP derives its family permutation from the **exact FC namespace/seed/epoch rule** and fills E/C/U slots at offsets 0/2667/5334 — removing co-location while holding the family presentation permutation, label sequence, and 43/43/42 balance identical to FC. The fail-closed pre-training verification passed every check: permutation digests identical to FC across all 12 epochs consumed, E-slot family k = FC family k for all k, epoch coverage exact, label sequence identical across all 2,000 batches, and **max family multiplicity 1 in every batch including all epoch-boundary crossings** (zero violations; pre-verified deterministically for this exact horizon).

**Result: LB-MP stays pinned at the uniform floor** — train CE 1.108 → 1.1005 (ln 3) at every probe, train SMA flat (0.3332), eval_ID 0.334 / eval_STRUCT 0.337 (chance), unseen-family alignment −0.003. Against the D4-B FC reference (train CE 0.761, train SMA 0.669), the permutation-order confound is ruled out: with the FC permutation held exactly constant, removing family co-location alone removes the optimization engagement.

**Per issue #3's D4-B2 table: `COLOCATION_CAUSES_ENGAGEMENT_CONFIRMED`** — the co-location mechanism is now causally isolated rather than merely correlated. The 8,000-update FC extension becomes eligible upon project-authority review of this checkpoint. D4-C and all other factors remain **held**.


## D4-B label-balanced family-disjoint decomposition — COMPLETE: `COLOCATION_CAUSES_ENGAGEMENT` (two-commit lineage: code `b1d33d3` → evidence below)

Paired 2,000-update FC/LB arms on the full 24k corpus from clean Commit I (`b1d33d30ac35f7e3f12dee9018392618d3949475`; `parameter_count = 53,232,643`, clean tree at start, frozen Q1 digests, runtime digest, family-stream config digest `3a65c503...`, eval-probe family digest identical to D4-A's `2288b187...`, per-arm wall time FC 28.3 min / LB 26.5 min). LB was verified pre-run to have the **identical global E/C/U label sequence and identical 43/43/42 per-128 balance cycle as FC** (max family multiplicity 1 per batch vs FC's 3; every example consumed exactly once per epoch). Per-probe metrics now include **full train_ID surfaces** (all in `diagnostics/d4b_label_balanced_disjoint.json`):

- **FC: engaged and fitting the training families** — train SMA 0.3335 → **0.6694**, train CE 1.358 → **0.761**, mixed train histogram [7046, 10544, 6410]; eval_ID 0.332 / eval_STRUCT 0.331 remain chance; unseen-family alignment stays at baseline (0.0294). FC scalar-reproduces the D4-A FC arm at all six probes.
- **LB: not engaged** — train SMA flat (0.3343), train CE pinned at ln(3) ≈ 1.099–1.109 through T2000, near-constant train histogram [720, 6535, 16745]; eval chance; unseen-family alignment 0.0107.

**Per issue #3's D4-B decision table: `COLOCATION_CAUSES_ENGAGEMENT`** — with label sequence and per-batch balance held exactly constant, removing family co-location alone removes the optimization engagement FC showed. Label balance/order does not explain the D4-A CE reduction; local counterfactual-family co-presence is the operative factor. FC nevertheless still transfers nothing within 2,000 updates (eval and unseen-family geometry unchanged), so the 8,000-update FC extension question — whether co-location-driven train fitting eventually generalizes — remains open and is the project authority's call. All other D4 factors, Q3, corpus generation, and new qualification runs remain **held**.


## D4-A family-coherent batching falsifier — COMPLETE: `BOTH_REMAIN_CHANCE` (two-commit lineage: code `37f632f` → evidence below)

Paired 2,000-update C0 arms on the full 24k corpus from clean Commit G (`37f632f8abe9acc828fc60c57582c7db71ed1d67`; `parameter_count = 53,232,643`, clean tree at start, frozen Q1 digests, runtime digest, family-stream config digest, eval-probe family digest, per-arm wall time all recorded in `diagnostics/d4a_family_coherent_batching.json`):

- **R (production example-level stream): eval_ID SMA = 0.333333 at every probe** (T0–T2000), eval_STRUCT 0.333333, constant-label histograms, min-recall 0 — and **bit-matches the production C0 seed-1647674144 eval history at all five probe updates**, the strongest internal control available (same code, same trajectory, repaired provenance).
- **FC (family-unit stream, E/C/U contiguous, 42 complete families + 2-example fragment per 128-batch): eval_ID SMA = 0.3320 at T2000** (dips to 0.327–0.330 are noise), eval_STRUCT 0.3308 — chance, though with mixed prediction histograms and nonzero min-recall 0.2325.
- **The mechanism was engaged but did not transfer:** FC's batch CE falls monotonically 1.119 → 1.101 → 1.053 → 0.919 → **0.864** (clearly below the ln(3)=1.0986 uniform floor that R never leaves), while unseen-family E−C displacement alignment stays at its T0 baseline (R: 0.0017, FC: 0.0294 vs T0 −0.004).

**Per issue #3's decision table: family co-location is insufficient — stop before changing another factor.** The nuance for the record: within the 2,000-update horizon, family-coherent batching does materially change the optimization (loss escapes the uniform floor; predictions de-degenerate) yet produces zero generalization to unseen families or eval surfaces. Whether a longer FC horizon would eventually generalize is untested and is the authority's call; the falsifier as specified is answered. All other D4 factors, Q3, corpus generation, and new qualification runs remain **held**.


## D3 representation probe — COMPLETE: `TRAIN_ALIGNED_HOLDOUT_CHANCE` (two-commit lineage: code `2f50cd1` → evidence below)

D3 (C0, seed 1647674144, D1's exact 24-family train set + disjoint 24-family holdout = next 6 sha256-ranked complete families per depth) probed the post-final-RMSNorm 512-d `<DECIDE>` state at T0/T100/T2000 with clean code→evidence lineage (`code_git_commit = 2f50cd1...`, tree clean at start, frozen Q1 digests, runtime snapshot digest, train/holdout family-list digests all recorded in `diagnostics/d3_representation.json`):

- **TRAIN becomes strongly aligned**: E−C displacement mean pairwise cosine 0.0033 (T0) → 0.7393 (T100) → **0.9914 (T2000)**; train residual-centroid classification 1.0000 from T100 onward.
- **HOLDOUT stays at its T0 baseline**: E−C displacement mean cosine 0.0093 (T0) → 0.0037 (T100) → **−0.0165 (T2000)**; holdout residual-centroid classification 0.3056–0.3889 ≈ chance; within-family holdout states remain nearly identical (cos 0.965) while train within-family states separate (cos 0.360, L2 26 vs 3).
- **Production classifier**: train accuracy 1.0000 / CE 0.0000 (clean-lineage replication of D1 — identical selection, seed, stream, production LR values) while holdout accuracy 0.333–0.347 with near-constant-label histograms and min-recall 0.
- Cross-set (train↔holdout) displacement cosine never exceeds 0.13.

**Per issue #3's decision table this is the "family-specific memorization" pattern → architecture/representation/task structure becomes the leading mechanism**; scale-optimization is disfavored as the sole cause (the same budget memorizes and aligns perfectly on 24 families but produces zero cross-family transfer). No measurement contradiction occurred (train geometry and train classifier agree). D4 single-factor perturbations remain **held** pending project-authority inspection of this geometry.


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
