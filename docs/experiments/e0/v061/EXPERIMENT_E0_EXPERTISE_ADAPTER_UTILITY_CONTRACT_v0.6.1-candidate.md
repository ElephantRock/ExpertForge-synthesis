# ExpertForge Synthesis Experiment E0 — Expertise Adapter Utility Contract v0.6.1 Candidate

**Document class:** Authoritative experiment-contract candidate  
**Project:** ExpertForge Synthesis  
**Experiment ID:** E0  
**Contract version:** 0.6.1-candidate  
**Prepared:** 2026-09-11  
**Status:** **CANDIDATE — mechanism-selection scientific rules revised for audit; no bootstrap, mechanism scoring, qualification, Q3, final-corpus, adapter, or terminal-execution authority is granted**  
**Supersedes:** `EXPERIMENT_E0_EXPERTISE_ADAPTER_UTILITY_CONTRACT_v0.6-draft.md` for the mechanism-selection stage. It does **not** reopen or supersede the historical v0.5 Q2 result.  
**Normative inherited source:** `EXPERIMENT_E0_EXPERTISE_ADAPTER_UTILITY_CONTRACT_v0.5-candidate.md`, SHA-256 `25c829a63b7e6cf17d43c8b09b0fa6cd1cf0d739dac7fee8f0042a0a278e3dc6`.  
**Superseded v0.6 draft SHA-256:** `6d3bd396dcf60d216b81a91d750d1f6d7d0749efe8a79dc055ca1a0d046bd651`.

**Purpose:** Repair the failed v0.5 student prerequisite without reopening v0.5 Q2. Before adapter utility can be tested, v0.6 must establish a trainable student regime that transfers across held-out CMDR families and structural surfaces under a frozen, auditable resource envelope.

---

# 0. Immutable Historical Boundary

The following historical result is immutable:

```text
v0.5 Q2 = CLOSED / UNQUALIFIED
selected M0 = null
adapter utility was never tested
Q3 remained held
final 108k corpus was not authorized
```

No v0.6 result retroactively changes, reinterprets, or reopens v0.5 Q2.

All D0–D4 diagnostics remain:

```yaml
diagnostic_only: true
non_scientific: true
```

The strongest diagnostic result motivating this revision remains:

```text
D4-C5E = PERSISTENT_TARGET_TRAIN_ONLY
```

That result establishes only that the tested random-initialized C0 regime can learn a persistent label-aligned representation on training families while failing the preregistered unseen-family transfer gate at the terminal horizon. It does **not** establish that all architectures, corpus allocations, tokenizers, or pretrained regimes fail to generalize.

---

# 1. v0.6.1 Scientific Question

The E0 terminal causal question is unchanged:

> Does a source-specific learned Expertise Adapter transfer useful CMDR capability to a standalone student beyond simpler source supervision and ordinary latent-state distillation, under a bounded reproducible resource envelope?

v0.6.1 inserts a prerequisite mechanism-selection stage with two operational questions:

1. **Diversity–repetition allocation:**  
   > Does reallocating a fixed example-presentation budget from repeated exposure toward greater unique counterfactual-family diversity improve unseen-family CMDR transfer?

2. **Externally pretrained treatment:**  
   > Under matched visible CMDR text and matched example-presentation budget, does a compact externally pretrained P0 treatment produce transferable behavior that is unavailable to the tested random-initialized regimes?

The first question does **not** isolate “diversity” from “repetition.” Under a fixed 1,024,000-presentation budget, increasing the number of unique families necessarily reduces average repeated exposure per family/example. The causal treatment is therefore the **diversity–repetition allocation**, not unique-family diversity alone.

The second question is intentionally **not** phrased as “representation prior versus data exposure.” Pretrained weights jointly encode external data exposure, optimization history, and learned representations. v0.6.1 may identify a **pretraining-treatment effect within a frozen architecture/tokenizer/training recipe**, but it may not rename that result as proof of an isolated representation prior.

---

# 2. Governing Epistemic Rules

The v0.5 governing rule remains binding:

```text
observation ≠ adoption
adoption ≠ implementation
implementation ≠ verification
verification ≠ generalization
```

v0.6.1 adds:

```text
training fit ≠ unseen-family transfer
fixed presentations ≠ isolated diversity effect
pretraining treatment ≠ isolated representation prior
same visible string ≠ same token sequence
shared vocabulary ≠ split leakage
linear-readout success ≠ zero-shot capability
mechanism-selection PASS ≠ downstream E0 eligibility
O(1) key derivation ≠ O(1) storage latency
```

Architecture changes, Engram/HCM, optimizer rescue, learning-rate sweeps, FRA variants, and other architecture interventions are outside this mechanism-selection stage.

---

# 3. Binding Classes and Lifecycle — v0.6.1 SUPERSESSION

The binding classes remain:

```text
SPEC-BOUND
QUAL-BOUND
EXECUTION-BOUND
```

The v0.6.1 lifecycle is:

```text
CANDIDATE
  ↓
mechanism-selection specification fixed
  ↓
finite pretrained-candidate registry frozen
  ↓
V06_MECHANISM_BOOTSTRAP_AUTHORIZED
  ↓
generator / verifier / structural-space audit / runtime preflights only
  ↓
selection corpus identities + candidate snapshot + metric/sampler/decision digests frozen
  ↓
MECHANISM_SELECTION_EXECUTION_RELEASED
  ↓
primary evidence cells:
    R1 / R4 / R16
    P-FROZEN
    P-RANDOM@R1
    P-FT@R1
  ↓
conditional evidence:
    P-RANDOM-FROZEN iff P-FROZEN reaches TRANSFER_PASS
    P-FT@R16 iff R1,R4,R16,P-RANDOM@R1 are all NO_TRANSFER and P-FT@R1 is NO_TRANSFER
  ↓
if ≥2 trainable regimes are eligible:
    six preregistered comparison-only seeds execute for those eligible regimes
  ↓
mechanism-selection closure
  ↓
candidate regime set frozen
  ↓
NEW V0.6 STUDENT QUALIFICATION on a fresh corpus
  ↓
exact M0-v0.6 selected or null
  ↓
source/headroom qualification + remaining pre-freeze bindings
  ↓
FROZEN
  ↓
AUTHORIZE_E0_IMPLEMENTATION
  ↓
EXECUTION_RELEASED
  ↓
E0 adapter evidence
```

A mechanism-selection PASS does not itself authorize student qualification, source work, Q3, final-corpus generation, adapter training, architecture experiments, or terminal E0 evidence.

---

# 4. Mechanism-Selection Corpus — `CMDR-MSEL-v0`

## 4.1 Corpus topology

`CMDR-MSEL-v0` is new and selection-only.

```yaml
CMDR-MSEL-v0:
  train_pool:
    complete_families: 128000
    examples: 384000
    families_per_depth: 32000
    examples_per_label_depth_cell: 32000

  dev_ID:
    complete_families: 2000
    examples: 6000
    families_per_depth: 500

  eval_ID:
    complete_families: 2000
    examples: 6000
    families_per_depth: 500

  eval_STRUCT:
    complete_families: 2000
    examples: 6000
    families_per_depth: 500

  total_examples: 402000
```

Every family contains exactly one example for each semantic label:

```text
ENTAILED
CONTRADICTED
UNKNOWN
```

Incomplete families are rejected.

## 4.2 Nested diversity–repetition ladder

Family membership is stratified by reasoning depth before ranking.

Within each depth `d ∈ {1,2,3,4}`, sort families by ascending 256-bit integer value of:

```text
SHA256("ExpertForge-E0-v061-msel-rank|" || family_id)
```

with lexicographic `family_id` as a deterministic collision fallback.

For each depth, take the first:

```text
R1:   2,000 families/depth  =   8,000 total families =  24,000 examples
R4:   8,000 families/depth  =  32,000 total families =  96,000 examples
R16: 32,000 families/depth  = 128,000 total families = 384,000 examples
```

Thus:

```text
R1 ⊂ R4 ⊂ R16
```

Under the frozen 1,024,000-presentation budget, approximate average presentations per unique example are:

```text
R1  ≈ 42.67
R4  ≈ 10.67
R16 ≈  2.67
```

Therefore downstream claim language must use **diversity–repetition allocation** or **in-domain allocation**, not “diversity alone caused transfer.”

## 4.3 Structural leakage semantics — `CMDR-StructSig-v1`

A family is converted to a canonical structural signature by:

1. retaining fact/rule/query graph topology, polarity, arity, direction, reasoning-depth stratum, and distractor connectivity;
2. replacing entity and predicate identifiers with deterministic first-occurrence canonical symbols;
3. removing lexical surface choices, slot-order nuisance, whitespace, and generator seed identity;
4. sorting semantically unordered fact/rule sets under the canonical symbol map;
5. serializing canonical UTF-8 JSON and hashing it with SHA-256.

Required split audits:

```text
family_id overlap:
    pairwise 0 across train_pool / dev_ID / eval_ID / eval_STRUCT

exact rendered-string overlap:
    pairwise 0 across all four splits

CMDR-StructSig-v1:
    eval_STRUCT overlap with train_pool = 0
    eval_STRUCT overlap with dev_ID     = 0
    eval_STRUCT overlap with eval_ID    = 0
```

`dev_ID` and `eval_ID` are family-disjoint in-distribution surfaces. They may share canonical structural signatures with `train_pool`; this is intentional. `eval_STRUCT` is the structural-signature-disjoint surface.

Shared tokenizer vocabulary, lexical atoms, and logical primitives do not count as leakage by themselves.

## 4.4 Structural-support/depletion preflight

Before corpus freeze, a blocking report must provide, by depth:

```text
attempted families
accepted complete families
unique StructSig count
StructSig multiplicity distribution
family rejection rate
StructSig collision/rejection rate
number of signatures consumed by eval_STRUCT exclusion
estimated remaining admissible signature support after burning CMDR-MSEL-v0
projected rejection rate for a fresh v0.6 qualification corpus
```

If the validator cannot demonstrate sufficient remaining support for a future family- and structure-separated qualification corpus without a material generator distribution shift, the selection corpus is not frozen.

## 4.5 Generator namespace and burn namespace

All mechanism-selection generator seeds must be derived under the reserved namespace root:

```text
ExpertForge-E0-v061-msel
```

That entire namespace is permanently reserved after selection closure and may never be reused for future qualification or E0 corpora.

## 4.6 Generator and shortcut gates

The independent proof verifier and nuisance/lexical shortcut audits remain mandatory.

Required shortcut gate:

```text
Q_shortcut ≤ 0.38
```

for every required shortcut model/evaluation surface.

---

# 5. Random-Init C0 Regimes — `R1/R4/R16`

## 5.1 Architecture

All R arms use the exact v0.5 C0 implementation:

```text
M0-CausalDense-v1 / C0
15 layers
width 512
8 query / 8 key-value heads
SwiGLU FFN width 1536
final RMSNorm
linear 512→3 task classifier
53,232,643 parameters
CMDR-Lex-v1 tokenizer
random initialization as v0.5
```

The exact implementation/config digest becomes QUAL-BOUND before execution.

No C1/C2/C3 ladder is reopened.

## 5.2 Common training recipe

```yaml
optimizer: AdamW
beta1: 0.9
beta2: 0.95
epsilon: 1.0e-8
peak_lr: 5.0e-4
weight_decay: 0.05
weight_decay_scope: exclude_norm_bias
grad_clip_global_norm: 1.0

updates: 8000
effective_batch_examples: 128
microbatch_examples: 16
gradient_accumulation_steps: 8
example_presentations: 1024000

warmup_updates: 400
schedule: linear_warmup_then_cosine
final_lr_fraction: 0.10
dropout: 0
label_smoothing: 0
activation_checkpointing: false
student_parameter_offload: false
student_optimizer_offload: false
```

Training always executes all 8,000 updates unless the attempt becomes invalid or a scientifically valid divergence occurs. `early_stopping = false` means training is never terminated because a dev metric peaks.

Checkpoint candidates are updates:

```text
400, 800, 1200, ..., 8000
```

Checkpoint selection is:

```text
argmax dev_ID CMDR-SMA
tie-break = earliest update
```

`eval_ID` and `eval_STRUCT` may not influence checkpoint selection, hyperparameters, run continuation, or architecture choice.

## 5.3 Frozen training stream — `MSEL-ExampleStream-v1`

For an arm with frozen example set `S`, master seed `s`, and cycle `c = 0,1,2,...`, rank every sample independently by the unsigned 256-bit value:

```text
SHA256(
  "ExpertForge-E0-v061-msel-stream|"
  || decimal(s) || "|"
  || decimal(c) || "|"
  || sample_id
)
```

Sort all samples in `S` by that rank, with lexicographic `sample_id` as collision fallback.

The training stream is the concatenation of complete cycle permutations. Consume exactly 1,024,000 examples:

```text
- no replacement within a cycle
- cycles may repeat examples
- logical batches may cross cycle boundaries
- no short batches
- no dropped examples
- final cycle is truncated exactly at the 1,024,000th presentation
```

The same ranking function is used for every regime. Common samples therefore preserve their pairwise relative order within a cycle, although their absolute positions differ when arm membership differs.

No family-coherent batching is introduced in this stage.

---

# 6. Externally Pretrained Treatment — `P0-70M`

## 6.1 Finite candidate registry

The sole candidate is:

```yaml
candidate_id: P0-70M
upstream_provider: EleutherAI
upstream_id: EleutherAI/pythia-70m
upstream_revision: a39f36b100fe8a5377810d56c3f4789b9c53ac42
weights_file: model.safetensors
weights_sha256_observed: ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15
license_observed: Apache-2.0
model_class: GPT-NeoX causal language model
pretraining_class: base pretrained, not instruction-tuned
hidden_size: 512
layers: 6
attention_heads: 8
ffn_width: 2048
native_context: 2048
native_vocab_size: 50304
```

The proposed revision string is 40 hexadecimal characters and resolves on the upstream repository. The observed safetensors SHA-256 is published by the upstream file pointer. These observations are **not** a substitute for the bootstrap-time independent snapshot/digest verification.

For CMDR classification:

```text
retain token embeddings
retain transformer backbone
retain final normalization
remove upstream LM head
add bias-enabled linear 512→3 task classifier
```

Expected counts:

```text
backbone without LM head: 44,670,976
task classifier:               1,539
total classification model: 44,672,515
```

The candidate is chosen to be in the same order of magnitude as C0, not to be parameter-matched.

Before registry freeze, independently compute and bind:

```text
exact upstream commit/snapshot identity
config.json SHA-256
tokenizer.json SHA-256
tokenizer_config.json SHA-256
special_tokens_map.json SHA-256
model.safetensors SHA-256
snapshot-manifest SHA-256
license/provenance record
exact instantiated parameter count
```

The registry may not grow or substitute candidates after any `CMDR-MSEL-v0` model score is observed.

## 6.2 Tokenization declaration

P0 retains its native tokenizer. Tokenizer behavior is part of the treatment.

“All models see the same canonical CMDR visible text” means **the same canonical UTF-8 rendered string**, not the same token sequence.

The complete selection corpus must be audited under both tokenizer families:

```text
R arms:
    CMDR-Lex-v1
P arms:
    exact frozen P0 native tokenizer
```

Required audit fields:

```text
min / median / p95 / p99 / max token length
special tokens added
invalid/unknown encoding count
truncation count
```

For P0:

```text
max sequence length ≤ 384 native tokens
truncation = 0
invalid encoding = 0
```

Tokenizer remapping is prohibited.

If P0 violates the interface bound, it is `INTERFACE_INELIGIBLE`; the pretrained branch does not substitute a new candidate without a contract revision.

## 6.3 Common P-arm batching/checkpoint semantics

Unless explicitly stated otherwise, all trainable P arms use:

```text
effective batch = 128
microbatch = 16
gradient accumulation = 8
updates = 8000
presentations = 1,024,000
checkpoint candidates = every 400 updates
selection metric = dev_ID CMDR-SMA only
tie-break = earliest update
eval_ID/eval_STRUCT influence on selection = prohibited
early termination on dev peak = prohibited
```

## 6.4 `P-FROZEN` — pretrained frozen-backbone linear-readout probe

```text
P0 backbone: frozen
readout state: final hidden state at terminal prompt token after answer cue
trainable parameters: new 512→3 classifier only
train corpus: R1
loss: mean 3-way cross entropy
updates: 8000
presentations: 1,024,000
```

Optimizer:

```yaml
algorithm: AdamW
lr: 1.0e-3
beta1: 0.9
beta2: 0.95
epsilon: 1.0e-8
weight_decay: 0.0
grad_clip_global_norm: 1.0
schedule: constant
warmup_updates: 0
```

Approved interpretation:

```text
P-FROZEN TRANSFER_PASS
→ a linear readout trained on R1 achieves transfer from the frozen pretrained P0 representation
```

This is not a zero-shot claim and does not prove an intrinsic or isolated “representation prior.”

## 6.5 `P-RANDOM-FROZEN` — conditional frozen-random probe

Execute only if `P-FROZEN = TRANSFER_PASS`.

It uses:

```text
exact P0 architecture
exact P0 tokenizer
random backbone initialization from the frozen P0 config
backbone frozen for the entire run
same linear task head
same R1 corpus
same six primary seeds
same stream/checkpoint/readout protocol
same P-FROZEN optimizer
```

Interpretation:

```text
P-FROZEN pass + P-RANDOM-FROZEN NO_TRANSFER
    → PRETRAINED_FIXED_FEATURES_ADD_VALUE_WITHIN_MATCHED_PROBE

P-FROZEN pass + P-RANDOM-FROZEN TRANSFER_PASS
    → RANDOM_FIXED_FEATURES_SUFFICIENT_WITHIN_MATCHED_PROBE

P-RANDOM-FROZEN inconclusive
    → FROZEN_FEATURE_ATTRIBUTION_INCONCLUSIVE
```

This probe is never eligible as the final trainable E0 student.

## 6.6 `P-FT@R1` — pretrained full-finetune regime

```text
initial backbone: exact P0 pretrained weights
trainable: retained backbone + task classifier
train corpus: R1
```

Optimizer:

```yaml
algorithm: AdamW
beta1: 0.9
beta2: 0.95
epsilon: 1.0e-8
peak_lr: 5.0e-5
weight_decay: 0.01
weight_decay_scope: matrix_weights_only_no_norm_or_bias
grad_clip_global_norm: 1.0
warmup_updates: 400
schedule: linear_warmup_then_cosine
final_lr_fraction: 0.10
```

No learning-rate or weight-decay sweep is permitted.

## 6.7 `P-RANDOM@R1` — unconditional matched architecture/tokenizer control

`P-RANDOM@R1` executes unconditionally with the six primary seeds.

It uses:

```text
exact P0 architecture
exact P0 tokenizer
random backbone initialization from the frozen P0 config
same trainable parameter set as P-FT@R1
same task classifier interface
same R1 corpus
same P-FT optimizer and schedule
same master seeds
same MSEL-ExampleStream-v1
```

The intended changed factor is the initial backbone state:

```text
P-FT@R1      = pretrained upstream weights
P-RANDOM@R1  = random weights from frozen P0 config
```

A negative P-RANDOM result is scoped only to the **matched P0 architecture/tokenizer/P-FT recipe**. It may not be generalized to “random initialization cannot work.”

## 6.8 `P-FT@R16` — conditional pretraining × maximum-diversity test

Execute only if all of the following are definitive:

```text
R1  = NO_TRANSFER
R4  = NO_TRANSFER
R16 = NO_TRANSFER
P-RANDOM@R1 = NO_TRANSFER
P-FT@R1 = NO_TRANSFER
```

It uses the exact P-FT recipe but trains on the R16 corpus.

Purpose:

> Before declaring the tested trainable regimes jointly insufficient, test whether the pretrained treatment requires the maximum in-domain diversity–repetition allocation that was also insufficient for C0.

Interpretation:

```text
P-FT@R16 TRANSFER_PASS
    → PRETRAINING_PLUS_MAX_DIVERSITY_VIABLE

P-FT@R16 NO_TRANSFER
    → permits NEITHER_SUFFICIENT if all other required trainable cells are also NO_TRANSFER

P-FT@R16 inconclusive
    → stage cannot emit NEITHER_SUFFICIENT
```

---

# 7. Budget Comparability and Accepted Confounds

For every 8,000-update trainable regime in this stage:

```text
8,000 logical updates × 128 examples = 1,024,000 example presentations
```

The following are held fixed where specified:

```text
logical update count
effective batch
presentation count
checkpoint cadence
visible rendered text
master-seed identities
stream algorithm
```

The following are measured but intentionally not equalized:

```text
parameter count
tokenizer
native sequence length
forward/backward FLOPs
wall clock
peak accelerator memory
number of unique families
average repetition per example
```

Claims may say:

```text
equal logical updates and example presentations
```

Claims may not say:

```text
equal compute
parameter-matched
tokenization-matched across R vs P
diversity-only intervention
```

---

# 8. Seeds, Variance, and Comparison-Only Extension

## 8.1 Six primary evidence seeds

```text
806915476
1031646469
128439691
555223894
454204619
1678768041
```

Derivation:

```text
uint31(SHA256("ExpertForge-E0-v06-msel-seed|" + decimal_index))
indices 0..5
```

These six seeds determine each arm's operational transfer state.

## 8.2 Six preregistered comparison-only seeds

If and only if **two or more trainable regimes already reach `TRANSFER_PASS` on the six primary seeds**, execute the following six additional seeds for those eligible regimes:

```text
1228139313
1536284461
293488859
1941939586
1046059599
920620107
```

Derivation:

```text
uint31(SHA256("ExpertForge-E0-v06-msel-seed|" + decimal_index))
indices 6..11
```

These seeds are used only for paired regime ordering/equivalence. They do **not** retroactively alter the six-seed `TRANSFER_PASS` / `NO_TRANSFER` / `INCONCLUSIVE_FOR_QUALIFICATION` status.

No seed beyond index 11 may be added without a new contract revision.

## 8.3 RNG substreams

Named substreams:

```text
backbone_init
classifier_init
data_order
dataloader_workers
```

The frozen derivation is:

```text
uint31(
  SHA256(
    "ExpertForge-E0-v061-rng|"
    || decimal(master_seed)
    || "|"
    || substream_name
  )
)
```

Arm name is excluded from paired substreams. Decimal master-seed encoding has no zero padding.

## 8.4 Planning sensitivity

Primary transfer gates remain six-seed gates.

Planning assumption:

```text
decision-relevant absolute Q difference: 0.05
paired-seed SD for matched comparisons: ≤ 0.03
```

For 12 paired seeds, SD = 0.03 gives an approximate 90% t-CI half-width of ~0.0156, making the frozen ±0.02 equivalence region practically identifiable near zero difference.

The 0.05 value is a **power-planning effect size**, not a decision threshold.

---

# 9. Executable Metrics and Statistical Estimands

## 9.1 Cell accuracy

For surface `s`, depth `d`, label `l`:

```text
Accuracy(s,d,l)
  = correct predictions in cell (s,d,l)
    / number of examples in cell (s,d,l)
```

Missing output, invalid output, or an output outside the three legal semantic labels counts as incorrect.

## 9.2 CMDR-SMA

```text
Q_ID
  = (1/12) × Σ[d=1..4] Σ[l∈{E,C,U}] Accuracy(ID,d,l)

Q_STRUCT
  = (1/12) × Σ[d=1..4] Σ[l∈{E,C,U}] Accuracy(STRUCT,d,l)

Q
  = (Q_ID + Q_STRUCT) / 2
  = (1/24) × Σ[s∈{ID,STRUCT}] Σ[d=1..4] Σ[l∈{E,C,U}] Accuracy(s,d,l)
```

This is the v0.5 CMDR-SMA metric restated normatively for v0.6.1.

## 9.3 `Q_2:4`

```text
Q_2:4
  = (1/18)
    × Σ[s∈{ID,STRUCT}]
      Σ[d∈{2,3,4}]
      Σ[l∈{E,C,U}]
      Accuracy(s,d,l)
```

Depth-1 accuracy is reported separately.

## 9.4 Per-label recall and minimum recall

For each semantic label `l`, aggregate equally over surface and depth cells:

```text
Recall_l
  = (1/8)
    × Σ[s∈{ID,STRUCT}]
      Σ[d=1..4]
      Accuracy(s,d,l)

minimum_label_recall
  = min(Recall_E, Recall_C, Recall_U)
```

## 9.5 Counterfactual-family exact consistency

For each evaluation surface:

```text
family_exact_consistency(surface)
  = fraction of complete families for which all three E/C/U members are predicted correctly
```

Also report the equal-weight mean across `eval_ID` and `eval_STRUCT`.

## 9.6 Train-surface diagnostics

For interpretation only, report the same behavioral metrics on the training-membership surface used by each regime. Train metrics cannot satisfy a transfer gate.

## 9.7 Representation geometry

D3/D4 cross-family displacement geometry may be reported for continuity:

```text
delta_f,a,b = h_f,a - h_f,b
```

For each pair `{E−C, E−U, C−U}`, report mean pairwise cosine across distinct probe families; `mean_alignment` is their arithmetic mean.

Cross-model comparisons involving P0 are diagnostic only because different tokenizers/architectures can shift hidden geometry even on identical visible strings.

## 9.8 Hierarchical bootstrap estimand

Absolute uncertainty is defined around the **same seed-median statistic used by the operational gate**.

For each of 50,000 deterministic replicates:

1. resample the six primary seed identities with replacement;
2. for each sampled seed and each evaluation surface, resample complete family IDs with replacement **within depth stratum**;
3. retain all three E/C/U members of each sampled family;
4. recompute the requested per-seed metric;
5. take the median of the six resampled seed metrics.

Bootstrap seed:

```text
1611111118
```

derived from:

```text
uint31(SHA256("ExpertForge-E0-v06-msel-bootstrap|0"))
```

Report 2.5th/97.5th percentiles as the 95% interval.

Always report the six individual seed metrics alongside the bootstrap interval. Increasing bootstrap iterations does not increase the number of independent training seeds.

## 9.9 Paired regime ordering

When at least two trainable regimes are eligible, paired ordering uses all 12 preregistered seeds.

For regimes X and Y:

```text
d_s = Q_X(s) - Q_Y(s)
δ_eq = 0.02
```

`BEHAVIORALLY_EQUIVALENT(X,Y)` requires the complete paired-seed 90% Student-t CI for `mean(d)` to lie strictly inside:

```text
[-0.02, +0.02]
```

`X_BEHAVIORALLY_SUPERIOR_TO_Y` requires:

```text
one-sided 95% paired-seed LCB(mean(d)) > 0.02
```

All unordered pairs in the eligible set are evaluated.

A regime is a unique **behavioral winner** only if it is behaviorally superior to every other eligible regime.

A set is **mutually behaviorally equivalent** only if every unordered pair in that set satisfies the equivalence predicate.

If neither a unique behavioral winner nor a single mutually-equivalent eligible set is established, the ordering is:

```text
ORDERING_INCONCLUSIVE
```

No post-hoc seed addition, pair selection, or authority preference may resolve `ORDERING_INCONCLUSIVE`; a new contract revision is required.

The pairwise confidence procedures are preregistered decision procedures and are reported as nominal pairwise intervals; no unreported multiplicity correction is applied.

---

# 10. Run Validity and Operational Transfer States

## 10.1 Run-status state machine

Every arm/seed attempt receives exactly one status:

```text
VALID
INVALID_CONTRACT
INVALID_INFRASTRUCTURE
DIVERGED_SCIENTIFIC
REPLAY_MISMATCH
```

Definitions:

- `VALID`: frozen contract executed and produced a finite selectable checkpoint.
- `INVALID_CONTRACT`: code/config/data violated a frozen contract field; no scientific metric is admitted.
- `INVALID_INFRASTRUCTURE`: hardware/runtime/file-system failure unrelated to the scientific recipe; repair may rerun the **same seed** under incident lineage.
- `DIVERGED_SCIENTIFIC`: frozen recipe executed correctly but numerical/training behavior produced no finite selectable checkpoint. This is a scientifically valid negative result.
- `REPLAY_MISMATCH`: a required deterministic replay identity/tolerance check failed; execution lane stops pending incident review.

Retries never replace a scientifically valid attempt. Infrastructure/contract repairs use new attempt IDs and preserve all prior evidence.

For gate aggregation, `DIVERGED_SCIENTIFIC` receives:

```text
Q = 0
Q_ID = 0
Q_STRUCT = 0
Q_2:4 = 0
all label recalls = 0
family exact consistency = 0
```

and the divergence reason is reported separately.

## 10.2 `TRANSFER_PASS`

All must hold on the six primary seeds:

```text
median:
    Q ≥ 0.55
    Q_2:4 ≥ 0.50
    Q_STRUCT ≥ 0.50
    minimum_label_recall ≥ 0.45

every seed:
    Q ≥ 0.50
```

No upper headroom ceiling is applied at mechanism selection. A very high-Q regime may later be rejected for insufficient E0 source headroom.

## 10.3 `NO_TRANSFER`

All must hold:

```text
median Q ≤ 0.40
median Q_STRUCT ≤ 0.40
hierarchical-bootstrap 95% UCB(Q) < 0.45
```

`NO_TRANSFER` is about central transfer failure; catastrophic individual seeds are reported but are not an additional requirement.

## 10.4 `INCONCLUSIVE_FOR_QUALIFICATION`

Any scientifically valid arm satisfying neither `TRANSFER_PASS` nor `NO_TRANSFER` is inconclusive.

The intentionally wide region between the pass and no-transfer gates is a governance feature:

> Intermediate performance is insufficient to justify downstream qualification, but also insufficient to reject the regime mechanistically.

No additional seeds may be added to escape this region.

---

# 11. Pre-Registered Mechanism Interpretation

## 11.1 C0 diversity–repetition ladder

```text
R1 TRANSFER_PASS
    → BASE_C0_RANDOM_VIABLE

R1 not pass; R4 TRANSFER_PASS
    → DIVERSITY_REPETITION_ALLOCATION_SUPPORTED_4X

R1/R4 not pass; R16 TRANSFER_PASS
    → DIVERSITY_REPETITION_ALLOCATION_SUPPORTED_16X

R1/R4/R16 all NO_TRANSFER
    → C0_ALLOCATION_INSUFFICIENT_WITHIN_16X

otherwise
    → C0_ALLOCATION_INCONCLUSIVE
```

Non-monotonic outcomes are valid and are reported explicitly. A smaller rung passing while a larger rung fails is not rewritten as noise or monotonicity.

## 11.2 Pretrained and matched-P0 interpretation

```text
P-FROZEN pass
    → PRETRAINED_FROZEN_LINEAR_READOUT_VIABLE

P-FROZEN pass + P-RANDOM-FROZEN NO_TRANSFER
    → PRETRAINED_FIXED_FEATURES_ADD_VALUE_WITHIN_MATCHED_PROBE

P-FROZEN pass + P-RANDOM-FROZEN pass
    → RANDOM_FIXED_FEATURES_SUFFICIENT_WITHIN_MATCHED_PROBE

P-FT@R1 pass + P-RANDOM@R1 NO_TRANSFER
    → PRETRAINING_TREATMENT_SUPPORTED_AT_R1_WITHIN_MATCHED_RECIPE

P-FT@R1 pass + P-RANDOM@R1 pass + behaviorally equivalent
    → MATCHED_P0_ARCH_TOKENIZER_RECIPE_SUFFICIENT_AT_R1

P-FT@R1 pass + P-RANDOM@R1 pass + P-FT superior
    → PRETRAINING_ADDS_VALUE_WITHIN_MATCHED_P0_RECIPE

P-FT@R1 pass + P-RANDOM@R1 inconclusive
    → PRETRAINING_ATTRIBUTION_INCONCLUSIVE

P-FT@R16 pass after its trigger
    → PRETRAINING_PLUS_MAX_DIVERSITY_VIABLE
```

No result may be renamed “representation prior proven.”

## 11.3 Primary stage outcome automaton

Primary outcome is based on **trainable regimes only**. P-FROZEN and P-RANDOM-FROZEN are mechanism probes and do not enter this automaton.

Trainable regime cells are:

```text
R1
R4
R16
P-RANDOM@R1
P-FT@R1
P-FT@R16  # only if triggered
```

Let `PASS_SET` be the set of executed trainable cells with `TRANSFER_PASS`.

Exactly one primary outcome is emitted:

```text
if |PASS_SET| >= 2:
    MULTIPLE_VIABLE

else if PASS_SET == {R1}:
    RANDOM_BASE_VIABLE

else if PASS_SET == {R4} or PASS_SET == {R16}:
    IN_DOMAIN_ALLOCATION_VIABLE

else if PASS_SET == {P-RANDOM@R1}:
    P0_RANDOM_VIABLE

else if PASS_SET == {P-FT@R1} or PASS_SET == {P-FT@R16}:
    PRETRAINED_VIABLE

else if |PASS_SET| == 0
        and R1 = NO_TRANSFER
        and R4 = NO_TRANSFER
        and R16 = NO_TRANSFER
        and P-RANDOM@R1 = NO_TRANSFER
        and P-FT@R1 = NO_TRANSFER
        and P-FT@R16 was triggered
        and P-FT@R16 = NO_TRANSFER:
    NEITHER_SUFFICIENT

else:
    INCONCLUSIVE
```

This formulation is mutually exclusive and exhaustive over all scientifically valid combinations, including non-monotonic R-ladder outcomes.

`P-FROZEN` success with no trainable success is reported as a mechanism substatus and the primary outcome remains `INCONCLUSIVE`, not `NEITHER_SUFFICIENT`.

---

# 12. Candidate-Regime Selection for Later v0.6 Qualification

Mechanism selection freezes a **candidate regime set**; it does not itself bind final M0.

Trainable cell eligibility:

```text
R1 eligible iff TRANSFER_PASS
R4 eligible iff TRANSFER_PASS
R16 scientifically eligible iff TRANSFER_PASS
P-RANDOM@R1 eligible iff TRANSFER_PASS
P-FT@R1 eligible iff TRANSFER_PASS
P-FT@R16 eligible iff executed and TRANSFER_PASS

P-FROZEN and P-RANDOM-FROZEN are never eligible final students
```

If exactly one trainable cell is eligible, it is the sole mechanism candidate subject to the resource conditions below.

If two or more are eligible, execute comparison-only seeds 6..11 for all eligible cells and apply Section 9.9.

Selection order:

1. unique behavioral winner, if one exists;
2. otherwise, if all eligible cells form one mutually behaviorally equivalent set:
   1. lower frozen projected student active hours;
   2. lower measured peak accelerator memory;
   3. fewer trainable parameters;
   4. lexicographic regime ID;
3. otherwise:
   ```text
   ORDERING_INCONCLUSIVE
   ```
   and no candidate is silently selected.

`ORDERING_INCONCLUSIVE` halts the lifecycle before the candidate-regime set can be reduced to a single preferred regime. A new contract revision is required.

## 12.1 Frozen student-side resource projection

The upper-bound number of full 8,000-update student runs inherited from the complete v0.5 E0 plan is frozen conservatively as:

```text
60 tuning runs
+ up to 48 decision-authoritative evidence runs
= 108 full student runs

N_full_student_runs_projection = 108
```

For each eligible regime, define:

```text
t_run_p95
  = 95th percentile wall-clock hours over all VALID 8,000-update
    mechanism-selection runs for that regime, including comparison seeds if executed

projected_student_active_hours
  = 108 × t_run_p95
```

Student-side resource PASS requires:

```text
projected_student_active_hours + 8 h A0 allowance + 36 h source-side allowance ≤ 504 h
peak accelerator memory ≤ 12 GiB
projected workspace ≤ 128 GiB
```

Equivalently:

```text
projected_student_active_hours ≤ 460 h
```

A regime failing this student-side projection is:

```text
RESOURCE_INELIGIBLE
```

and cannot become the final E0 student regardless of behavioral superiority.

## 12.2 Frozen source-side feasibility formula for high-diversity regimes

For a later E0 corpus preserving the selected rung's unique-family scale, define:

```text
N_train_examples_required = 3 × selected_train_families
N_source_forwards_required = N_train_examples_required + 12,000 validation examples

max_source_seconds_per_example
  = (36 × 3600) / N_source_forwards_required
```

Examples:

```text
R1-scale:  24,000 + 12,000 =  36,000 forwards → ≤ 3.600 s/example
R4-scale:  96,000 + 12,000 = 108,000 forwards → ≤ 1.200 s/example
R16-scale: 384,000 + 12,000 = 396,000 forwards → ≤ 0.3273 s/example
```

This formula is frozen before mechanism scoring. Actual source-side eligibility cannot be resolved until a source candidate is benchmarked later. Therefore a high-diversity regime may be:

```text
MECHANISM_ELIGIBLE_RESOURCE_CONDITIONAL
```

after mechanism selection, but it cannot reach final v0.6 E0 freeze unless at least one selected/frozen source satisfies the corresponding latency and memory/storage ceilings.

This is a frozen future dependency, not a post-hoc feasibility judgment.

Final workspace feasibility uses the frozen formula:

```text
projected_workspace_bytes
  = fixed_workspace_overhead_bytes
  + checkpoint_bytes
  + N_source_forwards_required × source_event_bytes_per_example
  + N_train_examples_required × student_artifact_bytes_per_example
```

The byte coefficients are execution-derived only from the later frozen artifact schemas and are bound before final v0.6 E0 freeze. Pass requires:

```text
projected_workspace_bytes ≤ 128 GiB
```

The formula is fixed here so storage feasibility cannot be redefined after performance is known.

---

# 13. Selection-Corpus Burn Rule and Dependency Firewall

After mechanism-selection closure, permanently burn:

```text
all CMDR-MSEL-v0 family IDs
all CMDR-MSEL-v0 StructSig hashes
the entire ExpertForge-E0-v061-msel generator namespace
all smoke/preflight corpus family IDs and signatures
```

Burned identities/signatures are prohibited from:

```text
v0.6 student qualification
source/headroom qualification prompts
final E0 train/validation/test corpus
A0 training/validation material
terminal E0 test
```

A hard separation validator must compare family IDs and `CMDR-StructSig-v1` hashes before any later corpus is admitted.

Dependency graph:

```text
v0.5 Q2 CLOSED / UNQUALIFIED
        │
        └── immutable historical record

v0.6.1 mechanism selection
        ↓
fresh v0.6 student qualification
        ↓
exact M0-v0.6 binding or null
        ↓
source/headroom qualification
        ↓
pre-freeze binding snapshot
        ↓
v0.6 FROZEN
        ↓
E0 implementation/evidence
```

A mechanism-selection student PASS never means “v0.5 Q2 requalified.”

---

# 14. Resource Envelope and Preflight Timing

Hard execution surface remains:

```yaml
accelerator_count: 1
accelerator_memory_limit_bytes: 12884901888
host_memory_limit_bytes: 68719476736
storage_limit_bytes: 137438953472
active_execution_wall_clock_limit_hours: 504
```

For this mechanism stage:

```text
one full evidence run ceiling ≤ 8 h
model offload = prohibited
optimizer offload = prohibited
storage-backed live-model paging = prohibited
Engram/HCM = prohibited
```

Every run reports:

```text
wall time
peak accelerator memory
host peak RSS
logical non-padding token presentations
forward token count
parameter count
trainable parameter count
estimated/model-reported FLOPs where available
```

Before execution release, a separate burned smoke corpus must run a representative 400-update block for:

```text
R16
P-FT@R1
P-RANDOM@R1
```

and, if the conditional P-FT@R16 implementation differs beyond dataset membership, `P-FT@R16`.

The smoke must include one checkpoint/dev evaluation. Conservative full-run projection is:

```text
projected_8k_wall = 1.25 × 20 × observed_400_update_wall
```

and must be ≤ 8 h.

Smoke measurements do not authorize hyperparameter changes.

---

# 15. Architecture-Escalation Cooling-Off Clause

`NEITHER_SUFFICIENT` does not authorize generic architecture search.

A later architecture-hypothesis contract must state:

```text
specific missing inductive-bias hypothesis
mechanism expected to supply it
one simpler competing explanation
one bounded discriminating falsifier
terminal horizon
predefined negative interpretation
```

Candidate mechanism classes may include, without authorization:

```text
relational / variable-binding operators
recurrent state mechanisms
graph-like operators
content-addressed lookup tables
lexical Engram controls
structure-addressed conditional memory
hierarchical GPU→RAM→NVMe conditional-memory tiers
```

Naming these classes grants no implementation authority.

---

# 16. Governance and Authorization

This candidate grants **no execution authority**.

A future `V06_MECHANISM_BOOTSTRAP_AUTHORIZED` action may authorize only:

- `CMDR-MSEL-v0` generator extensions and corpus materialization;
- independent proof verification;
- `CMDR-StructSig-v1` implementation;
- structural-support/depletion audit;
- leakage and shortcut audits;
- exact P0 snapshot/digest/license/interface capture;
- metric/sampler/decision implementation from this candidate;
- token-length and resource smoke tests;
- deterministic replay validators;
- frozen reporting schemas/manifests.

It may not authorize:

- R/P evidence scoring;
- comparison-only seed scoring;
- v0.6 student qualification;
- Q3/source qualification;
- final E0 corpus generation;
- A0 or downstream E0 evidence;
- architecture experiments;
- Engram/HCM implementation.

A separate `MECHANISM_SELECTION_EXECUTION_RELEASED` action is required before any evidence cell is scored.

---

# 17. Pre-Execution Blocking Set

Before mechanism-selection execution may be released, all of the following must be frozen and PASS:

1. exact SHA-256 of this v0.6.1 candidate/final mechanism-selection contract;
2. exact generator and independent verifier digests;
3. exact `CMDR-StructSig-v1` implementation digest;
4. exact `MSEL-ExampleStream-v1` implementation digest;
5. exact metric implementation digest for Section 9;
6. exact decision-program digest implementing Sections 10–12;
7. exact run/report-schema digests;
8. family/rendered-string leakage audit PASS;
9. `eval_STRUCT` structural-signature isolation PASS;
10. structural-support/depletion report PASS;
11. shortcut audits PASS;
12. exact selection-corpus split digests;
13. exact six primary seeds, six comparison-only seeds, bootstrap seed, and substream derivation;
14. exact P0 snapshot identity;
15. exact config/tokenizer/special-token/model digests;
16. exact P0 license/provenance record;
17. exact instantiated P0 classification parameter count;
18. P0 token-length audit PASS with no truncation;
19. R tokenizer token-length audit recorded;
20. exact runtime/dependency snapshot;
21. 400-update wall-clock/resource smoke projections PASS;
22. deterministic stream replay PASS;
23. deterministic model replay PASS under the frozen tolerance policy;
24. project authority issues explicit mechanism-selection execution release.

---

# 18. Determinism and Replay Policy

The following must be bitwise identical when produced by deterministic pure-data code on the same frozen runtime:

```text
family membership
family ordering
MSEL-ExampleStream-v1 sample IDs
corpus manifests
StructSig hashes
decision-program output given identical input JSON
```

For model computation replay:

```text
selected checkpoint ID must match exactly
semantic predictions must match exactly
metric scalars must match exactly when replayed on the same frozen runtime/device stack
```

If exact floating-point metric replay is not achieved on an explicitly different but approved hardware/runtime stack, a tolerance policy must be frozen **before** execution release. No tolerance may be invented after evidence exists.

Any authoritative replay mismatch is `REPLAY_MISMATCH` and stops the execution lane.

---

# 19. Reporting Schema — Minimum Required Fields

Every evidence report must contain:

```text
contract digest
code digest
runtime snapshot digest
arm/regime ID
master seed
all derived RNG substream seeds
working-tree/repository identity
run status
dataset/split digests
tokenizer/config/model digests where applicable
parameter count
trainable parameter count
selected checkpoint + selection metric
final update checkpoint identity
per-seed Q / Q_ID / Q_STRUCT / Q_2:4
per-depth accuracy
per-label recall + minimum recall
family exact consistency
train-surface diagnostics
wall time
peak accelerator memory
host RSS
non-padding token presentations
FLOP estimate if available
incident lineage if any
```

Stage closure additionally reports:

```text
hierarchical-bootstrap intervals
arm transfer states
P probe substates
PASS_SET
conditional-arm trigger decisions
all-pairs 12-seed comparison results if applicable
primary stage outcome
ordering outcome
eligible candidate set
resource status
reason codes
```

---

# 20. Normative Inheritance for Downstream E0

The mechanism-selection stage in this document is self-contained enough to implement and audit.

The following downstream E0 concepts remain inherited from v0.5 candidate SHA-256 `25c829a63b7e6cf17d43c8b09b0fa6cd1cf0d739dac7fee8f0042a0a278e3dc6` until a standalone final v0.6 contract is materialized:

```text
terminal Expertise Adapter causal question
A/B/C/D/F/G conceptual arm semantics
D-vs-F primary utility contrast
source-tap context-completeness rule
A0 gradient isolation
source-event artifact immutability
downstream six-seed evidence principle
protected-regression principle
repository/publication boundaries
terminal E0 outcome family
contract-amendment rule
```

No inherited field may be used to execute mechanism selection unless it is restated or explicitly pinned in this candidate.

---

# Appendix A — Seed Registry

```yaml
primary_mechanism_selection_seeds:
  derivation: uint31(SHA256("ExpertForge-E0-v06-msel-seed|" + decimal_index))
  indices: [0, 1, 2, 3, 4, 5]
  values:
    - 806915476
    - 1031646469
    - 128439691
    - 555223894
    - 454204619
    - 1678768041

comparison_only_seeds:
  derivation: uint31(SHA256("ExpertForge-E0-v06-msel-seed|" + decimal_index))
  indices: [6, 7, 8, 9, 10, 11]
  values:
    - 1228139313
    - 1536284461
    - 293488859
    - 1941939586
    - 1046059599
    - 920620107

hierarchical_bootstrap_seed:
  derivation: uint31(SHA256("ExpertForge-E0-v06-msel-bootstrap|0"))
  value: 1611111118
```

`uint31` means: take the first four SHA-256 bytes as an unsigned big-endian 32-bit integer and clear the high bit with `0x7fffffff`.

Index encoding is canonical base-10 ASCII with no zero padding.

---

# Appendix B — P0 External Evidence Record

Observed upstream identity:

```text
EleutherAI/pythia-70m
revision: a39f36b100fe8a5377810d56c3f4789b9c53ac42
revision length: 40 hexadecimal characters
model.safetensors observed SHA-256:
  ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15
license observed: Apache-2.0
architecture observed:
  GPT-NeoX
  6 layers
  hidden 512
  8 heads
  FFN 2048
  context 2048
  vocab 50304
  tie_word_embeddings = false
```

The exact revision resolves in the upstream repository, and the safetensors page reports the stated SHA-256. Bootstrap must still independently download/verify the complete snapshot and compute local digests before registry freeze.

This external record establishes candidate identity only. It does not establish CMDR suitability.

---

# Appendix C — Machine-Readable Primary Outcome Definition

Equivalent pseudocode:

```python
trainable = executed_trainable_cells()
passes = {x for x in trainable if state[x] == "TRANSFER_PASS"}

if len(passes) >= 2:
    outcome = "MULTIPLE_VIABLE"
elif passes == {"R1"}:
    outcome = "RANDOM_BASE_VIABLE"
elif passes in ({"R4"}, {"R16"}):
    outcome = "IN_DOMAIN_ALLOCATION_VIABLE"
elif passes == {"P-RANDOM@R1"}:
    outcome = "P0_RANDOM_VIABLE"
elif passes in ({"P-FT@R1"}, {"P-FT@R16"}):
    outcome = "PRETRAINED_VIABLE"
elif (
    not passes
    and state["R1"] == "NO_TRANSFER"
    and state["R4"] == "NO_TRANSFER"
    and state["R16"] == "NO_TRANSFER"
    and state["P-RANDOM@R1"] == "NO_TRANSFER"
    and state["P-FT@R1"] == "NO_TRANSFER"
    and executed("P-FT@R16")
    and state["P-FT@R16"] == "NO_TRANSFER"
):
    outcome = "NEITHER_SUFFICIENT"
else:
    outcome = "INCONCLUSIVE"
```

This outcome function is independent of P-FROZEN/P-RANDOM-FROZEN probe substates.

---

# Appendix D — Current Binding Status

```text
SPEC-BOUND in this candidate:
    scientific mechanism questions
    corpus topology
    diversity–repetition interpretation
    split/leakage semantics
    R1/R4/R16 membership rule
    MSEL-ExampleStream-v1 mathematical definition
    R training recipe/checkpoint semantics
    P-FROZEN / P-RANDOM-FROZEN / P-FT@R1 / P-RANDOM@R1 / P-FT@R16 topology
    six primary + six comparison-only seed policy
    executable metric formulas
    bootstrap estimand
    ±0.02 equivalence/superiority procedure on 12 seeds
    run-status semantics
    mutually exclusive primary outcome automaton
    resource-projection formulas
    burn rule
    architecture cooling-off clause

QUAL-BOUND before mechanism execution:
    exact code digests for generator/verifier/StructSig/stream/metrics/decision/reporting
    exact CMDR-MSEL-v0 split digests
    structural-support/depletion report
    exact P0 snapshot/tokenizer/config/model/license/parameter binding
    runtime snapshot
    token-length audits
    400-update resource-smoke results
    replay validation

HELD:
    mechanism-selection evidence
    comparison-only evidence
    v0.6 student qualification
    Q3/source work
    final E0 corpus
    A0/student evidence
    Engram/HCM architecture work
```

---

# Candidate Closure Statement

v0.6.1 moves the next decision boundary before adapter experimentation and removes the remaining researcher degrees of freedom identified in review.

The mechanism-selection stage must first establish at least one trainable student regime that transfers across held-out CMDR families and the structural surface. Only then may the project create a fresh v0.6 student qualification corpus and determine whether any selected regime is suitable for the eventual Expertise Adapter experiment.

**No bootstrap or evidence execution is authorized by this candidate.**
