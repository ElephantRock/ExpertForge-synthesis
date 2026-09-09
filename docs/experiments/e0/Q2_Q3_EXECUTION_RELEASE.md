# E0 Q2/Q3 Qualification Execution Release

**Branch:** `e0/qualification-bootstrap`  
**Issue:** #1  
**Scope:** qualification/bootstrap only. No terminal E0 evidence execution is authorized.

## GitHub authority

GitHub is authoritative. Before every local phase:

```powershell
git fetch origin
git pull --ff-only origin e0/qualification-bootstrap
git status
git log -1 --oneline
```

Do not continue from unpushed scientific state.

## Environment correction

The frozen Decision-25 values `12 GiB accelerator`, `64 GiB host`, and `128 GiB workspace` are **maximum resource ceilings**, not minimum machine capacities. The Windows host's ~32 GiB RAM therefore does not fail the 64 GiB host-memory gate by itself. The machine is admissible if measured execution fits physical memory and all frozen ceilings. The RTX 3080 Ti's 12 GiB device capacity matches the accelerator ceiling; measured process usage must remain within it.

## Runtime installation

The qualification environment is pinned to current stable releases as of 2026-09-07:

- PyTorch 2.14.0
- Transformers 5.16.1
- Accelerate 1.14.0
- safetensors 0.8.0

Install the CUDA 12.6 PyTorch wheel first:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install transformers==5.16.1 accelerate==1.14.0 safetensors==0.8.0 numpy==2.3.5 scikit-learn==1.8.0
```

Then record:

```text
torch.__version__
torch.version.cuda
torch.cuda.is_available()
torch.cuda.get_device_name(0)
torch.cuda.get_device_properties(0).total_memory
torch.cuda.is_bf16_supported()
transformers.__version__
accelerate.__version__
safetensors.__version__
```

**Stop** if CUDA is unavailable or BF16 is unsupported. Do not silently substitute FP16 for the frozen S0 BF16 execution contract.

Set deterministic training/runtime controls before importing torch in Q2:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONHASHSEED=<master seed where applicable>
```

and in code:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

No `torch.compile` is authorized for qualification.

---

# Q2 — M0 qualification

## Required implementation

Implement `scripts/e0/q2_m0_qualify.py` plus any small project-owned support modules under `scripts/e0/`. Do not use an external pretrained model for M0.

The architecture is exactly `M0-CausalDense-v1`:

```text
CMDR-Lex-v1 token IDs
→ learned embedding width 512
→ N pre-norm causal Transformer blocks
→ final RMSNorm
→ post-final-RMSNorm state at <DECIDE>
→ linear 512→3 classifier
```

Candidates differ only by depth:

```text
C0 15 layers 53,232,643 params
C1 18 layers 63,459,331 params
C2 21 layers 73,686,019 params
```

Common geometry:

```text
vocab size               4096
hidden width              512
Q heads                   8
KV heads                  8
head dim                   64
attention                  full causal
attention projection bias false
FFN                        SwiGLU
FFN hidden width           1536
FFN bias                   false
normalization              RMSNorm
RMS epsilon                1e-5
position                   RoPE
RoPE theta                 10000.0
RoPE scaling               none
rotary dimension           64
dropout                     0
max length                 384
truncation                 prohibited
LM head                    absent
```

Initialization:

```text
token embedding                    Xavier uniform gain 1
all attention/MLP matrices         Xavier uniform gain 1
all RMSNorm scales                 exactly 1
task classifier matrix             Xavier uniform gain 1
task classifier bias               exactly 0
```

Numeric regime:

```text
parameters/activations/optimizer state  float32
TF32                                   disabled
student offload                        prohibited
activation checkpointing               false
dataloader workers                     0
```

## Q2 data

Use the locally reproduced, hash-verified Q1 data:

```text
train_ID.jsonl      24,000 — training stream
eval_ID.jsonl        6,000 — qualification checkpoint-selection surface
eval_STRUCT.jsonl    6,000 — structural qualification surface
```

`eval_ID` is the pre-registered qualification validation surface for checkpoint selection. This is qualification only, not terminal E0 evidence.

Use `CMDR-Lex-v1` deterministic lexical encoding and append `<DECIDE>`. Reject any overlength example; do not truncate.

## Q2 seeds and optimizer

Run candidates in order C0→C1→C2 and stop after the **smallest passing candidate**.

Frozen seeds:

```text
1647674144
1110194409
335767543
```

For each seed:

```text
optimizer                    AdamW
beta1                        0.9
beta2                        0.95
epsilon                      1e-8
peak LR                      5e-4
weight decay                 0.05
global grad clip             1.0
effective batch              128
physical microbatch          16
gradient accumulation        8
updates                      8000
warmup                       400 updates linear
post-warmup schedule         cosine
final LR                     0.1 × peak LR
early stopping               false
checkpoint/validation        every 400 updates
eligible checkpoints         400,800,...,8000
selection                    maximum exact eval_ID CMDR-SMA
tie                          earliest update
```

Training stream is repeated deterministic full-corpus permutations; batches may cross permutation boundaries and no example is dropped at a nominal epoch boundary.

Task loss:

```text
mean 3-way cross entropy against gold ENTAILED / CONTRADICTED / UNKNOWN
label smoothing = 0
```

Every run must finish all 8,000 updates. Scientific divergence cannot be rescued by an earlier checkpoint.

## Q2 qualification metrics and gate

For each selected seed checkpoint compute:

```text
Q_ID
Q_STRUCT
Q = (Q_ID + Q_STRUCT) / 2
Q_2:4 across both surfaces
per-label recall across both surfaces
per-depth accuracy
```

Candidate-level values use the median across the three frozen seeds unless the rule explicitly says per-seed.

PASS requires:

```text
0.55 <= median Q <= 0.70
median Q_2:4 >= 0.50
median Q_STRUCT >= 0.50
median min-label-recall >= 0.45
for every seed: 0.50 <= Q_seed <= 0.75
```

Selection rule:

```text
if C0 passes → select C0; do not run C1/C2
if C0 is below floor → run C1, then C2 only as needed
if C0 is above ceiling → STOP; do not scale upward or alter data after seeing the result
if C2 remains below floor → Q2 UNQUALIFIED; do not invent C3
```

Commit machine-readable per-seed metrics, selected checkpoint update/digests, parameter-count verification, and `q2_m0_qualification_summary.json`. Do not commit large recovery checkpoints unless needed for qualification provenance; keep bulky checkpoint payloads under gitignored local storage and record SHA-256 digests.

---

# Q3 — source qualification

Q3 source capability/headroom **must not close until Q2 produces the three Arm-A qualification scores**. Tokenizer/interface/runtime prechecks may be implemented while Q2 runs, but no candidate-registry changes are permitted.

The authoritative finite registry is:

```text
docs/experiments/e0/S0_CANDIDATE_REGISTRY_FROZEN.yaml
```

No candidate additions/removals after source scoring begins except objective static ineligibility already permitted by the registry.

## Q3 source wrapper

The logical CMDR content remains canonical. The source-facing decision suffix is exactly:

```text
Choose exactly one symbol:
A = ENTAILED
B = CONTRADICTED
C = UNKNOWN
Answer:
```

Use a single user message through the model's native chat template with `add_generation_prompt=True`. Native transport/role special tokens are permitted; no examples, rationale, hints, or additional semantic instruction may be added.

No source tokens are generated.

## Q3 label-interface gate

At the actual answer context, verify that `A`, `B`, and `C` each correspond to one distinct contextual token ID. Record the IDs and exact tokenization bytes.

For each evaluation example collect the raw next-token logits for those three token IDs. Semantic source prediction is argmax over the three semantic logits.

`C_format` is the fraction of examples for which the **full-vocabulary next-token argmax** is one of those same three token IDs. Require:

```text
C_format >= 0.95
```

## TPDS binding

Do not infer TPDS from a library's hidden-state naming convention. Attach a forward pre-hook to `model.get_output_embeddings()` / the actual LM output projection and capture its input tensor. The terminal TPDS vector is the captured projection input at the final non-padding prompt position.

Required integrity check:

```text
output_projection(captured_TPDS_terminal)
```

must replay the model's terminal logits within the frozen same-environment numerical tolerance. Record the exact resolved module path and hidden width.

This implements TPDS as the **input to the source output projection**, after the model's final norm where applicable.

## Q3 same-environment replay

On a frozen balanced 240-example subset of eval_ID (20 per label×depth cell selected by SHA-256 rank), run two clean forward passes and require:

```text
exact input token IDs/masks
exact terminal index
TPDS close with atol=1e-5, rtol=1e-5
A/B/C semantic logits close with atol=1e-5, rtol=1e-5
exact semantic argmax
```

Any semantic-decision mismatch fails deterministic-forward qualification.

## Q3 execution modes

All candidates use BF16. No quantization is authorized.

- S0C0: direct CUDA residency if it fits.
- S0C1/S0C2: bounded CPU offload is allowed; disk-backed/offload-folder weight paging is prohibited. Use Accelerate/Transformers dispatch with a maximum GPU allocation no greater than `11 GiB` and CPU allocation no greater than `24 GiB`; record the resolved device map. If this cannot load without disk paging or physical-memory failure, mark the candidate resource-infeasible rather than changing precision or quantizing.

Reset and record CUDA peak-memory statistics for measured sections. Also record process host-memory high-water information where available.

## Q3 capability/headroom gates

Evaluate the selected semantic A/B/C logits over all 6,000 eval_ID and 6,000 eval_STRUCT examples.

Require:

```text
Q(S0) >= 0.80
Q_2:4(S0) >= 0.75
Q_STRUCT(S0) >= 0.75
min label recall(S0) >= 0.70
Q(S0) - median_seed Q(A_pilot) >= 0.15
Q_2:4(S0) - median_seed Q_2:4(A_pilot) >= 0.10
C_format >= 0.95
```

where `Q = (Q_ID + Q_STRUCT)/2` and Arm-A pilot values come from Q2's three frozen seeds.

A source failing capability/headroom is simply non-qualified; it is not evidence against A0.

## Q3 source-cost benchmark and selection

For every candidate that passes all capability/interface/resource gates, benchmark the same fixed balanced 240-example eval_ID subset used for replay.

Before timing, execute one untimed 12-example warmup. For each of three timed repetitions:

```text
batch size = 1
the exact same 240 examples and order
no generation
torch.cuda.synchronize() immediately before and after timed region
record active wall time
record peak CUDA memory
```

Select the qualified candidate with the lowest median active wall time across the three repetitions. A tie within 1% resolves by lower peak accelerator memory, then lower parameter count, then lexicographically smaller immutable source ID.

Do not select the first passing candidate; all statically/runtimely admissible registry candidates must be qualified or objectively marked infeasible before source selection closes.

---

# Required phase discipline

1. Pull this release from GitHub.
2. Install the pinned runtime and commit an updated runtime/environment snapshot.
3. Implement Q2 and Q3 harnesses.
4. Run short non-scientific smoke tests only (shape, one forward/backward, deterministic data order, source load/interface check). No threshold/data changes from smoke-test outcomes.
5. Commit and push the harness + smoke-test manifests **before** expensive full qualification.
6. Comment on issue #1 with that commit SHA.
7. Then run Q2 full qualification.
8. Push Q2 results.
9. Run/complete Q3 capability and headroom qualification using the frozen Q2 pilot results.
10. Push Q3 results.

Do not generate `CMDR-Corpus-v1` final 108,000 examples until Q2 and Q3 are closed and the selected M0/S0 identities have been pushed to GitHub.
