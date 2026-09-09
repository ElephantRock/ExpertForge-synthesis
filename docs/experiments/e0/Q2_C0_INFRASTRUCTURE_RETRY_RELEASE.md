# E0 Q2 C0 Infrastructure Retry Release

**Branch:** `e0/qualification-bootstrap`  
**Issue:** #1  
**PR:** #2  
**Failed launch head:** `48b2597f14405ea046c71e627a5bc63a54feb85a`  
**Scope:** implementation-integrity remediation only. No scientific threshold, seed, architecture, data rule, optimizer rule, or statistical rule changes are authorized.

## Classification

The first C0 launch attempt is classified as:

```text
INFRASTRUCTURE_INVALID_PRE_MODEL
```

The scientific entry path referenced `subprocess.run(...)` while `subprocess` was only imported locally inside `subprocess_git_head()`. The failure therefore occurs during lineage assembly, before the seed loop and before model construction.

No C0 seed result, checkpoint, validation metric, training update, or terminal scientific evidence was produced by this failed launch. It is not a scientific/algorithmic failure and must not be scored as one. Because no seed entered training, it does not consume one of the per-seed infrastructure retry attempts; preserve it instead as a launcher-level infrastructure incident.

Do not retry from `48b2597...`.

## Required incident record

Keep the local stdout/stderr logs unchanged under the gitignored Q2 local-data directory. Before the remediation code commit, create and commit:

```text
docs/experiments/e0/q2_c0_launch_incident_001.json
```

with at least:

```text
schema_id
classification = INFRASTRUCTURE_INVALID_PRE_MODEL
attempted_branch_head = 48b2597f14405ea046c71e627a5bc63a54feb85a
candidate = C0
seed_started = null
updates_completed = 0
model_constructed = false
normal_result_emitted = false
scientific_failure_emitted = false
exception_type = NameError
exception_symbol = subprocess
stdout_log_local_path
stdout_log_sha256
stderr_log_local_path
stderr_log_sha256
```

Do not commit the raw logs unless later required; bind them by SHA-256 and local path.

## Required code remediation

A one-line module-level `import subprocess` is necessary but not sufficient. The failed branch had never been exercised by the smoke suite, so the scientific setup path must gain an explicit non-scientific preflight.

Refactor the common scientific setup into one shared function used by both the full `--candidate` path and the new preflight. It must perform the same pre-training setup in the same order:

1. reject non-authoritative `--wd-scope all`;
2. enable deterministic algorithms and disable TF32;
3. require CUDA;
4. re-hash all four frozen Q1 inputs;
5. assemble lineage, including Git HEAD, clean-tree-at-start, frozen input digests, and runtime snapshot digest;
6. load train_ID / eval_ID / eval_STRUCT with no truncation;
7. construct the requested M0 candidate at the requested frozen seed;
8. construct AdamW with the frozen `exclude_norm_bias` parameter groups.

The full scientific path and preflight must call this same setup function; do not duplicate a second implementation of lineage or input verification.

## New `--preflight` mode

Add a non-scientific command that exercises the real scientific setup branch and one disposable training step without emitting a qualification result:

```powershell
.\.venv\Scripts\python.exe .\scripts\e0\q2_m0_qualify.py `
    --preflight `
    --candidate C0 `
    --wd-scope exclude_norm_bias
```

The preflight must:

- traverse the shared scientific setup described above;
- use the first frozen C0 qualification seed `1647674144` unless a frozen seed is explicitly supplied for diagnostic purposes;
- construct the actual C0 model and optimizer;
- consume the first deterministic microbatch from the frozen training stream;
- execute forward, cross-entropy, backward, accumulated-gradient finiteness check, global grad clipping, one optimizer step, and post-step parameter-finiteness check;
- discard the model afterward;
- write no `result.json`, no `best.pt`, and no candidate summary;
- write only `docs/experiments/e0/q2_c0_scientific_preflight.json` with `non_scientific: true` and `status: PASS|FAIL`.

The preflight manifest must record at least:

```text
schema_id = E0-Q2-SCIENTIFIC-PREFLIGHT-v0
non_scientific = true
candidate = C0
seed = 1647674144
code_git_commit
working_tree_clean_at_start
verified_q1_input_digests
runtime_snapshot_sha256
wd_scope = exclude_norm_bias
parameter_count = 53232643
decay_parameter_count
no_decay_parameter_count
loss_finite
gradients_finite
grad_norm_finite
parameters_finite_after_step
cuda_peak_allocated_bytes
status
```

The preflight must exit non-zero on any setup or finiteness failure.

## Two-commit remediation protocol

Use the same evidence-binding discipline as P4:

### Commit C — code + incident

Commit and push only:

- the module-level import fix;
- shared scientific setup refactor;
- `--preflight` implementation;
- `q2_c0_launch_incident_001.json`;
- any narrowly necessary status text describing the infrastructure failure.

Do not include newly generated preflight/smoke evidence in Commit C.

### Commit D — clean evidence refresh

From a clean checkout of Commit C:

1. run `python -m py_compile scripts/e0/q2_m0_qualify.py`;
2. run the new exact `--preflight --candidate C0 --wd-scope exclude_norm_bias`;
3. rerun Q2 `--smoke` under `exclude_norm_bias`;
4. rerun the existing 16-update non-scientific Q2 training-path exercise under `exclude_norm_bias`;
5. verify the preflight and Q2 smoke manifests both bind to Commit C and report PASS;
6. update `LOCAL_BOOTSTRAP_STATUS.md`;
7. commit and push only the regenerated Q2 evidence/status as Commit D.

Q3 code did not participate in this defect and does not require another smoke rerun solely because of this Q2 import fix.

## Retry release condition

C0 remains **HELD** until Commit C and Commit D are pushed and GitHub-side validation confirms:

- the missing import is fixed;
- scientific setup and preflight share the same code path;
- the incident record binds the preserved local logs by SHA-256;
- preflight PASS;
- Q2 smoke PASS;
- the 16-update path PASS;
- evidence manifests correctly identify Commit C.

After those conditions pass, rerun C0 from the validated Commit-D branch head with the original three frozen seeds and original 8,000-update recipe. No scientific setting may change because of this infrastructure failure.

Q3 capability/headroom remains blocked on authoritative Q2 Arm-A results. Final `CMDR-Corpus-v1` generation remains unauthorized.
