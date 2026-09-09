# E0 Q2 Full Qualification Release

**Branch:** `e0/qualification-bootstrap`  
**Issue:** #1  
**PR:** #2  
**Reviewed harness commit:** `f5d5a88b376f0dba214b2d632947762d9a576124`  
**Scope:** Q2 qualification/bootstrap only. No terminal E0 evidence execution is authorized.

GitHub remains authoritative. The Windows checkout is a working copy only.

## Review disposition

The Q2/Q3 harness-and-smoke checkpoint at `f5d5a88b376f0dba214b2d632947762d9a576124` is accepted as an implementation checkpoint. Q1 remains exact PASS; Q2 smoke and short-path exercise are accepted as non-scientific implementation evidence; Q3 interface smoke is accepted as non-scientific interface evidence.

One load-bearing Q2 correction is required before the first full 8,000-update qualification run. No full scientific Q2 run has started, so this is a pre-execution correction rather than an outcome-conditioned amendment.

## Project-authority rulings on the four interpretation questions

### R1 — Q2 AdamW weight-decay scope

**RULING: `exclude_norm_bias`.**

The frozen training design already defines the parameter-group semantics:

- apply `weight_decay = 0.05` to token embeddings, attention matrices, MLP matrices, and the task-classifier matrix;
- apply `weight_decay = 0` to all RMSNorm scale parameters and the task-classifier bias.

Therefore the current scientific default `--wd-scope all` is not authoritative for full Q2 qualification. Before Q2 begins, change the full-run path to use `exclude_norm_bias` and fail closed if a scientific Q2 invocation requests `all`.

The optional `all` mode may remain only for explicitly non-scientific implementation diagnostics if desired.

After this correction, rerun the Q2 smoke and the short training-path exercise using `exclude_norm_bias`. Those reruns remain non-scientific.

### R2 — Q2 training permutation derivation

**RULING: APPROVED AS IMPLEMENTED.**

For master qualification seed `s` and zero-based permutation epoch `e`, use:

```text
u64_be(first_8_bytes(SHA256(
  "ExpertForge-E0-Q2|M0|perm|" + str(s) + "|" + str(e)
)))
```

Use that integer to seed the deterministic Python `random.Random` permutation for that complete corpus permutation. Epoch numbering starts at 0. Batches may cross permutation boundaries; no example is dropped.

This rule is now frozen for Q2 qualification.

### R3 — Q3 source answer context

**RULING: exact leading-space continuation; no fallback.**

The source content ends exactly with:

```text
Answer:
```

with no trailing newline. The only authorized semantic continuations are the literal UTF-8 strings:

```text
" A"
" B"
" C"
```

The smoke results established that these exact leading-space continuations each append one distinct contextual token for all three frozen registry candidates. Therefore Q3 scoring must require this exact form and must not fall back to bare `"A"`, `"B"`, or `"C"` on a candidate-specific basis.

Record the resulting contextual token IDs and token pieces per source candidate.

### R4 — Q3 deterministic replay subset

**RULING: APPROVED AS IMPLEMENTED.**

On `CMDR-QUAL-v1/eval_ID`, within each of the 12 `(gold_label, reasoning_depth_stratum)` cells, rank by ascending:

```text
SHA256("ExpertForge-E0-Q3|replay|" + sample_id)
```

and take the first 20 examples. The resulting 240-example ordered set is authoritative for Q3 replay and source-cost benchmarking.

This rule is now frozen for Q3 qualification.

## Required correction checkpoint before expensive Q2

The Windows assistant must:

1. pull this GitHub release;
2. change Q2 scientific execution to `exclude_norm_bias` as specified in R1;
3. change Q3 contextual-answer verification to require exactly `" A"`, `" B"`, `" C"` with no fallback, as specified in R3;
4. rerun Q2 smoke and the non-scientific short-path training exercise under the corrected Q2 weight-decay scope;
5. rerun the relevant Q3 tokenizer/interface smoke after removing the fallback;
6. update `LOCAL_BOOTSTRAP_STATUS.md` so the four interpretation questions are shown as resolved;
7. commit and push this correction checkpoint.

Do not start a full Q2 run from commit `f5d5a88...` itself.

## Full Q2 release after the correction checkpoint

Once the correction checkpoint is pushed and its smoke manifests PASS, **C0 full qualification is authorized** with the three frozen seeds:

```text
1647674144
1110194409
335767543
```

Each seed must run all 8,000 updates under the frozen recipe. Do not use the non-authoritative `all` weight-decay mode. Do not shorten a run based on intermediate validation performance.

Run **C0 only** first. After all three C0 seed results and `q2_m0_qualification_summary.json` are committed and pushed:

- if C0 passes all frozen Decision-6 gates, select C0 and do not run C1/C2;
- if C0 is above the primary learnability ceiling, stop Q2 and report the qualification failure; do not scale upward;
- if C0 does not pass and is not above the ceiling, the frozen ladder permits C1 next, but push the C0 evidence before starting C1;
- C2 remains the final candidate; no C3 may be invented.

## Q3 boundary

Q3 tokenizer/interface/runtime smoke may be maintained, but Q3 capability/headroom scoring remains blocked until the three authoritative Q2 Arm-A pilot results for the currently evaluated M0 candidate are pushed.

The final 108,000-example E0 corpus remains **NOT AUTHORIZED** until Q2 and Q3 close and exact M0/S0 identities are committed.
