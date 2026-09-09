# E0 Q2 Prelaunch Integrity Audit

**Branch:** `e0/qualification-bootstrap`  
**Issue:** #1  
**PR:** #2  
**Reviewed correction checkpoint:** `aec481327e8e73467662f5469bcb5c50cbaf480a`  
**Scope:** implementation-integrity closure before the first scientific 8,000-update Q2 run. No scientific threshold, seed, architecture, data rule, or statistical rule changes are authorized here.

## Disposition

The R1/R3 correction checkpoint is substantively correct: scientific Q2 now rejects `--wd-scope all`, uses `exclude_norm_bias`, and Q3 accepts only the exact leading-space `" A"`, `" B"`, `" C"` continuations. The corrected smoke results are PASS.

GitHub-side validation nevertheless found four evidence-integrity defects that must be closed before the expensive C0 run. These are implementation/preflight corrections, not design changes.

## P1 — Scientific Q2 must re-verify the frozen Q1 inputs at startup

The full Q2 path currently loads `train_ID.jsonl`, `eval_ID.jsonl`, and `eval_STRUCT.jsonl` but does not fail closed on content-digest mismatch. A prior Q1 PASS is not enough if local files could later change.

Before any scientific `--candidate` run begins, recompute and require the exact frozen SHA-256 values already recorded by `q1_local_reproduction.json`:

```text
train_ID.jsonl     79daa2c007bc914e228a0312d8278b8f5ca7cb2c28f1450d508f9924362aacf1
eval_ID.jsonl      b66b6629173449f71022fba0e7907826733aa7ae28e0b7502716ba4fdd45fce8
eval_STRUCT.jsonl  688865d27b8e1fd208b2b453670306d1b4ecc9ff9a21130a297e67cc4b07c7d8
CMDR-Lex-v1.json   c3d44e7a99163456ef149fdd1b1caf2fd694e8412e6f082480cb05c2ef237471
```

If any file is missing or mismatched, exit before model construction. Record the verified digest set in each scientific seed result or in a shared Q2 run manifest referenced by every seed result.

## P2 — Scientific divergence must fail closed and cannot be rescued by an earlier checkpoint

The frozen release states that every valid seed run completes all 8,000 updates and scientific divergence cannot be rescued by an earlier validation checkpoint. The current training loop has no explicit non-finite guard, so a later NaN/Inf state could leave an earlier `best` checkpoint available.

For scientific Q2 runs, fail the seed immediately if any of the following becomes non-finite:

- microbatch loss;
- accumulated gradients before the optimizer step;
- returned global gradient norm;
- model parameters after the optimizer step;
- validation logits/metric inputs.

On failure, write a small machine-readable `scientific_failure.json` containing candidate, seed, update, failure code, code Git SHA, verified data digests, and last finite checkpoint update if any, then exit non-zero. Do not emit a normal seed qualification result from an earlier checkpoint.

## P3 — Qualification summaries/results must report the corrected semantics and execution lineage

`write_summary()` at the reviewed checkpoint still writes the stale text:

```text
wd_scope_default: "all (literal contract reading; flagged for authority)"
```

This contradicts R1 and must be replaced by an unambiguous Decision-22 binding, e.g.:

```text
wd_scope: exclude_norm_bias
```

Each scientific seed result must also record at least:

```text
code_git_commit
working_tree_clean_at_start
verified_q1_input_digests
runtime_snapshot_sha256_or_blob_reference
```

The candidate summary must preserve those per-seed lineage fields.

## P4 — Smoke manifests must bind to the clean code revision that produced them

The smoke files committed in `aec481...` report `git_commit = 1070a61...` because they were generated while the R1/R3 code changes were still uncommitted. The content demonstrates the corrected behavior, but the `git_commit` field therefore points to a revision that did not contain that behavior.

Use a two-step correction checkpoint:

1. commit and push the P1/P2/P3 code corrections;
2. from that clean pushed code revision, rerun Q2 smoke, the 16-update non-scientific path, and the relevant Q3 smokes;
3. commit only the regenerated manifests/status in a second evidence-refresh commit.

The regenerated smoke manifests must name the clean code commit from step 1. This avoids a circular requirement for a manifest to name the later evidence-only commit that contains it.

## Release condition

C0 full Q2 qualification remains **HELD** until P1-P4 are closed and the refreshed smoke evidence is pushed. Once those conditions pass, no further scientific design ruling is required: run C0 only, all three frozen seeds, 8,000 updates each, then push the three seed results plus `q2_m0_qualification_summary.json` before applying the frozen C0→C1/C2 ladder.

Q3 capability/headroom remains blocked on authoritative Q2 Arm-A results. Final `CMDR-Corpus-v1` generation remains unauthorized.
