# Incident: EXECUTION_RUNTIME_SNAPSHOT_DRIFT

**Severity:** §17-blocking (item 20 reopened)
**Discovered by:** project authority, final §17 audit of V06-GPU-BOOTSTRAP-EXECUTION-AUTHORIZED
**Remediation authority:** V06-GPU-BOOTSTRAP-REMEDIATION-1
**Status:** OPEN — remediation in flight

## What happened

The frozen pre-execution runtime snapshot (bound in
`PREEXEC_REMEDIATION_EVIDENCE.json` → `runtime_snapshot`, carried by the
closure) records the execution environment as:

```text
transformers_version = 5.16.1
tokenizers_version   = 0.23.2
pip freeze contains transformers==5.16.1
```

The r0 GPU bootstrap work (smokes + replay, evidence `d9a7fc1`) actually
executed under **transformers 4.50.0** (system Python
`C:\Users\xjedd\...\Python312\python.exe`), not the repo `.venv`
(`C:\AI\ExpertForge-synthesis\.venv`, which carries the frozen stack:
Python 3.12.10, transformers 5.16.1, tokenizers 0.23.2, torch 2.14.0+cu126).
The wrong interpreter was invoked (`python` resolving to the system
installation instead of `.venv\Scripts\python.exe`).

Observable consequences under 4.50.0 that flagged the drift:

- `from_pretrained(..., dtype=...)` rejected — the 4.50 kwarg is `torch_dtype`
  (5.16.1 accepts `torch_dtype` as a deprecated alias);
- the untied GPT-NeoX LM head attribute is `embed_out` (4.50) rather than
  `lm_head` (5.16.1 rename), requiring a convention-robust removal;
- `AutoModelForCausalLM.from_config(cfg)` materialized FP16 from pythia's
  config.json `torch_dtype` default — caught by the harness's FP32 dtype
  assertion (the exact silent-FP16-substitution failure the authority
  prohibited) and overridden explicitly.

## Governance handling

- The prior 5.16.1 record is **preserved as the frozen runtime snapshot** —
  it is not reclassified as an error and is not deleted.
- The r0 GPU evidence executed under the drifted runtime is preserved as
  diagnostic incident evidence (see
  `STRUCTSIG_FIREWALL_GATE_WEAKENED_POST_OBSERVATION.md` for the artifact
  list) and is not admitted as §17 evidence.
- **Remedy chosen:** retain the frozen environment, not the drift. All
  remediation GPU execution runs under the repo `.venv` (transformers 5.16.1 /
  tokenizers 0.23.2 / torch 2.14.0+cu126). A GPU runtime-freeze artifact
  (`GPU_RUNTIME_FREEZE.json`) binds the actual execution interpreter
  (`sys.executable`), Python/torch/CUDA/cuDNN/driver versions, full
  `pip freeze` text + SHA-256, package `RECORD` SHA-256 digests, deterministic
  state, P0 snapshot file digests, the P0 classification-form parameter /
  interface check, and the complete 402k P0 tokenizer audit — each recorded
  field compared field-by-field against the frozen pre-execution snapshot.
- The P0 interface facts re-verified under 5.16.1 (backbone 44,670,976;
  classifier 1,539; readout shape/finite; FP32 enforcement) supersede, for
  execution purposes, the 4.50.0-specific interface observations recorded in
  commit `ac23919`'s message.
