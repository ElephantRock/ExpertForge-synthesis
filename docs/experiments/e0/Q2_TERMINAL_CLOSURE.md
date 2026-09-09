# E0 Q2 Terminal Closure — M0 Qualification

**Date:** 2026-09-09  
**Authority:** `PROJECT_CONTRACT_AUTHORITY:chat-user`  
**Evidence head reviewed:** `9990f8c18f14c93a5a40eb19aa4f5e7dd1346399`  
**Disposition:** `Q2_UNQUALIFIED`

## Observation

The complete frozen M0 ladder has executed without scientific divergence or recipe alteration:

| Candidate | Layers | Parameters | median Q | median Q2:4 | median Q_STRUCT | median min-label recall | Decision-6 |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 | 15 | 53,232,643 | 0.3333333333 | 0.3333333333 | 0.3333333333 | 0.0 | BELOW_FLOOR |
| C1 | 18 | 63,459,331 | 0.3333333333 | 0.3333333333 | 0.3333333333 | 0.0 | BELOW_FLOOR |
| C2 | 21 | 73,686,019 | 0.3333333333 | 0.3333333333 | 0.3333333333 | 0.0 | BELOW_FLOOR |

All nine frozen seed runs completed 8,000 updates. C2 then emitted the pre-registered terminal conclusion `Q2 UNQUALIFIED: C2 remains below floor.` No C3 exists under the frozen ladder.

## Closure ruling

1. **Q2 is closed as `UNQUALIFIED`.** No M0 candidate satisfies Decision-6, so there is no selected `M0-CausalDense-v1` binding for E0 v0.5.
2. **This is not an adapter rejection.** Adapter utility has not been tested. The qualification stage failed to produce the required Arm-A student learnability regime, so the causal E0 experiment cannot advance under this contract revision.
3. **Do not invent C3, retune the frozen ladder, alter thresholds, or reinterpret C2 as a passing Arm-A pilot.** Any such change is a new scientific design and requires a new contract revision.

## Downstream disposition

### Q3 — source qualification

Existing tokenizer/interface/runtime smoke evidence remains valid as precheck evidence. However, **Q3 capability/headroom and source selection are held and must not close under E0 v0.5** because Q2 produced no selected/qualified M0 Arm-A binding. Do not substitute the chance-level C2 result as a selected Arm-A reference merely to satisfy the headroom arithmetic.

### Q4 — final CMDR corpus

**Not authorized.** `CMDR-Corpus-v1` final 108,000-example materialization requires Q2 and Q3 closure with selected M0/S0 identities. Q2 closed without a selected M0, so that prerequisite is unsatisfied.

### Q5/Q6 and freeze

Q5/Q6 remain blocked. E0 v0.5 remains **DRAFT** and cannot advance to the pre-freeze binding snapshot, `FROZEN`, `AUTHORIZE_E0_IMPLEMENTATION`, or `EXECUTION_RELEASED` states.

## Scientific interpretation

The discriminating observation is stronger than a single failed architecture point: increasing depth from 15 to 21 layers did not move the qualification metrics away from chance across nine complete runs. This rules out the hypothesis that the frozen failure is solved merely by scaling depth within the registered M0 family.

It does **not** distinguish among other plausible mechanisms, including an optimization failure, a training-signal or label-interface defect not caught by the current smoke suite, an unsuitable representation/architecture for this task, or a mismatch between the qualification task and the frozen training recipe.

## Next valid action

Preserve all v0.5 qualification evidence unchanged. Any further work should occur in a **separate non-qualification diagnostic/revision lane** whose purpose is to identify the failure mechanism and propose an E0 v0.6 candidate. Diagnostic work may inspect training dynamics and run deliberately non-qualification sanity tests, but it must not mutate or retroactively rescue v0.5 evidence.

A future v0.6 proposal must explicitly state which frozen artifacts are reused, which scientific decisions change, and which qualification stages must be rerun before a new freeze can be considered.
