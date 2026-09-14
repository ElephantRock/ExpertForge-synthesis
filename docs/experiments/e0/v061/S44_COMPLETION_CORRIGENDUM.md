# §4.4 Completion Report Corrigendum — V06-MSEL-FREEZE-PREP-1A

**Corrigendum to:** `S44_COMPLETION_REPORT.json` at `8a4b87a2b6d4e8b4e0770c5e012f80dc059a8c41`
**Status:** The original report is retained unmodified as **conservative stress evidence**. This corrigendum distinguishes its quantities from the **contract-exact projected corpus quantities** and corrects two reporting mismatches.

---

## Correction 1: Conservative stress vs contract-exact projection

The original `S44_COMPLETION_REPORT.json` scans all 10,000 eval_STRUCT candidates and inserts every admissible signature from that full horizon into `msel_burn`. This means fields labelled "total_unique_burned" describe a **larger hypothetical full-horizon burn**, not the actual 500-family selection. For example, d1 reports a post-eval_STRUCT burn of 13,776 rather than the ~9,293 from the actual 500-family selection.

**Classification:** the original report is a **conservative support demonstration** (harder test → PASS is valid). The contract-exact projected corpus quantities are:

| depth | train_pool unique | dev+eval_ID unique | eval_STRUCT selected (500-family) | **projected corpus burn** |
|---|---|---|---|---|
| 1 | 8,606 | 408+401 = 809 | 500 | **~9,915** |
| 2 | 7,270 | 373+358 = 731 | 500 | **~8,501** |
| 3 | 5,735 | 344+359 = 703 | 500 | **~6,938** |
| 4 | 4,496 | 307+309 = 616 | 500 | **~5,612** |

These are the actual burn sizes against which future qualification rejection should be projected. The conservative full-horizon burns (9,293/7,899/6,346/5,101 from `DEPLETION_EXACT_R3`) sit between these and the original report's 13,776/11,635/9,085/7,094.

## Correction 2: Sequential qualification exclusions

The original completion report evaluates each qualification split **independently** against the MSEL burn, without carrying accepted qualification signatures forward across train_ID → eval_ID → eval_STRUCT. The earlier `DEPLETION_EXACT_R3` replay did exercise that stricter sequential separation and filled all splits at all depths. The contract-exact projection should use sequential exclusions.

**Quantities from the sequential replay (`DEPLETION_EXACT_R3`, `a2b6be0`):**

| depth | qual train_ID | qual eval_ID | qual eval_STRUCT | all filled |
|---|---|---|---|---|
| 1 | 2,000/2,000 | 500/500 | 500/500 | YES |
| 2 | 2,000/2,000 | 500/500 | 500/500 | YES |
| 3 | 2,000/2,000 | 500/500 | 500/500 | YES |
| 4 | 2,000/2,000 | 500/500 | 500/500 | YES |

## Correction 3: Rejection-rate decomposition

The original report labels the MSEL-burn collision rate as `structural_collision_rate`. To avoid conflating two distinct rejection reasons, the corrigendum decomposes:

- **burn_overlap_rejection_rate**: fraction of horizon candidates whose StructSig collides with an already-burned MSEL signature (the binding constraint for future qualification)
- **within_split_repeated_structsig_rate**: fraction of horizon candidates rejected only because another candidate in the same selection scan already used that StructSig (relevant to within-split diversity, not to MSEL disjointness)

The `DEPLETION_EXACT_R3` evidence already records these separately as `msel_burn_rejected` and `duplicate_rejected`.

---

## Authoritative eval_STRUCT selection algorithm (frozen)

Per the SPEC-BOUND addendum and this corrigendum, the authoritative eval_STRUCT selection for the CMDR-MSEL-v0 corpus is:

1. Per depth, construct the fixed train_pool/dev_ID/eval_ID burn (deterministic from namespace + split + surface + depth + family_index).
2. Enumerate STRUCT candidate `family_index` in **ascending order** from 0.
3. Accept the **first 500** families whose StructSig is absent from that ID burn.
4. **Do not** reject a candidate merely because another accepted eval_STRUCT family has the same StructSig.
5. Use a fixed **10,000-candidate fail-closed ceiling**. If the ceiling is reached without filling 500, the corpus freeze fails closed.

This algorithm is deterministic before any corpus evidence is examined and does not silently impose the stricter within-eval_STRUCT uniqueness used in the depletion margin test.
