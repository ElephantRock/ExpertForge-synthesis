# E0 v0.6.1 SPEC-BOUND Bootstrap Addendum 1

**Addendum SHA-256 will be bound at commit.**
**Candidate contract SHA-256:** `a70a910e2781fe5c37d625f5474032efaba32855d16446412eb590d55caf7e5b`
**Effective mechanism specification = candidate SHA + this addendum SHA.**
**Authority:** `V06-MSEL-CORPUS-FREEZE-PREP-1` (project contract authority).
**Date:** 2026-09-14.

This addendum does not rewrite the historical candidate. It records two SPEC-BOUND clarifications that the candidate text left underspecified, both resolved by the authority before any authoritative corpus work.

---

## 1. train_pool surface binding

**Ruling:** `train_pool` is **ID surface only**.

The candidate contract SS4.1 fixes the train-pool family count (32,000/depth, 128,000 total) but does not bind its surface composition. The authority rules that R1/R4/R16 train on ID families; `dev_ID` and `eval_ID` remain ID; `eval_STRUCT` remains the structurally shifted and StructSig-disjoint evaluation surface. This preserves the intended distinction between in-distribution family transfer and structural transfer rather than putting the STRUCT construction itself into training.

## 2. CMDR-StructSig-v1 implementation binding

**Ruling:** `CMDR-StructSig-v1` is implemented by `structsig_r3.py` (BLISS-based exact canonical labeling via igraph).

The candidate SS4.3 describes the canonicalization procedure using "deterministic first-occurrence canonical symbols" over sorted fact/rule sets. The validated implementation instead encodes each variant as a colored undirected typed incidence graph and computes an exact canonical labeling using the BLISS algorithm in igraph 1.0.0. The two descriptions define the same equivalence relation (structural isomorphism under the contract's retained fields with identifier abstraction and slot-order removal); the BLISS construction is the validated, mature-package realization of that specification.

### Bound implementation identities

| Artifact | SHA-256 |
|---|---|
| `scripts/e0/v061/structsig_r3.py` | `fb8e4944251f98fccd8bd13a6d521a42311fe33407b6862075f63a33e5743af6` |
| `igraph-1.0.0-cp39-abi3-win_amd64.whl` | `faeff8ede0cf15eb4ded44b0fcea6e1886740146e60504c24ad2da14e0939563` |
| `_igraph.pyd` (native binary) | `92d9e9e773313320417c0e9193d04f3e94b948833b4e6c34e46c7f53d122441f` |

### Runtime environment

| Field | Value |
|---|---|
| igraph version | 1.0.0 |
| Python | 3.12.10 (CPython) |
| Platform | Windows-11-10.0.26200-SP0 (x86_64) |

### Validation lineage

- `V06-STRUCTSIG-REMEDIATION-3` differential oracle: 266 structures, 8/8 gates PASS (clean replay `f0b54c7`)
- 4,000-family correctness pilot: 0 verifier/rename/permute/cross-cell/completed-GI failures (`048420a`)
- `V06-MSEL-SURFACE-BINDING-1` contract-exact depletion: all 28 fill checks PASS (`a2b6be0`)

## 3. eval_STRUCT within-surface semantics

The frozen contract requires `eval_STRUCT` StructSigs to be **disjoint from train_pool, dev_ID, and eval_ID**. It does **not** require pairwise StructSig uniqueness within eval_STRUCT. The depletion replay's stricter within-split uniqueness selection is conservative evidence of margin; it is not a new corpus requirement.

---

This addendum is a SPEC-BOUND clarification under SS3 binding classes. No frozen scientific threshold, seed, arm definition, corpus topology count, or statistical rule is changed.
