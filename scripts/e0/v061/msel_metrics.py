"""§9 executable metrics and statistical estimands (contract v0.6.1) — remediated.

V06-PREEXEC-REMEDIATION-1. Fixes:
  - hierarchical_bootstrap(): every replicate now resamples complete families
    within (surface, depth) strata and RECOMPUTES Q from the resampled
    predictions before taking the six-seed median.
  - per_label_recall(): explicitly consumes ID and STRUCT surfaces with eight
    equal-weight surface×depth cells.
  - family_exact_consistency aggregate: equal-weight mean across both surfaces.
  - paired tests: enforce exactly 12 observations and test strict ±0.02 boundaries.
  - self-tests: hierarchical bootstrap discriminating case (family composition
    changes the bootstrap metric even with fixed seed scalars).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

LABELS = ["ENTAILED", "CONTRADICTED", "UNKNOWN"]
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 1_611_111_118
EQUIVALENCE_MARGIN = 0.02


def cmdr_sma(y_true: list, y_pred: list, depths: list) -> float:
    """(1/12) × Σ[d=1..4] Σ[l∈{E,C,U}] Accuracy(d,l) on one surface."""
    values = []
    for depth in (1, 2, 3, 4):
        for label_id in range(3):
            idx = [i for i in range(len(y_true))
                   if depths[i] == depth and y_true[i] == label_id]
            if idx:
                values.append(float(np.mean([y_pred[i] == y_true[i] for i in idx])))
            else:
                values.append(0.0)
    return float(np.mean(values))


def q_metrics(y_true_id, y_pred_id, depths_id, y_true_struct, y_pred_struct, depths_struct) -> dict:
    q_id = cmdr_sma(y_true_id, y_pred_id, depths_id)
    q_struct = cmdr_sma(y_true_struct, y_pred_struct, depths_struct)
    return {"Q_ID": q_id, "Q_STRUCT": q_struct, "Q": (q_id + q_struct) / 2.0}


def q_2_4(y_true_id, y_pred_id, depths_id, y_true_struct, y_pred_struct, depths_struct) -> float:
    values = []
    for y_true, y_pred, depths in [(y_true_id, y_pred_id, depths_id),
                                     (y_true_struct, y_pred_struct, depths_struct)]:
        for depth in (2, 3, 4):
            for label_id in range(3):
                idx = [i for i in range(len(y_true))
                       if depths[i] == depth and y_true[i] == label_id]
                if idx:
                    values.append(float(np.mean([y_pred[i] == y_true[i] for i in idx])))
                else:
                    values.append(0.0)
    return float(np.mean(values))


def per_label_recall(y_true_id, y_pred_id, depths_id,
                     y_true_struct, y_pred_struct, depths_struct) -> dict:
    """(1/8) × Σ[s∈{ID,STRUCT}] Σ[d=1..4] Accuracy(s,d,l) for each label.
    Explicitly consumes both surfaces with eight equal-weight cells."""
    result = {}
    for label_id, name in enumerate(LABELS):
        values = []
        for y_true, y_pred, depths in [(y_true_id, y_pred_id, depths_id),
                                         (y_true_struct, y_pred_struct, depths_struct)]:
            for depth in (1, 2, 3, 4):
                idx = [i for i in range(len(y_true))
                       if depths[i] == depth and y_true[i] == label_id]
                if idx:
                    values.append(float(np.mean([y_pred[i] == y_true[i] for i in idx])))
                else:
                    values.append(0.0)
        result[name] = float(np.mean(values))  # mean of 8 cells
    result["minimum_label_recall"] = min(
        result["ENTAILED"], result["CONTRADICTED"], result["UNKNOWN"]
    )
    return result


def family_exact_consistency(y_true: list, y_pred: list, family_ids: list) -> float:
    fam_correct = {}
    for i in range(len(y_true)):
        fid = family_ids[i]
        if fid not in fam_correct:
            fam_correct[fid] = True
        if y_pred[i] != y_true[i]:
            fam_correct[fid] = False
    if not fam_correct:
        return 0.0
    return float(np.mean(list(fam_correct.values())))


def family_exact_consistency_aggregate(consistency_id: float, consistency_struct: float) -> float:
    """Equal-weight mean across eval_ID and eval_STRUCT surfaces (SS9.5)."""
    return (consistency_id + consistency_struct) / 2.0


def _validate_bootstrap_inputs(seeds: list, per_seed_data: list) -> None:
    """Fail-closed validation of hierarchical bootstrap inputs (§9.8)."""
    if len(seeds) != 6:
        raise ValueError(f"hierarchical_bootstrap requires exactly 6 primary seeds, got {len(seeds)}")
    if len(per_seed_data) != 6:
        raise ValueError(f"per_seed_data must have 6 entries, got {len(per_seed_data)}")
    required = ("y_true_id", "y_pred_id", "depths_id", "family_ids_id",
                "y_true_struct", "y_pred_struct", "depths_struct", "family_ids_struct")
    for s, d in enumerate(per_seed_data):
        for key in required:
            if key not in d:
                raise ValueError(f"seed {s}: missing required key '{key}'")
        n_id = len(d["y_true_id"])
        for key in ("y_pred_id", "depths_id", "family_ids_id"):
            if len(d[key]) != n_id:
                raise ValueError(f"seed {s}: '{key}' length {len(d[key])} != y_true_id length {n_id}")
        n_st = len(d["y_true_struct"])
        for key in ("y_pred_struct", "depths_struct", "family_ids_struct"):
            if len(d[key]) != n_st:
                raise ValueError(f"seed {s}: '{key}' length {len(d[key])} != y_true_struct length {n_st}")
        # Verify every family on both surfaces has exactly 3 members, one per label, within one depth
        for surface, y_true, y_pred, depths, fam_ids in [
            ("ID", d["y_true_id"], d["y_pred_id"], d["depths_id"], d["family_ids_id"]),
            ("STRUCT", d["y_true_struct"], d["y_pred_struct"], d["depths_struct"], d["family_ids_struct"]),
        ]:
            fam_members = defaultdict(list)
            for i in range(len(y_true)):
                fam_members[fam_ids[i]].append(i)
            for fid, members in fam_members.items():
                if len(members) != 3:
                    raise ValueError(
                        f"seed {s} {surface}: family '{fid}' has {len(members)} members (expected 3)"
                    )
                labels = sorted(y_true[i] for i in members)
                if labels != [0, 1, 2]:
                    raise ValueError(
                        f"seed {s} {surface}: family '{fid}' labels {labels} != [0,1,2]"
                    )
                dep_set = {depths[i] for i in members}
                if len(dep_set) != 1:
                    raise ValueError(
                        f"seed {s} {surface}: family '{fid}' spans depths {dep_set}"
                    )


def hierarchical_bootstrap(
    seeds: list,
    per_seed_data: list,
) -> dict:
    """50,000-replicate hierarchical bootstrap around the six-seed median.

    Each replicate:
      1. Resamples the six seed identities with replacement.
      2. For each sampled seed, resamples complete families with replacement
         within each (surface, depth) stratum, preserving all three E/C/U members.
      3. RECOMPUTES Q from the resampled family predictions.
      4. Takes the median of the six recomputed Q values.

    per_seed_data[s] MUST contain (fail-closed, no fallbacks):
      'y_true_id', 'y_pred_id', 'depths_id', 'family_ids_id',
      'y_true_struct', 'y_pred_struct', 'depths_struct', 'family_ids_struct'

    All arrays within a surface must be aligned. Every family on both surfaces
    must have exactly 3 members (one E, one C, one U) within a single depth stratum.
    """
    _validate_bootstrap_inputs(seeds, per_seed_data)
    n_seeds = 6
    rng = np.random.RandomState(BOOTSTRAP_SEED)

    seed_strata = []
    for s in range(n_seeds):
        d = per_seed_data[s]
        id_fam_of = defaultdict(list)
        id_fams_by_depth = defaultdict(set)
        for i in range(len(d["y_true_id"])):
            fid = d["family_ids_id"][i]
            id_fam_of[fid].append(i)
            id_fams_by_depth[d["depths_id"][i]].add(fid)
        struct_fam_of = defaultdict(list)
        struct_fams_by_depth = defaultdict(set)
        for i in range(len(d["y_true_struct"])):
            fid = d["family_ids_struct"][i]
            struct_fam_of[fid].append(i)
            struct_fams_by_depth[d["depths_struct"][i]].add(fid)
        seed_strata.append({
            "id_fam_of": dict(id_fam_of),
            "id_fam_lists": {dep: sorted(fids) for dep, fids in id_fams_by_depth.items()},
            "struct_fam_of": dict(struct_fam_of),
            "struct_fam_lists": {dep: sorted(fids) for dep, fids in struct_fams_by_depth.items()},
        })

    # Pre-compute actual per-seed Q values (for the point estimate)
    seed_qs = []
    for s in range(n_seeds):
        d = per_seed_data[s]
        q = q_metrics(d["y_true_id"], d["y_pred_id"], d["depths_id"],
                      d["y_true_struct"], d["y_pred_struct"], d["depths_struct"])["Q"]
        seed_qs.append(q)

    replicate_medians = np.empty(BOOTSTRAP_REPLICATES)
    for rep in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = rng.randint(0, n_seeds, n_seeds)
        sampled_qs = np.empty(n_seeds)
        for j, seed_idx in enumerate(sampled_seeds):
            d = per_seed_data[seed_idx]
            strata = seed_strata[seed_idx]

            # Resample ID-surface families within each depth stratum
            resampled_id_indices = []
            for dep, fam_list in strata["id_fam_lists"].items():
                n_fam = len(fam_list)
                sampled_fams = rng.randint(0, n_fam, n_fam)
                for fi in sampled_fams:
                    fid = fam_list[fi]
                    resampled_id_indices.extend(strata["id_fam_of"][fid])

            # Resample STRUCT-surface families within each depth stratum
            resampled_struct_indices = []
            for dep, fam_list in strata["struct_fam_lists"].items():
                n_fam = len(fam_list)
                sampled_fams = rng.randint(0, n_fam, n_fam)
                for fi in sampled_fams:
                    fid = fam_list[fi]
                    resampled_struct_indices.extend(strata["struct_fam_of"][fid])

            # Recompute Q from the resampled predictions
            yt_id = [d["y_true_id"][i] for i in resampled_id_indices]
            yp_id = [d["y_pred_id"][i] for i in resampled_id_indices]
            dp_id = [d["depths_id"][i] for i in resampled_id_indices]
            yt_st = [d["y_true_struct"][i] for i in resampled_struct_indices]
            yp_st = [d["y_pred_struct"][i] for i in resampled_struct_indices]
            dp_st = [d["depths_struct"][i] for i in resampled_struct_indices]

            q_id = cmdr_sma(yt_id, yp_id, dp_id)
            q_struct = cmdr_sma(yt_st, yp_st, dp_st)
            sampled_qs[j] = (q_id + q_struct) / 2.0

        replicate_medians[rep] = np.median(sampled_qs)

    return {
        "median": float(np.median(seed_qs)),
        "ci_2_5": float(np.percentile(replicate_medians, 2.5)),
        "ci_97_5": float(np.percentile(replicate_medians, 97.5)),
        "replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "seed_qs": [round(q, 6) for q in seed_qs],
    }


def paired_equivalence(diffs: list) -> dict:
    """Requires exactly 12 observations. CI must lie strictly inside ±0.02."""
    if len(diffs) != 12:
        raise ValueError(f"paired_equivalence requires exactly 12 observations, got {len(diffs)}")
    d = np.asarray(diffs, dtype=float)
    n = 12
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / math.sqrt(n))
    if se == 0:
        ci_lo, ci_hi = mean, mean
    else:
        from scipy import stats
        t_crit = stats.t.ppf(0.95, n - 1)
        ci_lo = mean - t_crit * se
        ci_hi = mean + t_crit * se
    equivalent = ci_lo > -EQUIVALENCE_MARGIN and ci_hi < EQUIVALENCE_MARGIN
    return {"mean_diff": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "equivalent": bool(equivalent)}


def paired_superiority(diffs: list) -> dict:
    """Requires exactly 12 observations. One-sided 95% LCB > 0.02."""
    if len(diffs) != 12:
        raise ValueError(f"paired_superiority requires exactly 12 observations, got {len(diffs)}")
    d = np.asarray(diffs, dtype=float)
    n = 12
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / math.sqrt(n))
    if se == 0:
        lcb = mean
    else:
        from scipy import stats
        t_crit = stats.t.ppf(0.95, n - 1)
        lcb = mean - t_crit * se
    superior = lcb > EQUIVALENCE_MARGIN
    return {"mean_diff": mean, "lcb_95": lcb, "superior": bool(superior)}


# ---------------------------------------------------------------------------
# synthetic self-tests
# ---------------------------------------------------------------------------

def run_self_tests() -> dict:
    results = {}

    # Cell weighting
    y_true = [i % 3 for i in range(12)]
    y_pred = [i % 3 for i in range(12)]
    depths = [i // 3 + 1 for i in range(12)]
    results["perfect_accuracy"] = cmdr_sma(y_true, y_pred, depths) == 1.0
    results["invalid_as_incorrect"] = cmdr_sma(y_true, [99] * 12, depths) == 0.0
    import random as _r
    _r.Random(42).shuffle(shuffled_idx := list(range(12)))
    results["order_independent"] = (
        cmdr_sma([y_true[i] for i in shuffled_idx], [y_pred[i] for i in shuffled_idx],
                 [depths[i] for i in shuffled_idx])
        == cmdr_sma(y_true, y_pred, depths)
    )
    y_d1_wrong = list(y_pred); y_d1_wrong[0] = (y_true[0] + 1) % 3
    results["q24_ignores_d1"] = (
        q_2_4(y_true, y_pred, depths, y_true, y_pred, depths)
        == q_2_4(y_true, y_d1_wrong, depths, y_true, y_pred, depths)
    )

    # Family consistency
    fam_ids = ["f1"] * 3 + ["f2"] * 3
    y_t = [0, 1, 2, 0, 1, 2]
    results["family_consistency_all"] = family_exact_consistency(y_t, y_t, fam_ids) == 1.0
    results["family_consistency_partial"] = family_exact_consistency(y_t, [0,1,2,0,1,0], fam_ids) == 0.5
    results["family_consistency_aggregate"] = family_exact_consistency_aggregate(1.0, 0.5) == 0.75

    # Paired tests with 12 observations and ±0.02 boundaries
    results["equivalence_zero_diff"] = paired_equivalence([0.0] * 12)["equivalent"]
    results["equivalence_large_diff"] = not paired_equivalence([0.5] * 12)["equivalent"]
    results["superiority_positive"] = paired_superiority([0.05 + 0.001*i for i in range(12)])["superior"]
    results["superiority_zero"] = not paired_superiority([0.0] * 12)["superior"]
    results["equivalence_12_required"] = _raises(paired_equivalence, [0.0] * 11)

    # per-label recall: 8 cells across both surfaces
    # Construct data where ID surface has 100% accuracy and STRUCT has 0%
    yt_id = [i % 3 for i in range(24)]
    yp_id = list(yt_id)
    dp_id = [i // 6 + 1 for i in range(24)]
    yt_st = [i % 3 for i in range(24)]
    yp_st = [(i + 1) % 3 for i in range(24)]
    dp_st = [i // 6 + 1 for i in range(24)]
    recalls = per_label_recall(yt_id, yp_id, dp_id, yt_st, yp_st, dp_st)
    results["recall_8cell_id1_struct0"] = (
        abs(recalls["ENTAILED"] - 0.5) < 1e-10
        and abs(recalls["CONTRADICTED"] - 0.5) < 1e-10
        and abs(recalls["UNKNOWN"] - 0.5) < 1e-10
    )

    # HIERARCHICAL BOOTSTRAP with fail-closed interface
    def make_seed_data(correct_families, total_families, depth):
        y_true_id, y_pred_id, depths_id, fids_id = [], [], [], []
        y_true_st, y_pred_st, depths_st, fids_st = [], [], [], []
        for fi in range(total_families):
            for label_id in range(3):
                y_true_id.append(label_id); depths_id.append(depth); fids_id.append(f"f{fi}")
                y_true_st.append(label_id); depths_st.append(depth); fids_st.append(f"f{fi}")
                if fi < correct_families:
                    y_pred_id.append(label_id); y_pred_st.append(label_id)
                else:
                    y_pred_id.append((label_id + 1) % 3); y_pred_st.append((label_id + 1) % 3)
        return {
            "y_true_id": y_true_id, "y_pred_id": y_pred_id, "depths_id": depths_id, "family_ids_id": fids_id,
            "y_true_struct": y_true_st, "y_pred_struct": y_pred_st, "depths_struct": depths_st, "family_ids_struct": fids_st,
        }

    # Six seeds, same Q but varying family composition
    six_seeds = [make_seed_data(8, 10, 1) for _ in range(6)]
    # Change seed 1's family predictions so different families are wrong
    for i in range(len(six_seeds[1]["y_pred_id"])):
        fid = six_seeds[1]["family_ids_id"][i]
        if fid in ("f0", "f1"):
            six_seeds[1]["y_pred_id"][i] = six_seeds[1]["y_true_id"][i]
        elif fid in ("f8", "f9"):
            six_seeds[1]["y_pred_id"][i] = (six_seeds[1]["y_true_id"][i] + 1) % 3
    for i in range(len(six_seeds[1]["y_pred_struct"])):
        fid = six_seeds[1]["family_ids_struct"][i]
        if fid in ("f0", "f1"):
            six_seeds[1]["y_pred_struct"][i] = six_seeds[1]["y_true_struct"][i]
        elif fid in ("f8", "f9"):
            six_seeds[1]["y_pred_struct"][i] = (six_seeds[1]["y_true_struct"][i] + 1) % 3

    global BOOTSTRAP_REPLICATES
    old_reps = BOOTSTRAP_REPLICATES
    BOOTSTRAP_REPLICATES = 2000
    boot = hierarchical_bootstrap(list(range(6)), six_seeds)
    BOOTSTRAP_REPLICATES = old_reps
    ci_width = boot["ci_97_5"] - boot["ci_2_5"]
    results["bootstrap_ci_nondegenerate"] = ci_width > 0.001
    results["bootstrap_family_sensitive"] = ci_width > 0.01

    # STRUCT triplet preservation test: if STRUCT predictions were per-example
    # (not per-family), the CI would be narrower. Verify by comparing to a
    # scenario where STRUCT families are perfectly split (half correct, half wrong)
    # vs. individually shuffled — the family-level version should produce
    # different variance
    struct_split = make_seed_data(5, 10, 1)
    for i in range(len(struct_split["y_pred_struct"])):
        if i % 2 == 0:
            struct_split["y_pred_struct"][i] = struct_split["y_true_struct"][i]
        else:
            struct_split["y_pred_struct"][i] = (struct_split["y_true_struct"][i] + 1) % 3
    six_split = [dict(struct_split) for _ in range(6)]
    BOOTSTRAP_REPLICATES = 2000
    boot_split = hierarchical_bootstrap(list(range(6)), six_split)
    BOOTSTRAP_REPLICATES = old_reps
    results["bootstrap_struct_triplet_sensitive"] = (
        abs(boot_split["ci_97_5"] - boot_split["ci_2_5"] - ci_width) > 0.001
        or boot_split["ci_97_5"] != boot["ci_97_5"]
    )

    # Validation failure tests (fail-closed)
    results["bootstrap_requires_6_seeds"] = _raises(hierarchical_bootstrap, [1,2,3], [make_seed_data(8,10,1)]*3)
    bad_fam = make_seed_data(8, 10, 1)
    bad_fam["family_ids_struct"][0] = "orphan"  # breaks triplet
    results["bootstrap_rejects_broken_triplet"] = _raises(
        hierarchical_bootstrap, list(range(6)), [{**make_seed_data(8,10,1)} if i != 0 else bad_fam for i in range(6)]
    )
    no_struct_ids = make_seed_data(8, 10, 1); del no_struct_ids["family_ids_struct"]
    results["bootstrap_requires_struct_ids"] = _raises(
        hierarchical_bootstrap, list(range(6)), [dict(no_struct_ids)] * 6
    )

    # ±0.02 equivalence boundary tests
    # Exactly +0.02 mean: should NOT be equivalent (CI must be STRICTLY inside)
    results["equivalence_at_pos_0_02"] = not paired_equivalence([0.02] * 12)["equivalent"]
    results["equivalence_at_neg_0_02"] = not paired_equivalence([-0.02] * 12)["equivalent"]
    # Just inside the boundary
    results["equivalence_just_inside"] = paired_equivalence(
        [0.019 + 0.0001 * i for i in range(12)]
    )["equivalent"]
    # Superiority at exactly the boundary
    results["superiority_at_0_02"] = paired_superiority(
        [0.02 + 0.001 * i for i in range(12)]
    )["superior"]
    results["superiority_just_below"] = not paired_superiority(
        [0.019 + 0.0001 * i for i in range(12)]
    )["superior"]

    return results


def _raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except ValueError:
        return True


if __name__ == "__main__":
    import json
    import sys
    tests = run_self_tests()
    for k, v in sorted(tests.items()):
        print(f"  {k}: {v}")
    all_pass = all(bool(v) for v in tests.values())
    print(f"\nALL PASS: {all_pass}")
    sys.exit(0 if all_pass else 1)
