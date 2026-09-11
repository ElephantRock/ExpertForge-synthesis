#!/usr/bin/env python3
"""
Reference decision program for E0 v0.6.1 mechanism selection.

Design artifact only. It does not authorize execution and it does not read
model checkpoints. It consumes frozen per-arm summary JSON and emits the
mechanism-selection outcome deterministically.

Expected state strings:
  TRANSFER_PASS
  NO_TRANSFER
  INCONCLUSIVE_FOR_QUALIFICATION

Probe cells P-FROZEN / P-RANDOM-FROZEN are reported separately and do not
enter the primary trainable-regime outcome automaton.

r1 remediation (V06-DESIGN-REMEDIATION-1, 2026-09-11):
  conditional_pft_r16_required() now includes the full contract SS6.8
  trigger, adding the previously omitted conjunct
      P-RANDOM@R1 == NO_TRANSFER.
  self_test() now asserts the exact contract-correct exhaustive state space
  (245 assignments) and the exact seven outcome counts, and includes a
  targeted regression case for the omitted-conjunct failure region.
  No other behavior is changed. Supersedes the defective artifact with
  SHA-256 8fef40ef4450c970de6491ba0600aa25ed03a609323c5ffddc39633fef6ffcaa,
  which is preserved unmodified for provenance.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Optional, Set

TRANSFER_PASS = "TRANSFER_PASS"
NO_TRANSFER = "NO_TRANSFER"
INCONCLUSIVE = "INCONCLUSIVE_FOR_QUALIFICATION"

PRIMARY_TRAINABLE = ("R1", "R4", "R16", "P-RANDOM@R1", "P-FT@R1")
CONDITIONAL_TRAINABLE = "P-FT@R16"
ALLOWED_STATES = {TRANSFER_PASS, NO_TRANSFER, INCONCLUSIVE}


def conditional_pft_r16_required(states: Dict[str, str]) -> bool:
    """Frozen trigger from contract SS6.8: all five trainable cells at
    R1/max-diversity scope must be NO_TRANSFER."""
    return (
        states.get("R1") == NO_TRANSFER
        and states.get("R4") == NO_TRANSFER
        and states.get("R16") == NO_TRANSFER
        and states.get("P-RANDOM@R1") == NO_TRANSFER
        and states.get("P-FT@R1") == NO_TRANSFER
    )


def executed_trainable_cells(states: Dict[str, str]) -> Set[str]:
    cells = set(PRIMARY_TRAINABLE)
    if CONDITIONAL_TRAINABLE in states:
        cells.add(CONDITIONAL_TRAINABLE)
    return cells


def primary_outcome(states: Dict[str, str]) -> str:
    """
    Implements contract SS11.3 exactly.

    Missing primary cells are invalid input. P-FT@R16 must be present iff its
    frozen trigger fired; callers should validate this before outcome emission.
    """
    missing = [k for k in PRIMARY_TRAINABLE if k not in states]
    if missing:
        raise ValueError(f"missing primary trainable cells: {missing}")

    for k, v in states.items():
        if k in set(PRIMARY_TRAINABLE) | {CONDITIONAL_TRAINABLE} and v not in ALLOWED_STATES:
            raise ValueError(f"illegal state for {k}: {v}")

    trigger = conditional_pft_r16_required(states)
    has_conditional = CONDITIONAL_TRAINABLE in states
    if trigger != has_conditional:
        raise ValueError(
            f"P-FT@R16 presence mismatch: trigger={trigger}, present={has_conditional}"
        )

    trainable = executed_trainable_cells(states)
    passes = {k for k in trainable if states[k] == TRANSFER_PASS}

    if len(passes) >= 2:
        return "MULTIPLE_VIABLE"
    if passes == {"R1"}:
        return "RANDOM_BASE_VIABLE"
    if passes in ({"R4"}, {"R16"}):
        return "IN_DOMAIN_ALLOCATION_VIABLE"
    if passes == {"P-RANDOM@R1"}:
        return "P0_RANDOM_VIABLE"
    if passes in ({"P-FT@R1"}, {"P-FT@R16"}):
        return "PRETRAINED_VIABLE"

    neither = (
        not passes
        and states["R1"] == NO_TRANSFER
        and states["R4"] == NO_TRANSFER
        and states["R16"] == NO_TRANSFER
        and states["P-RANDOM@R1"] == NO_TRANSFER
        and states["P-FT@R1"] == NO_TRANSFER
        and has_conditional
        and states["P-FT@R16"] == NO_TRANSFER
    )
    if neither:
        return "NEITHER_SUFFICIENT"

    return "INCONCLUSIVE"


def self_test() -> None:
    """
    Exhaustively checks that every valid state assignment emits exactly one
    primary outcome, that the conditional P-FT@R16 trigger is respected, and
    that the exhaustive space and outcome counts match the contract-correct
    oracle from the V06-DESIGN-REMEDIATION-1 ruling.
    """
    counts = {}
    n = 0
    for vals in product(sorted(ALLOWED_STATES), repeat=len(PRIMARY_TRAINABLE)):
        base = dict(zip(PRIMARY_TRAINABLE, vals))
        trigger = conditional_pft_r16_required(base)
        variants = [base]
        if trigger:
            variants = []
            for cstate in sorted(ALLOWED_STATES):
                d = dict(base)
                d[CONDITIONAL_TRAINABLE] = cstate
                variants.append(d)
        for states in variants:
            out = primary_outcome(states)
            counts[out] = counts.get(out, 0) + 1
            n += 1

    expected_counts = {
        "MULTIPLE_VIABLE": 131,
        "RANDOM_BASE_VIABLE": 16,
        "IN_DOMAIN_ALLOCATION_VIABLE": 32,
        "P0_RANDOM_VIABLE": 16,
        "PRETRAINED_VIABLE": 17,
        "NEITHER_SUFFICIENT": 1,
        "INCONCLUSIVE": 32,
    }
    assert counts == expected_counts, {"actual": counts, "expected": expected_counts}
    assert n == 245, n

    # Targeted regression: the omitted-conjunct failure region. P-RANDOM@R1
    # TRANSFER_PASS must suppress the P-FT@R16 trigger even when every other
    # trigger condition holds, and the contract-correct input must yield
    # P0_RANDOM_VIABLE without raising.
    regression = {
        "R1": NO_TRANSFER,
        "R4": NO_TRANSFER,
        "R16": NO_TRANSFER,
        "P-RANDOM@R1": TRANSFER_PASS,
        "P-FT@R1": NO_TRANSFER,
    }
    assert conditional_pft_r16_required(regression) is False
    assert primary_outcome(regression) == "P0_RANDOM_VIABLE"

    print({
        "assignments_checked": n,
        "outcome_counts": counts,
        "regression_case": "PASS",
        "status": "PASS",
    })


if __name__ == "__main__":
    self_test()
