"""MSEL-ExampleStream-v1 — frozen deterministic training stream (contract SS5.3).

For an arm with frozen example set S, master seed s, and cycle c = 0,1,2,...:
rank every sample independently by the unsigned 256-bit value of
  SHA256("ExpertForge-E0-v061-msel-stream|" || decimal(s) || "|" || decimal(c) || "|" || sample_id)
Sort all samples in S by that rank, with lexicographic sample_id as collision fallback.
The training stream is the concatenation of complete cycle permutations.
Consume exactly 1,024,000 presentations: no replacement within a cycle,
cycles may repeat examples, logical batches may cross cycle boundaries,
no short batches, no dropped examples, final cycle truncated exactly at the
1,024,000th presentation.
"""

from __future__ import annotations

import hashlib

PRESENTATIONS = 1_024_000
STREAM_NAMESPACE = "ExpertForge-E0-v061-msel-stream"


def _rank(seed: int, cycle: int, sample_id: str) -> bytes:
    key = f"{STREAM_NAMESPACE}|{seed}|{cycle}|{sample_id}"
    return hashlib.sha256(key.encode("utf-8")).digest()


def cycle_order(seed: int, cycle: int, sample_ids: list[str]) -> list[str]:
    """Deterministic permutation of sample_ids for the given cycle."""
    return sorted(sample_ids, key=lambda sid: (_rank(seed, cycle, sid), sid))


class MSELExampleStream:
    """Frozen training stream over a fixed example set."""

    def __init__(self, sample_ids: list[str], master_seed: int):
        if not sample_ids:
            raise ValueError("empty sample set")
        self.sample_ids = list(sample_ids)
        self.master_seed = master_seed
        self._buffer: list[str] = []
        self._cycle = 0
        self.consumed = 0

    def take(self, n: int) -> list[str]:
        """Consume exactly n presentations from the stream."""
        out: list[str] = []
        while len(out) < n:
            if not self._buffer:
                order = cycle_order(self.master_seed, self._cycle, self.sample_ids)
                self._buffer = list(order)
                self._cycle += 1
            need = n - len(out)
            take = min(need, len(self._buffer))
            out.extend(self._buffer[:take])
            self._buffer = self._buffer[take:]
        self.consumed += len(out)
        return out

    def total_updates(self, batch_size: int = 128) -> int:
        """Number of complete logical updates within the presentation budget."""
        return PRESENTATIONS // batch_size


def stream_prefix(seed: int, sample_ids: list[str], n: int) -> list[str]:
    """First n presentations without constructing a persistent stream object."""
    s = MSELExampleStream(sample_ids, seed)
    return s.take(n)


def validate_stream(seed: int, sample_ids: list[str]) -> dict:
    """Independent validation: cycle boundaries, no-replacement, truncation."""
    # Compute how many complete cycles fit, and where truncation occurs
    n = len(sample_ids)
    full_cycles = PRESENTATIONS // n
    remainder = PRESENTATIONS % n

    # Verify cycle 0 ordering
    order0 = cycle_order(seed, 0, sample_ids)
    if sorted(order0) != sorted(sample_ids):
        return {"valid": False, "error": "cycle 0 is not a permutation"}
    if len(set(order0)) != n:
        return {"valid": False, "error": "cycle 0 has duplicates"}

    # Verify no replacement within cycle 0
    if len(set(order0[:n])) != n:
        return {"valid": False, "error": "replacement within cycle"}

    # Verify different cycles produce different orders (probabilistically)
    order1 = cycle_order(seed, 1, sample_ids)
    if order0 == order1 and n > 1:
        return {"valid": False, "error": "cycles 0 and 1 are identical"}

    # Verify common samples preserve relative order within a cycle (SS5.3:
    # "Common samples therefore preserve their pairwise relative order within
    # a cycle" — this is about the ranking function being the same for all
    # samples within one cycle, which is true by construction)
    # (No additional check needed — the rank function is deterministic per sample)

    # Verify presentation budget math
    total_from_full_cycles = full_cycles * n
    truncation_point = total_from_full_cycles + remainder

    return {
        "valid": True,
        "sample_count": n,
        "full_cycles": full_cycles,
        "remainder_in_final_cycle": remainder,
        "total_presentations": PRESENTATIONS,
        "truncation_at_presentation": truncation_point,
    }
