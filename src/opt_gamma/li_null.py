# -*- coding: utf-8 -*-
"""Random annotation model of Li, Rose, Yuan and Zhou (ACL 2024),
non-overlapping sub-model.

Per annotator, the number of segments and their lengths are fixed; every
non-overlapping placement over the text is equally likely. Their closed-form
expectation (Prop. 1) requires an additive similarity measure, which the
alignment-based gamma disorder is not, so we sample from the model instead
and reuse the same Hungarian alignment.

Sampling is exact and cheap: a configuration is a permutation of the k
segments plus a composition of the n - a free tokens into k + 1 gaps, so a
uniform draw costs O(k). The count k! * C(n-a+k, k) equals the normalizer
pi(n-a+k, k) of their Prop. 2, and start_distribution() implements that
proposition; both were checked against exhaustive enumeration in the tests.
"""
from __future__ import annotations

from math import comb, factorial
from typing import List, Sequence

import numpy as np

from .units import Relation, Span, Unit


def perm(n: int, r: int) -> int:
    """``pi(n, r) = n!/(n-r)!``; 0 when the arrangement is impossible."""
    if r < 0 or n < 0 or n < r:
        return 0
    return factorial(n) // factorial(n - r)


def n_configurations(n: int, lengths: Sequence[int]) -> int:
    """Total number of valid non-overlapping configurations."""
    k = len(lengths)
    a = int(sum(lengths))
    if k == 0:
        return 1
    if n - a < 0:
        return 0
    return perm(n - a + k, k)


def start_distribution(n: int, lengths: Sequence[int], i: int) -> np.ndarray:
    """Analytical law of the start index of segment ``i`` (Proposition 2).

    Returns an array ``p`` of length ``n - a_i + 1`` where ``p[l - 1]`` is
    ``P(ST_i = l)`` for 1-based positions ``l``.
    """
    k = len(lengths)
    a = int(sum(lengths))
    a_i = int(lengths[i])
    others = [int(lengths[j]) for j in range(k) if j != i]

    # For each subset size m, the total of subset sums, obtained by DP over
    # the other segments: sums_by_size[m] maps subset-sum -> count.
    sums_by_size: List[dict] = [{} for _ in range(k)]
    sums_by_size[0][0] = 1
    for length in others:
        for m in range(k - 2, -1, -1):
            for s, cnt in list(sums_by_size[m].items()):
                sums_by_size[m + 1][s + length] = (
                    sums_by_size[m + 1].get(s + length, 0) + cnt
                )

    total = n_configurations(n, lengths)
    out = np.zeros(max(0, n - a_i + 1), dtype=float)
    for l in range(1, n - a_i + 2):
        count = 0
        for m in range(0, k):
            for s, mult in sums_by_size[m].items():
                left = perm(l - s + m - 1, m)
                right = perm(n - l - a + s + k - m, k - m - 1)
                if left and right:
                    count += mult * left * right
        out[l - 1] = count / total if total else 0.0
    return out


def uniform_approximation_ok(
    n: int, lengths: Sequence[int], i: int, alpha: float = 0.99
) -> bool:
    """Proposition 3 criterion: is the start law close enough to uniform?

    True when ``(n - a + k) / (n - a_i + 1) > alpha`` (alpha close to but
    below 1), i.e. when the text is long relative to total annotated length.
    """
    k = len(lengths)
    a = int(sum(lengths))
    denom = n - int(lengths[i]) + 1
    if denom <= 0:
        return False
    return (n - a + k) / denom > alpha


def sparsity_ratio(n: int, a1: int, a2: int) -> float:
    """``n / (a1 + a2)``; Proposition 4 says chance agreement vanishes when
    this is large (``n >> a1 + a2``)."""
    return n / (a1 + a2) if (a1 + a2) > 0 else float("inf")


def _sample_placement(
    n: int, lengths: Sequence[int], rng: np.random.Generator
) -> List[int]:
    """Uniform draw over valid non-overlapping configurations.

    Returns the start offsets (0-based), in the order of ``lengths``.
    """
    k = len(lengths)
    if k == 0:
        return []
    a = int(sum(lengths))
    free = n - a
    if free < 0:  # text too short: fall back to contiguous packing
        pos, out = 0, []
        for length in lengths:
            out.append(pos)
            pos += length
        return out

    order = rng.permutation(k)
    # Uniform composition of `free` into k+1 non-negative parts:
    # choose k cut points among free + k slots.
    cuts = np.sort(rng.choice(free + k, size=k, replace=False))
    gaps = np.diff(np.concatenate(([-1], cuts, [free + k]))) - 1

    starts = [0] * k
    pos = int(gaps[0])
    for slot, seg in enumerate(order):
        starts[seg] = pos
        pos += int(lengths[seg]) + int(gaps[slot + 1])
    return starts


class LiNonOverlapNull:
    """Null model of Li et al. (2024), non-overlapping sub-model."""

    name = "li_non_overlap"

    def randomize_all_segments(self, sets, text_len, rng):
        return [self._one(units, text_len, rng) for units in sets]

    def randomize_all_relations(self, sets, text_len, rng):
        return [self._one_rel(rels, text_len, rng) for rels in sets]

    def _one(self, units: Sequence[Unit], text_len: int, rng) -> List[Unit]:
        if not units:
            return []
        lengths = [u.span.length for u in units]
        n = max(int(text_len), sum(lengths))
        starts = _sample_placement(n, lengths, rng)
        return [
            Unit(Span(s, s + length), u.category)
            for s, length, u in zip(starts, lengths, units)
        ]

    def _one_rel(self, rels: Sequence[Relation], text_len: int, rng) -> List[Relation]:
        """Relations: both argument spans of every relation are placed jointly
        under the same non-overlap constraint, preserving each relation's
        internal ordering."""
        if not rels:
            return []
        lengths, owner = [], []
        for idx, r in enumerate(rels):
            lengths += [r.arg1.length, r.arg2.length]
            owner += [(idx, 1), (idx, 2)]
        n = max(int(text_len), sum(lengths))
        starts = _sample_placement(n, lengths, rng)

        spans = {}
        for (idx, which), s, length in zip(owner, starts, lengths):
            spans[(idx, which)] = Span(s, s + length)
        return [
            Relation(spans[(idx, 1)], spans[(idx, 2)], r.type)
            for idx, r in enumerate(rels)
        ]

    # per-annotator convenience
    def randomize_segments(self, units, text_len, rng):
        return self._one(units, text_len, rng)

    def randomize_relations(self, rels, text_len, rng):
        return self._one_rel(rels, text_len, rng)
