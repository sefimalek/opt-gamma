# -*- coding: utf-8 -*-
"""Null model for the expected disagreement D_e.

CircularShiftNull implements the single-continuum strategy of Mathet,
Widlocher and Metivier (2015), section 5.2.1: each annotator's annotation is
circularly shifted as a block, which preserves unit counts, order, lengths,
gaps and categories while randomizing placement. Within a replica, the drawn
offsets are kept at least one mean unit length apart from each other and from
zero, as the paper prescribes to avoid units realigning with themselves.

Relations keep their internal Arg1->Arg2 offset and are placed independently.
"""
from __future__ import annotations

from typing import List, Protocol, Sequence

import numpy as np

from .units import Relation, Span, Unit


class NullModel(Protocol):
    name: str

    def randomize_all_segments(
        self, sets: Sequence[Sequence[Unit]], text_len: int, rng: np.random.Generator
    ) -> List[List[Unit]]: ...

    def randomize_all_relations(
        self, sets: Sequence[Sequence[Relation]], text_len: int, rng: np.random.Generator
    ) -> List[List[Relation]]: ...


def _place_relation(
    l1: int, l2: int, delta: int, text_len: int, rng: np.random.Generator
) -> tuple[int, int]:
    low = max(0, -delta)
    high = min(text_len - l1, (text_len - l2) - delta)
    if high < low:
        return low, low + delta
    s1 = int(rng.integers(low, high + 1))
    return s1, s1 + delta


def _randomize_relations_independent(rels, text_len, rng):
    if not rels:
        return []
    out = []
    for r in rels:
        l1, l2 = r.arg1.length, r.arg2.length
        delta = r.arg2.start - r.arg1.start
        needed = max(l1, delta + l2) - min(0, delta)
        tl = max(int(text_len), needed)
        s1, s2 = _place_relation(l1, l2, delta, tl, rng)
        out.append(Relation(Span(s1, s1 + l1), Span(s2, s2 + l2), r.type))
    return out


class CircularShiftNull:
    """Mathet (2015) §5.2.1: split-and-permute circular shift."""

    name = "circular_shift"
    _MAX_TRIES = 200

    def randomize_all_segments(self, sets, text_len, rng):
        used_shifts: List[int] = [0]  # forbid near-identity
        return [self._shift_one(units, text_len, rng, used_shifts) for units in sets]

    def randomize_all_relations(self, sets, text_len, rng):
        return [_randomize_relations_independent(s, text_len, rng) for s in sets]

    def _shift_one(self, units, text_len, rng, used_shifts):
        if not units:
            return []
        L = max(int(text_len), max(u.span.end for u in units))
        mean_len = float(np.mean([u.span.length for u in units]))

        def ok(s: int) -> bool:
            for u in units:
                ns = (u.span.start + s) % L
                if ns + u.span.length > L:  # would wrap through the end
                    return False
            for prev in used_shifts:
                d = abs(s - prev) % L
                if min(d, L - d) < mean_len:  # circular distance to other shifts
                    return False
            return True

        shift = None
        for _ in range(self._MAX_TRIES):
            cand = int(rng.integers(0, L))
            if ok(cand):
                shift = cand
                break
        if shift is None:  # dense annotation: relax the min-distance constraint
            for _ in range(self._MAX_TRIES):
                cand = int(rng.integers(0, L))
                if all((u.span.start + cand) % L + u.span.length <= L for u in units):
                    shift = cand
                    break
        if shift is None:  # give up on wrapping: linear translation fallback
            ordered = sorted(units, key=lambda u: u.span.start)
            base = ordered[0].span.start
            span_len = max(u.span.end for u in ordered) - base
            lin = int(rng.integers(0, max(1, L - span_len)))
            return [Unit(Span(u.span.start - base + lin,
                              u.span.end - base + lin), u.category) for u in ordered]

        used_shifts.append(shift)
        return [
            Unit(Span((u.span.start + shift) % L,
                      (u.span.start + shift) % L + u.span.length), u.category)
            for u in units
        ]

    # per-annotator convenience (used in tests / ad-hoc calls)
    def randomize_segments(self, units, text_len, rng):
        return self._shift_one(units, text_len, rng, [0])

    def randomize_relations(self, rels, text_len, rng):
        return _randomize_relations_independent(rels, text_len, rng)
