# -*- coding: utf-8 -*-
"""Spans, units and relations.

Spans are half-open [start, end) internally, as in spaCy or BRAT.
If your offsets are inclusive [start, end], convert at load time with
Span.from_inclusive() or pass convention="inclusive" to the loaders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Span:
    """A half-open character (or token) interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self):
        if self.end < self.start:
            object.__setattr__(self, "start", self.end)
            object.__setattr__(self, "end", self.start)
        if self.end <= self.start:
            raise ValueError(f"Empty or invalid span [{self.start}, {self.end})")

    @classmethod
    def from_inclusive(cls, start: int, end: int) -> "Span":
        """Build from an inclusive interval ``[start, end]`` (end += 1)."""
        return cls(int(start), int(end) + 1)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Unit:
    """A categorized segment produced by one annotator."""

    span: Span
    category: str


@dataclass(frozen=True)
class Relation:
    """A typed, directed link between two spans (e.g. claim -> example)."""

    arg1: Span
    arg2: Span
    type: str


def units_from_records(
    records: Sequence[tuple], convention: str = "half_open"
) -> list[Unit]:
    """Build units from ``(start, end, category)`` tuples.

    Parameters
    ----------
    convention : ``"half_open"`` (default) or ``"inclusive"``.
    """
    out = []
    for start, end, cat in records:
        span = (
            Span.from_inclusive(start, end)
            if convention == "inclusive"
            else Span(int(start), int(end))
        )
        out.append(Unit(span, str(cat)))
    return out


def relations_from_records(
    records: Sequence[tuple], convention: str = "half_open"
) -> list[Relation]:
    """Build relations from ``(s1, e1, s2, e2, type)`` tuples."""
    mk = (
        Span.from_inclusive
        if convention == "inclusive"
        else lambda a, b: Span(int(a), int(b))
    )
    return [Relation(mk(s1, e1), mk(s2, e2), str(t)) for s1, e1, s2, e2, t in records]
