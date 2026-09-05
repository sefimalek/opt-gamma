# -*- coding: utf-8 -*-
"""Positional (Jaccard) and categorical dissimilarities, all in [0, 1].
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from .units import Unit

CategoryDistanceFn = Callable[[str, str], float]


def jaccard_distance_matrix(
    starts1: np.ndarray,
    ends1: np.ndarray,
    starts2: np.ndarray,
    ends2: np.ndarray,
) -> np.ndarray:
    """Pairwise Jaccard *distance* (1 - IoU) between two sets of half-open spans.

    Fully vectorized: returns an ``(n1, n2)`` matrix in one broadcast pass.
    """
    s1 = starts1[:, None].astype(np.int64)
    e1 = ends1[:, None].astype(np.int64)
    s2 = starts2[None, :].astype(np.int64)
    e2 = ends2[None, :].astype(np.int64)

    inter = np.maximum(0, np.minimum(e1, e2) - np.maximum(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jd = np.where(union > 0, 1.0 - inter / union, 1.0)
    return jd


def spans_to_arrays(units: Sequence[Unit]) -> Tuple[np.ndarray, np.ndarray]:
    starts = np.array([u.span.start for u in units], dtype=np.int64)
    ends = np.array([u.span.end for u in units], dtype=np.int64)
    return starts, ends


class CategoryDistance:
    """Categorical dissimilarity from a matrix, a callable, or binary default.

    Parameters
    ----------
    matrix : mapping ``(cat_a, cat_b) -> float`` in [0, 1]. Symmetric lookup
        is attempted (``(a, b)`` then ``(b, a)``); missing pairs of *distinct*
        categories fall back to ``default`` (1.0). Identical categories always
        cost 0 unless explicitly overridden.
    fn : arbitrary callable ``(a, b) -> float``; takes precedence over matrix.
    binary : if neither matrix nor fn given, 0/1 same/different.
    """

    def __init__(
        self,
        matrix: Mapping[Tuple[str, str], float] | None = None,
        fn: CategoryDistanceFn | None = None,
        default: float = 1.0,
        validate: bool = True,
    ):
        self._fn = fn
        self._matrix = dict(matrix) if matrix is not None else None
        self._default = float(default)
        if validate and self._matrix is not None:
            self._validate_matrix()

    def _validate_matrix(self) -> None:
        problems = []
        for (a, b), v in self._matrix.items():
            if not (0.0 <= v <= 1.0):
                problems.append(f"d({a},{b})={v} out of [0,1]")
            if a == b and v != 0.0:
                problems.append(f"nonzero diagonal d({a},{a})={v}")
            sym = self._matrix.get((b, a))
            if sym is not None and sym != v:
                problems.append(f"asymmetric d({a},{b})={v} vs d({b},{a})={sym}")
        if problems:
            raise ValueError(
                "Invalid category distance matrix:\n  " + "\n  ".join(problems)
            )

    def __call__(self, a: str, b: str) -> float:
        if self._fn is not None:
            return float(self._fn(a, b))
        if self._matrix is not None:
            v = self._matrix.get((a, b))
            if v is None:
                v = self._matrix.get((b, a))
            if v is None:
                v = 0.0 if a == b else self._default
            return float(v)
        return 0.0 if a == b else 1.0

    @classmethod
    def binary(cls) -> "CategoryDistance":
        return cls()

    @classmethod
    def binarized(cls, matrix: Mapping[Tuple[str, str], float]) -> "CategoryDistance":
        """Collapse any weighted matrix to 0/1 (same/different)."""
        labels = sorted({x for pair in matrix for x in pair})
        return cls({(a, b): (0.0 if a == b else 1.0) for a in labels for b in labels})


def category_distance_matrix(
    units1: Sequence[Unit], units2: Sequence[Unit], cat_dist: CategoryDistance
) -> np.ndarray:
    """Pairwise categorical dissimilarity matrix (memoized per label pair)."""
    cats1 = [u.category for u in units1]
    cats2 = [u.category for u in units2]
    cache: Dict[Tuple[str, str], float] = {}
    out = np.empty((len(cats1), len(cats2)), dtype=float)
    for i, a in enumerate(cats1):
        for j, b in enumerate(cats2):
            key = (a, b)
            v = cache.get(key)
            if v is None:
                v = cat_dist(a, b)
                cache[key] = v
            out[i, j] = v
    return out
