# -*- coding: utf-8 -*-
"""Bipartite alignment between two annotators (Hungarian algorithm).

Admissibility comes in two flavours:
- "hard" (default): a pair is alignable iff d_pos <= tau. Pairs above the
  threshold get BIG_COST and end up as orphans, at soft penalty eta.
- "soft": no threshold; every pair is alignable at its combined cost, and
  the disorder computation (in opt_gamma.gamma) weights each aligned pair by
  the pairing confidence p = max(0, 1 - d_pos), following Mathet (2017).

The cost matrix is padded to (n1+n2) x (n1+n2): real-dummy cells cost eta,
dummy-dummy cells cost delta_empty (0, structural fillers). Padding on both
sides lets the two annotators have orphans in the same alignment, which
max(n1, n2) padding cannot represent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .distances import (
    CategoryDistance,
    category_distance_matrix,
    jaccard_distance_matrix,
    spans_to_arrays,
)
from .units import Relation, Unit

BIG_COST = 1e6
DELTA_EMPTY = 0.0

Admissibility = Literal["hard", "soft"]


@dataclass(frozen=True)
class AlignedPair:
    """One real<->real match, with its cost decomposition."""

    i: int
    j: int
    cost: float
    d_pos: float
    d_cat: float

    @property
    def confidence(self) -> float:
        """Mathet (2017) pairing confidence ``p = max(0, 1 - d_pos)``."""
        return max(0.0, 1.0 - self.d_pos)


@dataclass
class AlignmentResult:
    """Optimal alignment with full diagnostics."""

    pairs: List[AlignedPair] = field(default_factory=list)
    orphans1: List[int] = field(default_factory=list)
    orphans2: List[int] = field(default_factory=list)
    total_cost: float = 0.0
    n_admissible: int = 0
    n1: int = 0
    n2: int = 0
    admissibility: Admissibility = "hard"

    @property
    def n_orphans(self) -> int:
        return len(self.orphans1) + len(self.orphans2)


def _check_costs(eta: float, tau: float, alpha: float, admissibility: str) -> None:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau (Jaccard *distance* threshold) must be in [0,1], got {tau}")
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta (orphan penalty) must be in [0,1], got {eta}")
    if admissibility not in ("hard", "soft"):
        raise ValueError(f"admissibility must be 'hard' or 'soft', got {admissibility!r}")


def _empty_side_result(n1: int, n2: int, eta: float, admissibility: str) -> AlignmentResult:
    res = AlignmentResult(n1=n1, n2=n2, admissibility=admissibility)
    res.orphans1 = list(range(n1))
    res.orphans2 = list(range(n2))
    res.total_cost = eta * (n1 + n2)
    return res


def _solve(
    d_pos: np.ndarray,
    d_cat: np.ndarray,
    *,
    tau: float,
    alpha: float,
    eta: float,
    admissibility: Admissibility,
    delta_empty: float,
) -> AlignmentResult:
    n1, n2 = d_pos.shape
    combined = alpha * d_pos + (1.0 - alpha) * d_cat

    if admissibility == "hard":
        admissible = d_pos <= tau
        C = np.where(admissible, combined, BIG_COST)
        n_adm = int(admissible.sum())
    else:  # soft: every pair alignable; assignment vs eta decides
        C = combined
        n_adm = n1 * n2

    N = n1 + n2
    Cp = np.full((N, N), delta_empty, dtype=float)
    Cp[:n1, :n2] = C
    Cp[:n1, n2:] = eta
    Cp[n1:, :n2] = eta

    rows, cols = linear_sum_assignment(Cp)

    res = AlignmentResult(n1=n1, n2=n2, n_admissible=n_adm, admissibility=admissibility)
    for r, c in zip(rows, cols):
        if r < n1 and c < n2:
            res.pairs.append(
                AlignedPair(int(r), int(c), float(C[r, c]),
                            float(d_pos[r, c]), float(d_cat[r, c]))
            )
            res.total_cost += float(C[r, c])
        elif r < n1:
            res.orphans1.append(int(r))
            res.total_cost += eta
        elif c < n2:
            res.orphans2.append(int(c))
            res.total_cost += eta
        # dummy-dummy pairs are fillers, cost delta_empty = 0
    return res


def align_segments(
    units1: Sequence[Unit],
    units2: Sequence[Unit],
    *,
    cat_dist: CategoryDistance,
    tau: float,
    alpha: float,
    eta: float,
    admissibility: Admissibility = "hard",
    delta_empty: float = DELTA_EMPTY,
) -> AlignmentResult:
    """Optimal alignment between two annotators' segments."""
    _check_costs(eta, tau, alpha, admissibility)
    n1, n2 = len(units1), len(units2)
    if n1 == 0 and n2 == 0:
        return AlignmentResult(admissibility=admissibility)
    if n1 == 0 or n2 == 0:
        return _empty_side_result(n1, n2, eta, admissibility)

    s1, e1 = spans_to_arrays(units1)
    s2, e2 = spans_to_arrays(units2)
    d_pos = jaccard_distance_matrix(s1, e1, s2, e2)
    d_cat = category_distance_matrix(units1, units2, cat_dist)
    return _solve(d_pos, d_cat, tau=tau, alpha=alpha, eta=eta,
                  admissibility=admissibility, delta_empty=delta_empty)


def align_relations(
    rels1: Sequence[Relation],
    rels2: Sequence[Relation],
    *,
    rel_dist: CategoryDistance,
    tau: float,
    alpha: float,
    eta: float,
    admissibility: Admissibility = "hard",
    arg_aggregation: str = "mean",
    delta_empty: float = DELTA_EMPTY,
) -> AlignmentResult:
    """Optimal alignment between two annotators' relations.

    ``d_pos`` aggregates the Jaccard distances of the argument spans:
    ``arg_aggregation="mean"`` (paper default) or ``"max"`` (stricter:
    in hard mode, BOTH arguments must satisfy the threshold).
    """
    _check_costs(eta, tau, alpha, admissibility)
    if arg_aggregation not in ("mean", "max"):
        raise ValueError(f"arg_aggregation must be 'mean' or 'max', got {arg_aggregation!r}")

    n1, n2 = len(rels1), len(rels2)
    if n1 == 0 and n2 == 0:
        return AlignmentResult(admissibility=admissibility)
    if n1 == 0 or n2 == 0:
        return _empty_side_result(n1, n2, eta, admissibility)

    a1s = np.array([r.arg1.start for r in rels1]); a1e = np.array([r.arg1.end for r in rels1])
    b1s = np.array([r.arg2.start for r in rels1]); b1e = np.array([r.arg2.end for r in rels1])
    a2s = np.array([r.arg1.start for r in rels2]); a2e = np.array([r.arg1.end for r in rels2])
    b2s = np.array([r.arg2.start for r in rels2]); b2e = np.array([r.arg2.end for r in rels2])

    J1 = jaccard_distance_matrix(a1s, a1e, a2s, a2e)
    J2 = jaccard_distance_matrix(b1s, b1e, b2s, b2e)
    d_pos = 0.5 * (J1 + J2)

    units1 = [Unit(r.arg1, r.type) for r in rels1]  # reuse category machinery
    units2 = [Unit(r.arg1, r.type) for r in rels2]
    d_cat = category_distance_matrix(units1, units2, rel_dist)

    if admissibility == "hard" and arg_aggregation == "max":
        # both argument spans must pass the threshold individually
        admissible = (J1 <= tau) & (J2 <= tau)
        combined = alpha * d_pos + (1.0 - alpha) * d_cat
        C = np.where(admissible, combined, BIG_COST)
        # inline solve, since _solve assumes admissibility from d_pos alone
        n1_, n2_ = d_pos.shape
        N = n1_ + n2_
        Cp = np.full((N, N), delta_empty, dtype=float)
        Cp[:n1_, :n2_] = C
        Cp[:n1_, n2_:] = eta
        Cp[n1_:, :n2_] = eta
        rows, cols = linear_sum_assignment(Cp)
        res = AlignmentResult(n1=n1_, n2=n2_, n_admissible=int(admissible.sum()),
                              admissibility="hard")
        for r, c in zip(rows, cols):
            if r < n1_ and c < n2_:
                res.pairs.append(AlignedPair(int(r), int(c), float(C[r, c]),
                                             float(d_pos[r, c]), float(d_cat[r, c])))
                res.total_cost += float(C[r, c])
            elif r < n1_:
                res.orphans1.append(int(r)); res.total_cost += eta
            elif c < n2_:
                res.orphans2.append(int(c)); res.total_cost += eta
        return res

    return _solve(d_pos, d_cat, tau=tau, alpha=alpha, eta=eta,
                  admissibility=admissibility, delta_empty=delta_empty)
