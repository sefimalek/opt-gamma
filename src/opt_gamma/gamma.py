# -*- coding: utf-8 -*-
"""Optimized gamma: raw, chance-corrected, composite and per-category scores.

No global switches. Every methodological choice is a field of GammaConfig,
and every score comes back in a GammaResult that records how it was produced.

Disorder, depending on the admissibility mode:
- hard: D_o = total assignment cost / n_bar.
- soft: confidence-weighted mean, D_o = (sum p_i c_i + |O| eta) / (sum p_i + |O|),
  with p_i = max(0, 1 - d_pos). Weak pairs count less instead of being
  excluded, so no NaN can arise from admissibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np

from .alignment import Admissibility, AlignmentResult, align_relations, align_segments
from .distances import CategoryDistance
from .null_models import CircularShiftNull, NullModel
from .units import Relation, Unit

Normalization = Literal["mathet", "comparables"]
NoAdmissiblePolicy = Literal["nan", "orphan_all", "raise"]


@dataclass
class GammaConfig:
    """All methodological knobs, explicit and documented.

    Parameters
    ----------
    alpha : weight of positional dissimilarity; categorical weight is
        ``1 - alpha``.
    tau : hard-mode admissibility threshold on the Jaccard *distance*.
        Ignored in soft mode.
    eta : soft orphan penalty, in [0, 1].
    admissibility : ``"hard"`` (threshold) or ``"soft"`` (Mathet-2017-style
        confidence weighting). See module docstring.
    normalization : ``"mathet"`` -> n_bar = (n1+n2)/2 ;
        ``"comparables"`` -> orphans excluded from n_bar. Hard mode only.
    on_no_admissible : hard mode only. What to do when both annotators
        produced units but no pair passes the threshold: "orphan_all"
        scores the item as all-orphans (recommended), "nan" excludes it,
        which inflates nan-mean corpus averages, "raise" for strict
        pipelines.
    chance_correction : if True, gamma = 1 - D_o/D_e with D_e estimated on
        ``n_iter`` randomized replicas of ``null_model``.
    """

    alpha: float = 0.5
    tau: float = 0.5
    eta: float = 0.5
    lambda_typ: float = 0.7
    admissibility: Admissibility = "hard"
    normalization: Normalization = "mathet"
    on_no_admissible: NoAdmissiblePolicy = "orphan_all"
    chance_correction: bool = False
    null_model: NullModel = field(default_factory=CircularShiftNull)
    n_iter: "int | Literal['auto']" = 50
    max_iter: int = 200
    min_iter: int = 30
    ci_rel_halfwidth: float = 0.02
    seed: Optional[int] = 42
    clamp: bool = True
    arg_aggregation: Literal["mean", "max"] = "mean"


@dataclass
class GammaResult:
    """A gamma score with full provenance."""

    gamma: float
    D_o: float
    D_e: Optional[float]
    n_bar: float
    alignment: Optional[AlignmentResult]
    reason: str = ""
    kind: str = "segments"
    n_replicas: int = 0

    @property
    def is_defined(self) -> bool:
        return not math.isnan(self.gamma)


# --------------------------------------------------------------------------
# Disorder computation (shared by observed value, replicas, and gamma_k).
# `keys*`/`key` restrict the computation to units of a given category /
# relation type, following the gamma_k logic of Mathet (2017): a pair is
# kept iff at least one of its units bears the key.
# --------------------------------------------------------------------------
def _disorder(
    res: AlignmentResult,
    cfg: GammaConfig,
    keys1: Optional[Sequence[str]] = None,
    keys2: Optional[Sequence[str]] = None,
    key: Optional[str] = None,
) -> tuple[float, float, str]:
    """Return ``(D_o, denominator, reason)``; NaNs carry a reason string."""
    restrict = key is not None

    def pair_kept(p) -> bool:
        return (not restrict) or keys1[p.i] == key or keys2[p.j] == key

    def orphan1_kept(i) -> bool:
        return (not restrict) or keys1[i] == key

    def orphan2_kept(j) -> bool:
        return (not restrict) or keys2[j] == key

    pairs = [p for p in res.pairs if pair_kept(p)]
    o1 = [i for i in res.orphans1 if orphan1_kept(i)]
    o2 = [j for j in res.orphans2 if orphan2_kept(j)]

    if cfg.admissibility == "soft":
        mass = sum(p.confidence for p in pairs) + len(o1) + len(o2)
        if mass <= 0:
            return float("nan"), 0.0, "zero confidence mass"
        num = sum(p.confidence * p.cost for p in pairs) + (len(o1) + len(o2)) * cfg.eta
        return num / mass, mass, ""

    # hard mode: per-unit average
    if restrict:
        n1 = sum(1 for k in keys1 if k == key)
        n2 = sum(1 for k in keys2 if k == key)
    else:
        n1, n2 = res.n1, res.n2
    if cfg.normalization == "mathet":
        nbar = (n1 + n2) / 2.0
    else:
        nbar = ((n1 - len(o1)) + (n2 - len(o2))) / 2.0
    if nbar <= 0:
        return float("nan"), nbar, f"n_bar<=0 (normalization={cfg.normalization})"
    total = sum(p.cost for p in pairs) + (len(o1) + len(o2)) * cfg.eta
    return total / nbar, nbar, ""


def _observed_result(res: AlignmentResult, cfg: GammaConfig, kind: str) -> GammaResult:
    if res.n1 == 0 and res.n2 == 0:
        return GammaResult(float("nan"), float("nan"), None, 0.0, res,
                           reason="both annotators empty", kind=kind)

    if (cfg.admissibility == "hard" and res.n1 > 0 and res.n2 > 0
            and res.n_admissible == 0):
        if cfg.on_no_admissible == "raise":
            raise ValueError("No admissible pair above the overlap threshold")
        if cfg.on_no_admissible == "nan":
            return GammaResult(float("nan"), float("nan"), None, float("nan"), res,
                               reason="no admissible pair (policy=nan)", kind=kind)
        # orphan_all: the Hungarian solution already IS all-orphans; score it.

    Do, denom, reason = _disorder(res, cfg)
    if reason:
        return GammaResult(float("nan"), float("nan"), None, denom, res,
                           reason=reason, kind=kind)
    return GammaResult(1.0 - Do, Do, None, denom, res, kind=kind)


def _clamp(g: float, cfg: GammaConfig) -> float:
    return max(-1.0, min(1.0, g)) if (cfg.clamp and not math.isnan(g)) else g


def _text_len(units1, units2, floor: int = 1000) -> int:
    ends = [u.span.end for u in units1] + [u.span.end for u in units2]
    return max(max(ends) if ends else 0, floor)


def _text_len_rel(rels1, rels2, floor: int = 1000) -> int:
    ends = [max(r.arg1.end, r.arg2.end) for r in list(rels1) + list(rels2)]
    return max(max(ends) if ends else 0, floor)


def _expected_disorder(
    randomize, align, cfg: GammaConfig, keys_of=None, key: Optional[str] = None
) -> tuple[float, int, int]:
    """Mean disorder over null-model replicas, as (D_e, skipped, n_used).

    With n_iter="auto", sampling stops once the 95% CI half-width falls
    below ci_rel_halfwidth * D_e (min_iter replicas at least, max_iter cap).
    Replicas with no admissible pair are scored as all-orphans rather than
    skipped, since skipping them would deflate D_e; only degenerate replicas
    (n_bar <= 0, zero confidence mass) are dropped."""
    auto = cfg.n_iter == "auto"
    budget = cfg.max_iter if auto else max(1, int(cfg.n_iter))
    rng = np.random.default_rng(cfg.seed)
    vals: List[float] = []
    skipped = 0
    for it in range(budget):
        r1, r2 = randomize(rng)
        rres = align(r1, r2)
        k1 = keys_of(r1) if keys_of else None
        k2 = keys_of(r2) if keys_of else None
        D, _, reason = _disorder(rres, cfg, k1, k2, key)
        if reason:
            skipped += 1
            continue
        vals.append(D)
        if auto and len(vals) >= max(2, cfg.min_iter):
            m = float(np.mean(vals))
            hw = 1.96 * float(np.std(vals, ddof=1)) / math.sqrt(len(vals))
            if m > 0 and hw <= cfg.ci_rel_halfwidth * m:
                break
    return (float(np.mean(vals)) if vals else float("nan")), skipped, len(vals)


# --------------------------------------------------------------------------
# Public API: segments
# --------------------------------------------------------------------------
def gamma_segments(
    units1: Sequence[Unit],
    units2: Sequence[Unit],
    *,
    cat_dist: CategoryDistance | None = None,
    config: GammaConfig | None = None,
) -> GammaResult:
    """Typological gamma between two annotators' segments."""
    cfg = config or GammaConfig()
    cd = cat_dist or CategoryDistance.binary()

    def align(a, b):
        return align_segments(a, b, cat_dist=cd, tau=cfg.tau, alpha=cfg.alpha,
                              eta=cfg.eta, admissibility=cfg.admissibility)

    res = align(units1, units2)
    out = _observed_result(res, cfg, "segments")
    if not out.is_defined or not cfg.chance_correction:
        out.gamma = _clamp(out.gamma, cfg)
        return out

    tl = _text_len(units1, units2)
    De, skipped, n_used = _expected_disorder(
        lambda rng: cfg.null_model.randomize_all_segments([units1, units2], tl, rng),
        align, cfg,
    )
    out.n_replicas = n_used
    if not np.isfinite(De) or De <= 0:
        out.gamma = float("nan")
        out.reason = f"invalid D_e (skipped={skipped}/{cfg.n_iter})"
        return out
    out.D_e = De
    out.gamma = _clamp(1.0 - out.D_o / De, cfg)
    return out


def gamma_k_segments(
    units1: Sequence[Unit],
    units2: Sequence[Unit],
    *,
    cat_dist: CategoryDistance | None = None,
    config: GammaConfig | None = None,
) -> Dict[str, GammaResult]:
    """Per-category gamma (analogue of Mathet 2017's gamma_k).

    For each category ``k`` present in either annotation, the disorder is
    restricted to aligned pairs involving at least one unit of category
    ``k`` and to orphans of category ``k``, normalized by the mean count of
    ``k``-units per annotator. One alignment is computed and shared.
    """
    cfg = config or GammaConfig()
    cd = cat_dist or CategoryDistance.binary()

    def align(a, b):
        return align_segments(a, b, cat_dist=cd, tau=cfg.tau, alpha=cfg.alpha,
                              eta=cfg.eta, admissibility=cfg.admissibility)

    res = align(units1, units2)
    keys1 = [u.category for u in units1]
    keys2 = [u.category for u in units2]
    categories = sorted(set(keys1) | set(keys2))
    tl = _text_len(units1, units2)

    out: Dict[str, GammaResult] = {}
    for k in categories:
        Do, denom, reason = _disorder(res, cfg, keys1, keys2, k)
        if reason:
            out[k] = GammaResult(float("nan"), float("nan"), None, denom, res,
                                 reason=reason, kind=f"segments[k={k}]")
            continue
        if not cfg.chance_correction:
            out[k] = GammaResult(_clamp(1.0 - Do, cfg), Do, None, denom, res,
                                 kind=f"segments[k={k}]")
            continue
        De, skipped, n_used = _expected_disorder(
            lambda rng: cfg.null_model.randomize_all_segments([units1, units2], tl, rng),
            align, cfg,
            keys_of=lambda us: [u.category for u in us], key=k,
        )
        if not np.isfinite(De) or De <= 0:
            out[k] = GammaResult(float("nan"), Do, None, denom, res,
                                 reason=f"invalid D_e (skipped={skipped}/{cfg.n_iter})",
                                 kind=f"segments[k={k}]")
        else:
            out[k] = GammaResult(_clamp(1.0 - Do / De, cfg), Do, De, denom, res,
                                 kind=f"segments[k={k}]", n_replicas=n_used)
    return out


# --------------------------------------------------------------------------
# Public API: relations
# --------------------------------------------------------------------------
def gamma_relations(
    rels1: Sequence[Relation],
    rels2: Sequence[Relation],
    *,
    rel_dist: CategoryDistance | None = None,
    config: GammaConfig | None = None,
) -> GammaResult:
    """Relational gamma between two annotators' links."""
    cfg = config or GammaConfig()
    rd = rel_dist or CategoryDistance.binary()

    def align(a, b):
        return align_relations(a, b, rel_dist=rd, tau=cfg.tau, alpha=cfg.alpha,
                               eta=cfg.eta, admissibility=cfg.admissibility,
                               arg_aggregation=cfg.arg_aggregation)

    res = align(rels1, rels2)
    out = _observed_result(res, cfg, "relations")
    if not out.is_defined or not cfg.chance_correction:
        out.gamma = _clamp(out.gamma, cfg)
        return out

    tl = _text_len_rel(rels1, rels2)
    De, skipped, n_used = _expected_disorder(
        lambda rng: cfg.null_model.randomize_all_relations([rels1, rels2], tl, rng),
        align, cfg,
    )
    out.n_replicas = n_used
    if not np.isfinite(De) or De <= 0:
        out.gamma = float("nan")
        out.reason = f"invalid D_e (skipped={skipped}/{cfg.n_iter})"
        return out
    out.D_e = De
    out.gamma = _clamp(1.0 - out.D_o / De, cfg)
    return out


def gamma_k_relations(
    rels1: Sequence[Relation],
    rels2: Sequence[Relation],
    *,
    rel_dist: CategoryDistance | None = None,
    config: GammaConfig | None = None,
) -> Dict[str, GammaResult]:
    """Per-relation-type gamma: gamma_k transposed to the relational axis.

    No equivalent exists in the gamma family or in pygamma-agreement; it
    identifies which relation types drive relational disagreement.
    """
    cfg = config or GammaConfig()
    rd = rel_dist or CategoryDistance.binary()

    def align(a, b):
        return align_relations(a, b, rel_dist=rd, tau=cfg.tau, alpha=cfg.alpha,
                               eta=cfg.eta, admissibility=cfg.admissibility,
                               arg_aggregation=cfg.arg_aggregation)

    res = align(rels1, rels2)
    keys1 = [r.type for r in rels1]
    keys2 = [r.type for r in rels2]
    types = sorted(set(keys1) | set(keys2))
    tl = _text_len_rel(rels1, rels2)

    out: Dict[str, GammaResult] = {}
    for k in types:
        Do, denom, reason = _disorder(res, cfg, keys1, keys2, k)
        if reason:
            out[k] = GammaResult(float("nan"), float("nan"), None, denom, res,
                                 reason=reason, kind=f"relations[k={k}]")
            continue
        if not cfg.chance_correction:
            out[k] = GammaResult(_clamp(1.0 - Do, cfg), Do, None, denom, res,
                                 kind=f"relations[k={k}]")
            continue
        De, skipped, n_used = _expected_disorder(
            lambda rng: cfg.null_model.randomize_all_relations([rels1, rels2], tl, rng),
            align, cfg,
            keys_of=lambda rs: [r.type for r in rs], key=k,
        )
        if not np.isfinite(De) or De <= 0:
            out[k] = GammaResult(float("nan"), Do, None, denom, res,
                                 reason=f"invalid D_e (skipped={skipped}/{cfg.n_iter})",
                                 kind=f"relations[k={k}]")
        else:
            out[k] = GammaResult(_clamp(1.0 - Do / De, cfg), Do, De, denom, res,
                                 kind=f"relations[k={k}]", n_replicas=n_used)
    return out


def gamma_composite(
    typ: GammaResult,
    rel: GammaResult,
    lambda_typ: float = 0.7,
    *,
    on_missing_rel: Literal["typ_only", "nan"] = "typ_only",
) -> float:
    """Composite ``lambda * gamma_typ + (1 - lambda) * gamma_rel``."""
    if typ.is_defined and rel.is_defined:
        return lambda_typ * typ.gamma + (1.0 - lambda_typ) * rel.gamma
    if typ.is_defined and on_missing_rel == "typ_only":
        return typ.gamma
    return float("nan")
