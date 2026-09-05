# -*- coding: utf-8 -*-
"""opt-gamma: the optimized gamma agreement measure.

Two-annotator gamma for free segmentation, complex labels and explicit
relations. Companion library of the paper; see pygamma-agreement for the
classical n-annotator gamma, gamma-cat and gamma-k.
"""
from .alignment import (
    AlignedPair,
    AlignmentResult,
    align_relations,
    align_segments,
)
from .distances import CategoryDistance, jaccard_distance_matrix
from .gamma import (
    GammaConfig,
    GammaResult,
    gamma_composite,
    gamma_k_relations,
    gamma_k_segments,
    gamma_relations,
    gamma_segments,
)
from .li_null import (
    LiNonOverlapNull,
    n_configurations,
    sparsity_ratio,
    start_distribution,
    uniform_approximation_ok,
)
from .null_models import CircularShiftNull
from .units import (
    Relation,
    Span,
    Unit,
    relations_from_records,
    units_from_records,
)

__version__ = "0.4.1"

__all__ = [
    "Span", "Unit", "Relation",
    "units_from_records", "relations_from_records",
    "CategoryDistance", "jaccard_distance_matrix",
    "align_segments", "align_relations", "AlignmentResult", "AlignedPair",
    "GammaConfig", "GammaResult",
    "gamma_segments", "gamma_relations", "gamma_composite",
    "gamma_k_segments", "gamma_k_relations",
    "CircularShiftNull", "LiNonOverlapNull",
    "start_distribution", "n_configurations",
    "uniform_approximation_ok", "sparsity_ratio",
]
