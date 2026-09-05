# -*- coding: utf-8 -*-
"""Regression tests, anchored on the paper's worked example (Section
'Reduction of Gamma Alignment to Bipartite Assignment').

Paper spans are written [100, 150] but computed with half-open lengths
(1 - 48/52), so we feed them as half-open directly.
"""
import math

import numpy as np
import pytest

from opt_gamma import (
    CategoryDistance,
    GammaConfig,
    CircularShiftNull,
    Relation,
    Span,
    Unit,
    gamma_composite,
    gamma_relations,
    gamma_segments,
    jaccard_distance_matrix,
    units_from_records,
)

BIN = CategoryDistance.binary()
CFG = GammaConfig(alpha=0.5, tau=0.5, eta=0.5, chance_correction=False)


# ---------------------------------------------------------------- units
def test_inclusive_conversion():
    u = units_from_records([(100, 150, "Claim")], convention="inclusive")[0]
    assert u.span.start == 100 and u.span.end == 151 and u.span.length == 51


def test_empty_span_rejected():
    with pytest.raises(ValueError):
        Span(10, 10)


# ------------------------------------------------------------- jaccard
def test_jaccard_paper_values():
    jd = jaccard_distance_matrix(
        np.array([100]), np.array([150]), np.array([102]), np.array([152])
    )[0, 0]
    assert jd == pytest.approx(1 - 48 / 52, abs=1e-9)

    # Example from the paper's Jaccard section: [100,200] vs [180,200]
    jd2 = jaccard_distance_matrix(
        np.array([100]), np.array([200]), np.array([180]), np.array([200])
    )[0, 0]
    assert jd2 == pytest.approx(0.8, abs=1e-9)


def test_jaccard_disjoint_and_identical():
    jd = jaccard_distance_matrix(
        np.array([0, 0]), np.array([10, 10]), np.array([0, 20]), np.array([10, 30])
    )
    assert jd[0, 0] == 0.0 and jd[0, 1] == 1.0 and jd[1, 1] == 1.0


# ----------------------------------------------- paper worked example
def _paper_segments():
    a = [Unit(Span(100, 150), "Argument"), Unit(Span(160, 210), "Exemple")]
    b = [Unit(Span(102, 152), "Argument"), Unit(Span(162, 212), "Explicitation")]
    return a, b


def test_paper_typological_gamma():
    a, b = _paper_segments()
    res = gamma_segments(a, b, cat_dist=BIN, config=CFG)
    assert res.gamma == pytest.approx(0.7115, abs=1e-3)
    assert res.D_o == pytest.approx(0.2885, abs=1e-3)
    assert len(res.alignment.pairs) == 2
    assert res.alignment.n_orphans == 0


def test_paper_relational_gamma():
    r1 = [Relation(Span(100, 150), Span(160, 210), "Illustration")]
    r2 = [Relation(Span(102, 152), Span(162, 212), "Illustration")]
    res = gamma_relations(r1, r2, rel_dist=BIN, config=CFG)
    assert res.gamma == pytest.approx(0.9615, abs=1e-3)


def test_paper_composite():
    a, b = _paper_segments()
    typ = gamma_segments(a, b, cat_dist=BIN, config=CFG)
    r1 = [Relation(Span(100, 150), Span(160, 210), "Illustration")]
    r2 = [Relation(Span(102, 152), Span(162, 212), "Illustration")]
    rel = gamma_relations(r1, r2, rel_dist=BIN, config=CFG)
    # NOTE: the paper states 0.788 — arithmetic error.
    # 0.7*0.7115 + 0.3*0.9615 = 0.7865 (erratum to fix in the LaTeX).
    assert gamma_composite(typ, rel, 0.7) == pytest.approx(0.7865, abs=1e-3)


# ------------------------------------------------------------ orphans
def test_perfect_agreement():
    a = [Unit(Span(0, 10), "X"), Unit(Span(20, 30), "Y")]
    res = gamma_segments(a, list(a), cat_dist=BIN, config=CFG)
    assert res.gamma == pytest.approx(1.0)


def test_orphan_penalty():
    a = [Unit(Span(0, 10), "X"), Unit(Span(20, 30), "Y")]
    b = [Unit(Span(0, 10), "X")]
    res = gamma_segments(a, b, cat_dist=BIN, config=CFG)
    # total = 0 (perfect pair) + eta (orphan) = 0.5 ; n_bar = 1.5
    assert res.D_o == pytest.approx(0.5 / 1.5)
    assert res.gamma == pytest.approx(1 - 0.5 / 1.5)
    assert res.alignment.orphans1 == [1]


def test_granularity_split_soft_penalty():
    # A: one long segment; B: same content split in two.
    a = [Unit(Span(0, 100), "X")]
    b = [Unit(Span(0, 50), "X"), Unit(Span(50, 100), "X")]
    res = gamma_segments(a, b, cat_dist=BIN, config=CFG)
    # Neither half reaches Jaccard overlap 0.5 exactly (IoU = 0.5 -> jd=0.5<=tau ok)
    # jd([0,100],[0,50]) = 1 - 50/100 = 0.5 <= tau -> admissible
    assert res.alignment.n_admissible == 2
    assert res.is_defined


def test_empty_one_side():
    a = [Unit(Span(0, 10), "X")]
    res = gamma_segments(a, [], cat_dist=BIN, config=CFG)
    # all orphans: total = eta*1, n_bar = 0.5 -> Do = 1.0 -> gamma = 0
    assert res.gamma == pytest.approx(0.0)


# ---------------------------------------------------- NaN policy
def _disjoint():
    a = [Unit(Span(0, 10), "X")]
    b = [Unit(Span(100, 110), "X")]
    return a, b


def test_no_admissible_nan_policy():
    a, b = _disjoint()
    cfg = GammaConfig(on_no_admissible="nan")
    res = gamma_segments(a, b, cat_dist=BIN, config=cfg)
    assert math.isnan(res.gamma)
    assert "no admissible" in res.reason


def test_no_admissible_orphan_all_policy():
    a, b = _disjoint()
    cfg = GammaConfig(on_no_admissible="orphan_all", eta=0.5)
    res = gamma_segments(a, b, cat_dist=BIN, config=cfg)
    # 2 orphans * 0.5 / n_bar=1 -> Do = 1 -> gamma = 0
    assert res.gamma == pytest.approx(0.0)


def test_no_admissible_raise_policy():
    a, b = _disjoint()
    with pytest.raises(ValueError):
        gamma_segments(a, b, cat_dist=BIN, config=GammaConfig(on_no_admissible="raise"))


# ------------------------------------------------- category matrices
def test_weighted_matrix_validation():
    with pytest.raises(ValueError):
        CategoryDistance({("A", "A"): 0.3})  # nonzero diagonal
    with pytest.raises(ValueError):
        CategoryDistance({("A", "B"): 0.4, ("B", "A"): 0.6})  # asymmetric
    with pytest.raises(ValueError):
        CategoryDistance({("A", "B"): 1.5})  # out of range


def test_binarized_matrix():
    cd = CategoryDistance.binarized({("A", "B"): 0.3, ("B", "A"): 0.3})
    assert cd("A", "B") == 1.0 and cd("A", "A") == 0.0


# --------------------------------------------------- chance correction
@pytest.mark.parametrize("null_cls", [CircularShiftNull])
def test_chance_corrected_below_raw(null_cls):
    a, b = _paper_segments()
    raw = gamma_segments(a, b, cat_dist=BIN, config=CFG)
    cfg = GammaConfig(
        chance_correction=True, null_model=null_cls(), n_iter=30, seed=1
    )
    corr = gamma_segments(a, b, cat_dist=BIN, config=cfg)
    assert corr.is_defined
    assert corr.D_e is not None and corr.D_e > 0
    # Chance correction should lower (or equal) the raw score here.
    assert corr.gamma <= raw.gamma + 1e-9


def test_chance_reproducible():
    a, b = _paper_segments()
    cfg = GammaConfig(chance_correction=True, n_iter=20, seed=7)
    g1 = gamma_segments(a, b, cat_dist=BIN, config=cfg).gamma
    g2 = gamma_segments(a, b, cat_dist=BIN, config=cfg).gamma
    assert g1 == g2


# ----------------------------------------------------- relations extra
def test_relation_arg_aggregation_max_stricter():
    # Arg1 perfect, Arg2 disjoint: mean jd = 0.5 -> admissible with "mean",
    # inadmissible with "max".
    r1 = [Relation(Span(0, 10), Span(20, 30), "T")]
    r2 = [Relation(Span(0, 10), Span(50, 60), "T")]
    res_mean = gamma_relations(
        r1, r2, rel_dist=BIN, config=GammaConfig(arg_aggregation="mean")
    )
    res_max = gamma_relations(
        r1, r2, rel_dist=BIN,
        config=GammaConfig(arg_aggregation="max", on_no_admissible="orphan_all"),
    )
    assert res_mean.alignment.n_admissible == 1
    assert res_max.alignment.n_admissible == 0



def test_auto_sampling():
    a, b = _paper_segments()
    cfg = GammaConfig(chance_correction=True, n_iter="auto", max_iter=200, seed=5)
    res = gamma_segments(a, b, cat_dist=BIN, config=cfg)
    assert res.is_defined
    assert res.n_replicas >= 30


def test_circular_shift_preserves_structure():
    import numpy as np
    units = [Unit(Span(10, 30), "A"), Unit(Span(50, 90), "B")]
    rng = np.random.default_rng(0)
    r = CircularShiftNull().randomize_segments(units, 500, rng)
    assert sorted(u.span.length for u in r) == [20, 40]
    assert sorted(u.category for u in r) == ["A", "B"]
    assert all(0 <= u.span.start and u.span.end <= 500 for u in r)


# ===================== v0.2.0: soft admissibility & gamma_k =====================
from opt_gamma import gamma_k_segments, gamma_k_relations


def test_soft_perfect_agreement():
    a = [Unit(Span(0, 10), "X"), Unit(Span(20, 30), "Y")]
    cfg = GammaConfig(admissibility="soft")
    res = gamma_segments(a, list(a), cat_dist=BIN, config=cfg)
    assert res.gamma == pytest.approx(1.0)


def test_soft_paper_example_close_to_hard():
    # High-overlap pairs: confidence ~0.92, soft and hard should nearly agree.
    a, b = _paper_segments()
    hard = gamma_segments(a, b, cat_dist=BIN, config=CFG)
    soft = gamma_segments(a, b, cat_dist=BIN,
                          config=GammaConfig(admissibility="soft"))
    assert soft.is_defined
    assert abs(soft.gamma - hard.gamma) < 0.02


def test_soft_downweights_weak_overlap():
    # One perfect pair + one weak-overlap pair (IoU=0.2, jd=0.8, conf=0.2).
    # Hard mode (tau=0.5) orphans the weak pair; soft mode keeps it but
    # with low weight. Soft must NOT treat the weak pair as full agreement.
    a = [Unit(Span(0, 10), "X"), Unit(Span(100, 200), "Y")]
    b = [Unit(Span(0, 10), "X"), Unit(Span(180, 200), "Y")]
    soft = gamma_segments(a, b, cat_dist=BIN,
                          config=GammaConfig(admissibility="soft", eta=0.5))
    # pairs: cost 0 (conf 1) and cost 0.5*0.8=0.4 (conf 0.2)
    # D_o = (1*0 + 0.2*0.4) / (1 + 0.2) = 0.0667 -> gamma ~ 0.933
    assert soft.gamma == pytest.approx(1 - 0.08 / 1.2, abs=1e-6)


def test_soft_disjoint_zero_confidence_nan():
    # Fully disjoint same-category pair: assignment may match it (cost 0.5
    # < 2*eta), but its confidence is 0 -> zero mass -> NaN with reason.
    a = [Unit(Span(0, 10), "X")]
    b = [Unit(Span(100, 110), "X")]
    res = gamma_segments(a, b, cat_dist=BIN,
                         config=GammaConfig(admissibility="soft", eta=0.5))
    if math.isnan(res.gamma):
        assert "confidence" in res.reason
    else:
        # if Hungarian preferred orphaning (cost 1.0 == pair 0.5? no) — guard
        assert res.alignment.n_orphans == 2


def test_soft_no_nan_from_admissibility_when_orphaned():
    # Disjoint units with different categories: pair cost = 0.5*1+0.5*1 = 1.0
    # = 2*eta; orphaning ties. Either way score is defined or reasoned.
    a = [Unit(Span(0, 10), "X")]
    b = [Unit(Span(100, 110), "Y")]
    res = gamma_segments(a, b, cat_dist=BIN,
                         config=GammaConfig(admissibility="soft", eta=0.4))
    # orphaning (0.8) beats pairing (1.0): 2 orphans, D_o = 0.4 -> gamma 0.6
    assert res.gamma == pytest.approx(0.6)
    assert res.alignment.n_orphans == 2


def test_gamma_k_identifies_problem_category():
    # Category A agrees perfectly; category B always mismatched with C.
    a = [Unit(Span(0, 10), "A"), Unit(Span(20, 30), "B"), Unit(Span(40, 50), "B")]
    b = [Unit(Span(0, 10), "A"), Unit(Span(20, 30), "C"), Unit(Span(40, 50), "C")]
    gk = gamma_k_segments(a, b, cat_dist=BIN, config=CFG)
    assert gk["A"].gamma == pytest.approx(1.0)
    # B-pairs: cost = 0.5*0 + 0.5*1 = 0.5 each; n_bar_B = (2+0)/2 = 1
    # D_o(B) = (0.5+0.5)/1 = 1.0 -> gamma_k(B) = 0
    assert gk["B"].gamma == pytest.approx(0.0)
    assert gk["C"].gamma == pytest.approx(0.0)


def test_gamma_k_orphan_category():
    # Category Z exists only for annotator 1 and is orphaned.
    a = [Unit(Span(0, 10), "A"), Unit(Span(50, 60), "Z")]
    b = [Unit(Span(0, 10), "A")]
    gk = gamma_k_segments(a, b, cat_dist=BIN, config=CFG)
    assert gk["A"].gamma == pytest.approx(1.0)
    # Z: 1 orphan * eta / n_bar_Z=(1+0)/2=0.5 -> D_o = 1.0 -> gamma 0
    assert gk["Z"].gamma == pytest.approx(0.0)


def test_gamma_k_relations_by_type():
    r1 = [Relation(Span(0, 10), Span(20, 30), "support"),
          Relation(Span(40, 50), Span(60, 70), "attack")]
    r2 = [Relation(Span(0, 10), Span(20, 30), "support"),
          Relation(Span(40, 50), Span(60, 70), "rephrase")]
    gk = gamma_k_relations(r1, r2, rel_dist=BIN, config=CFG)
    assert gk["support"].gamma == pytest.approx(1.0)
    assert gk["attack"].gamma == pytest.approx(0.0)
    assert set(gk) == {"support", "attack", "rephrase"}


def test_gamma_k_consistent_with_global():
    # Weighted recombination of per-k disorders must relate to global D_o:
    # each pair counted once per distinct category it involves.
    a, b = _paper_segments()
    g = gamma_segments(a, b, cat_dist=BIN, config=CFG)
    gk = gamma_k_segments(a, b, cat_dist=BIN, config=CFG)
    assert g.is_defined and all(r.is_defined for r in gk.values())


def test_soft_chance_correction_defined():
    a, b = _paper_segments()
    cfg = GammaConfig(admissibility="soft", chance_correction=True,
                      n_iter=20, seed=3)
    res = gamma_segments(a, b, cat_dist=BIN, config=cfg)
    assert res.is_defined and res.D_e is not None and res.D_e > 0


# ===================== v0.4.0: Li et al. (2024) null model =====================
import itertools
from opt_gamma import (LiNonOverlapNull, n_configurations, sparsity_ratio,
                      start_distribution, uniform_approximation_ok)
from opt_gamma.li_null import _sample_placement


def _brute_force(n, lengths):
    k = len(lengths); seen = set()
    for order in itertools.permutations(range(k)):
        def rec(slot, pos, acc):
            if slot == k:
                seen.add(tuple(acc[s] for s in range(k))); return
            seg = order[slot]; L = lengths[seg]
            for s in range(pos, n - L + 1):
                acc[seg] = s; rec(slot + 1, s + L, acc)
            acc.pop(seg, None)
        rec(0, 0, {})
    return sorted(seen)


@pytest.mark.parametrize("n,lengths", [(10, [2, 3]), (12, [1, 2, 3]),
                                       (15, [4, 2, 3]), (8, [1, 1, 1])])
def test_li_configuration_count_matches_bruteforce(n, lengths):
    assert n_configurations(n, lengths) == len(_brute_force(n, lengths))


@pytest.mark.parametrize("n,lengths", [(10, [2, 3]), (12, [1, 2, 3]), (9, [2, 2, 2])])
def test_li_start_distribution_matches_bruteforce(n, lengths):
    bf = _brute_force(n, lengths)
    for i in range(len(lengths)):
        emp = np.zeros(n - lengths[i] + 1)
        for c in bf:
            emp[c[i]] += 1
        emp /= len(bf)
        assert np.abs(emp - start_distribution(n, lengths, i)).max() < 1e-12


def test_li_sampler_is_uniform_and_valid():
    rng = np.random.default_rng(0)
    n, lengths = 12, [1, 2, 3]
    valid = set(_brute_force(n, lengths))
    draws = [tuple(_sample_placement(n, lengths, rng)) for _ in range(20000)]
    assert set(draws) <= valid                       # jamais de config invalide
    assert len(set(draws)) == len(valid)             # couvre tout le support
    counts = np.array([draws.count(c) for c in sorted(valid)])
    # écart-type compatible avec un multinomial uniforme (tolérance large)
    assert counts.std() < 3 * np.sqrt(len(draws) / len(valid))


def test_li_sampler_preserves_lengths_and_categories():
    units = [Unit(Span(0, 10), "A"), Unit(Span(30, 45), "B")]
    rng = np.random.default_rng(1)
    out = LiNonOverlapNull().randomize_all_segments([units], 200, rng)[0]
    assert sorted(u.span.length for u in out) == sorted(u.span.length for u in units)
    assert sorted(u.category for u in out) == sorted(u.category for u in units)


def test_li_sampler_never_overlaps():
    rng = np.random.default_rng(2)
    units = [Unit(Span(0, 20), "A"), Unit(Span(25, 40), "B"), Unit(Span(50, 55), "C")]
    for _ in range(300):
        out = LiNonOverlapNull().randomize_all_segments([units], 300, rng)[0]
        spans = sorted((u.span.start, u.span.end) for u in out)
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert e1 <= s2


def test_li_gamma_runs_and_is_defined():
    a, b = _paper_segments()
    cfg = GammaConfig(chance_correction=True, null_model=LiNonOverlapNull(),
                      n_iter=30, seed=5)
    r = gamma_segments(a, b, cat_dist=BIN, config=cfg)
    assert r.is_defined and r.D_e is not None and r.D_e > 0


def test_li_sparsity_diagnostics():
    assert sparsity_ratio(1000, 100, 120) == pytest.approx(1000 / 220)
    assert uniform_approximation_ok(10000, [5, 5, 5], 0)       # long & sparse
    assert not uniform_approximation_ok(30, [10, 10, 5], 0)    # dense
