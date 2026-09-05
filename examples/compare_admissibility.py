# -*- coding: utf-8 -*-
"""Hard threshold vs soft confidence weighting, on synthetic annotations.

A reference annotator produces n_units segments; the second annotator is a
noisy copy (boundary shifts and shrinks proportional to a magnitude in
[0, 1], plus deletions and label confusions). For each magnitude we report
the mean gamma under both admissibility modes. Chance correction is off by
default so the two mechanisms are compared on the same ground; use --chance
to enable it.

    python compare_admissibility.py [--texts 200] [--units 12] [--chance]
"""
from __future__ import annotations

import argparse

import numpy as np

from opt_gamma import CategoryDistance, GammaConfig, Span, Unit, gamma_segments

CATS = ["A", "B", "C", "D"]


def make_pair(rng, n_units, magnitude, p_del=0.1, p_confuse=0.15):
    """One synthetic (annotator1, annotator2) pair at a given noise level."""
    units1, units2 = [], []
    pos = 0
    for _ in range(n_units):
        pos += int(rng.integers(5, 40))            # gap
        length = int(rng.integers(15, 120))
        cat = CATS[int(rng.integers(0, len(CATS)))]
        units1.append(Unit(Span(pos, pos + length), cat))

        if rng.random() >= p_del * magnitude * 2:  # deletions grow with noise
            shift = int(magnitude * length * rng.uniform(-1, 1))
            shrink = int(magnitude * length * rng.uniform(0, 0.8))
            s = pos + shift
            e = max(s + 1, pos + length + shift - shrink)
            c = cat
            if rng.random() < p_confuse * magnitude * 2:
                c = CATS[int(rng.integers(0, len(CATS)))]
            units2.append(Unit(Span(s, e), c))
        pos += length
    return units1, units2


def run(n_texts, n_units, chance, seed=0):
    rng = np.random.default_rng(seed)
    bin_dist = CategoryDistance.binary()
    print(f"{'magn.':>6} | {'hard γ':>8} {'undef':>6} {'orph':>5} | "
          f"{'soft γ':>8} {'undef':>6} {'orph':>5} | {'Δ(h-s)':>7}")
    print("-" * 68)
    for magnitude in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        rows = {"hard": [], "soft": []}
        undef = {"hard": 0, "soft": 0}
        orph = {"hard": 0, "soft": 0}
        for t in range(n_texts):
            u1, u2 = make_pair(rng, n_units, magnitude)
            for mode in ("hard", "soft"):
                cfg = GammaConfig(admissibility=mode, chance_correction=chance,
                                  n_iter=30, seed=seed + t,
                                  on_no_admissible="orphan_all")
                r = gamma_segments(u1, u2, cat_dist=bin_dist, config=cfg)
                if r.is_defined:
                    rows[mode].append(r.gamma)
                    orph[mode] += r.alignment.n_orphans
                else:
                    undef[mode] += 1
        h = float(np.mean(rows["hard"])) if rows["hard"] else float("nan")
        s = float(np.mean(rows["soft"])) if rows["soft"] else float("nan")
        print(f"{magnitude:>6.1f} | {h:>8.3f} {undef['hard']:>6} {orph['hard']:>5} | "
              f"{s:>8.3f} {undef['soft']:>6} {orph['soft']:>5} | {h - s:>7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts", type=int, default=100)
    ap.add_argument("--units", type=int, default=12)
    ap.add_argument("--chance", action="store_true")
    args = ap.parse_args()
    run(args.texts, args.units, args.chance)
