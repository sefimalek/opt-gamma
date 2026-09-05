# opt-gamma

Optimized gamma agreement measure for two annotators, covering free
segmentation, complex category labels and explicit relations between
segments. Companion library of the paper *An Optimized Gamma Agreement
Measure*.

Compared with the classical gamma of Mathet, Widlöcher and Métivier (2015),
this variant uses a Jaccard positional dissimilarity (scale-invariant,
interpretable as shared content), an explicit admissibility threshold with a
soft orphan penalty, a relational coefficient for typed links between
segments, and an exact Hungarian reduction for the two-annotator case. It is
not a replacement for [pygamma-agreement](https://pypi.org/project/pygamma-agreement/),
which implements the classical n-annotator gamma, gamma-cat and gamma-k.

## Install

```bash
pip install opt-gamma
```

## Usage

```python
from opt_gamma import (Span, Unit, Relation, GammaConfig,
                    gamma_segments, gamma_relations, gamma_composite)

# spans are half-open [start, end); use Span.from_inclusive for inclusive data
a = [Unit(Span(100, 150), "Claim"), Unit(Span(160, 210), "Example")]
b = [Unit(Span(102, 152), "Claim"), Unit(Span(162, 212), "Explanation")]

cfg = GammaConfig(alpha=0.5, tau=0.5, eta=0.5)
typ = gamma_segments(a, b, config=cfg)          # 0.7115 on this example

r1 = [Relation(Span(100, 150), Span(160, 210), "Illustration")]
r2 = [Relation(Span(102, 152), Span(162, 212), "Illustration")]
rel = gamma_relations(r1, r2, config=cfg)        # 0.9615

gamma_composite(typ, rel, lambda_typ=0.7)        # 0.7865
```

Per-category and per-relation-type diagnostics (the gamma-k idea, extended to
relations):

```python
from opt_gamma import gamma_k_segments, gamma_k_relations
gamma_k_segments(a, b, config=cfg)
gamma_k_relations(r1, r2, config=cfg)
```

Chance correction, with adaptive sampling:

```python
from opt_gamma import CircularShiftNull, LiNonOverlapNull
cfg = GammaConfig(chance_correction=True, null_model=CircularShiftNull(),
                  n_iter="auto", seed=42)
```

## Choices you have to make (and report)

| Parameter | Meaning | Default |
|---|---|---|
| `tau` | max Jaccard *distance* for a pair to be alignable (hard mode) | 0.5 |
| `eta` | orphan penalty, in [0, 1]; must be > 0 | 0.5 |
| `admissibility` | `"hard"` (threshold) or `"soft"` (confidence weighting à la Mathet 2017) | `"hard"` |
| `on_no_admissible` | items with units on both sides but no alignable pair: `"orphan_all"` keeps them in corpus averages, `"nan"` excludes them and inflates the mean | `"orphan_all"` |
| `null_model` | `CircularShiftNull` (Mathet 2015, §5.2.1) or `LiNonOverlapNull` (Li et al. 2024) | `CircularShiftNull` |

The direction of the tau effect on corpus averages depends on
`on_no_admissible`; see the paper. The Li null model is an exact uniform
sampler over non-overlapping placements (their closed form assumes an
additive measure, which an alignment-based disorder is not); the sampler and
the analytical start distribution are both tested against exhaustive
enumeration.

## Corpus example

`examples/french_argumentation.py` runs the full pipeline of the paper's
corpus study, with a CLI exposing every methodological switch.

## Tests

```bash
python -m pytest
```

The suite pins the paper's worked example, edge cases (orphans, empty sides,
aggregation policies, matrix validation) and the Li model against brute-force
enumeration.

## License

MIT. See CITATION.cff for how to cite.
