"""
04_choosing_criteria.py
=======================
A modelling team is designing a purely numeric 3-factor computer
experiment whose output will feed a Gaussian-process surrogate. Rather
than picking a space-filling criterion by habit, they let Mergen
compare every criterion compatible with the space and rank the
resulting designs on metrics that actually matter for a GP surrogate,
not on the raw optimisation score.

Parameters
----------
- factor_a, factor_b, factor_c (0.0-1.0, continuous): three generic
  normalised inputs to a numerical model; the ranges are unit-scaled
  so the comparison is not tied to any particular unit system.

What to look at
---------------
- comparison_table.md (saved to disk): columns are percentile ranks
  against a single shared Monte Carlo baseline, not raw criterion
  scores. Raw scores from different criteria are not comparable on
  their own scale, so ranking by percentile is the only sound way to
  put them side by side. The winning row is flagged.
- min_distance is the primary ranking metric here because a GP
  surrogate's prediction variance grows where design points are far
  apart; max_abs_correlation is secondary, since correlated inputs
  make the individual effect of each factor harder to separate.
- The winning design's saved pairplot: an even spread in every panel
  is the visual signature of a good space-filling design.
- winner_design.csv: the winning benchmark rows, ready for the
  downstream modelling step.

Mergen features used
--------------------
- Sampler.compare(criteria=None, ...): criteria=None resolves to
  every criterion compatible with the space; since this space has no
  nominal factor, the QQ-family criteria are automatically excluded.
- ComparisonResult: .table, .best, .best_result.
- comparison.table saved to outputs/comparison_table.md; the winning
  design saved as a pairplot (PNG) and a CSV. quality_report() stays
  in the terminal.

Estimated runtime: a few minutes (compare() runs one optimisation per
criterion; the optimiser is tuned down from its default restart count
to keep the sweep quick for a demonstration).
"""
from mergen import ParameterSpace, Sampler

# 1. Define a purely numeric parameter space.
space = ParameterSpace({
    'factor_a': ('continuous', 0.0, 1.0),
    'factor_b': ('continuous', 0.0, 1.0),
    'factor_c': ('continuous', 0.0, 1.0),
})

# 2. Configure the sampler. The optimiser is tuned down from its
#    default restart count so the five-criterion sweep below finishes
#    in a few minutes; increase n_restarts / max_iter for production
#    use of a single criterion.
sampler = Sampler(space)
sampler.set_design(n_samples=30)
sampler.set_optimizer('sa', n_restarts=2, max_iter=400)
comparison = sampler.compare(priority=('min_distance', 'max_abs_correlation'))

# 3. Save the comparison table itself: which criterion won, and by
#    how much on each quality metric.
import os
os.makedirs('outputs', exist_ok=True)
comparison.table.to_markdown('outputs/comparison_table.md', index=False)

# 4. Inspect and save the winning design.
best = comparison.best_result
best.quality_report()
best.plot('pairplot', save=True)
best.to_csv('winner_design.csv')
