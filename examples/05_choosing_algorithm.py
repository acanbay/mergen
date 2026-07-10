"""
05_choosing_algorithm.py
========================
A team has already settled on phi_p (maximin separation) as the right
criterion for a mid-sized numeric design, but wants to know which of
Mergen's three optimisers gets there fastest and best under a limited
compute budget. Running the same criterion with multiple algorithms
produces a direct, self-explanatory bar chart comparing their scores.

Parameters
----------
- factor_a, factor_b, factor_c, factor_d (0.0-1.0, continuous): four
  generic normalised inputs, large enough that the choice of
  optimiser starts to matter.

What to look at
---------------
- comparison_1.png (saved plot): a bar chart with one bar per
  algorithm, the phi_p score on the y-axis (lower is better) and the
  elapsed time annotated above each bar; the best algorithm is
  highlighted. This is the direct answer to "which optimiser wins",
  unlike a pairplot, which only shows point coverage, not the
  optimisation outcome.
- The printed summary(): confirms all three runs share the same
  criterion, so their scores are directly comparable (unlike
  comparing different criteria, which requires percentile ranking).
- distances_1.png (saved plot): for a maximin-style criterion, the
  pairwise-distance distribution of the winning design, pushed away
  from zero, is the natural coverage check.

Mergen features used
--------------------
- Sampler.run(algorithm=[...]): the same criterion optimised by every
  named algorithm in one call; the result carries algorithm_results
  and best_algorithm.
- result.plot('comparison'): the dedicated bar chart for this
  same-criterion, multi-algorithm case.
- Sampler.set_optimizer(): called once per algorithm to fix a shared,
  modest compute budget, so the comparison reflects a fixed cost
  rather than each optimiser's own (much heavier) defaults.

Estimated runtime: a few minutes (three optimisations under a shared
compute budget).
"""
from mergen import ParameterSpace, Sampler

# 1. Define a four-factor numeric space. The grid resolution is kept
#    moderate so the three-way optimiser comparison below finishes in
#    a few minutes; a finer resolution is fine for production use of
#    a single, chosen algorithm.
space = ParameterSpace({
    'factor_a': ('continuous', 0.0, 1.0, {'resolution': 20}),
    'factor_b': ('continuous', 0.0, 1.0, {'resolution': 20}),
    'factor_c': ('continuous', 0.0, 1.0, {'resolution': 20}),
    'factor_d': ('continuous', 0.0, 1.0, {'resolution': 20}),
})

# 2. Fix a shared, modest compute budget for each optimiser, then run
#    all three on the same criterion in a single call.
sampler = Sampler(space)
sampler.set_design(n_samples=20)
sampler.set_optimizer('sa', n_restarts=2, max_iter=300)
sampler.set_optimizer('sce', n_restarts=2, max_iter=300)
sampler.set_optimizer('ese', M=30, J=15)
result = sampler.run(criteria='phi_p', algorithm=['sa', 'sce', 'ese'])

# 3. Inspect and save the outcome.
result.summary()
result.plot('comparison', save=True)
result.plot('distances', save=True)
