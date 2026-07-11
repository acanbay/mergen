"""
12_cfd_engineering.py
=====================
A CFD study sweeps an aerofoil across four control factors: angle of
attack, inlet velocity, a geometry ratio, and the turbulence model
(a nominal switch). Each CFD run is expensive, so a tight space-filling
design matters, and comparing criteria before committing pays off. The
flow observables (lift, drag, separation point) are computed downstream
by the CFD solver, not by Mergen.

Because the space contains a nominal factor (the turbulence model),
compare() automatically restricts itself to the criteria that support
nominal factors, so the comparison stays valid.

Parameters
----------
- angle_of_attack (0-15 deg, 1-degree steps): the pre-stall range,
  swept in whole degrees.
- velocity (10.0-60.0 m/s, continuous, rounded to 1 decimal): the
  inlet speed range of interest.
- geometry_ratio (0.10-0.30, continuous, rounded to 2 decimals): a
  normalised thickness/chord-style ratio.
- turbulence_model (nominal: 'k-epsilon', 'k-omega', 'sst'): three
  standard closures with no intrinsic ordering.

What to look at
---------------
- comparison_table.md (saved): the criteria compatible with this mixed
  space, ranked by percentile against a shared Monte Carlo baseline;
  the winning row is flagged.
- best_result.quality_report() (printed): the winning design's quality.
- The saved pairplot: even coverage across the numeric factors, with
  all three turbulence-model levels visited.
- cfd_runs.csv: the run list handed to the CFD solver.

Mergen features used
--------------------
- A nominal factor in a mostly numeric space.
- Sampler.compare(): on a mixed space, criteria=None auto-selects the
  nominal-supporting criteria, so the comparison is valid. Each
  combination is optimised n_repeats times (default 5) from
  reproducible seeds and ranked on the mean metric percentile via a
  Pareto/Utopia rule.
- ComparisonResult.best_result and its saved table.

Estimated runtime: a few minutes on one core. On a multi-core machine
pass n_jobs=-1 to compare() to run the repeats in parallel (same
result, faster); leave it at the default to stay single-core.
"""
from mergen import ParameterSpace, Sampler

# 1. Define the mixed CFD control space.
space = ParameterSpace({
    'angle_of_attack':  range(0, 16, 1),                             # degrees
    'velocity':         ('continuous', 10.0, 60.0, {'round': 1}),    # m/s
    'geometry_ratio':   ('continuous', 0.10, 0.30, {'round': 2}),
    'turbulence_model': ('nominal', ['k-epsilon', 'k-omega', 'sst']),
})

# 2. Compare the criteria compatible with this mixed space. The
#    optimiser is tuned down so the sweep is quick for a demonstration.
sampler = Sampler(space)
sampler.set_design(n_samples=25)
sampler.set_optimizer('sa', n_restarts=2, max_iter=400)
comparison = sampler.compare(
    n_jobs=1,  # one core; set n_jobs=-1 to use all cores (same result)
)

# 3. Save the ranking table.
import os
os.makedirs('outputs', exist_ok=True)
comparison.table.to_markdown('outputs/comparison_table.md', index=False)

# 4. Inspect and save the winning design.
best = comparison.best_result
best.quality_report()
best.plot('pairplot', save=True)
best.to_csv('cfd_runs.csv')
