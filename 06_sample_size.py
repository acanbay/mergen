"""
How large should a design be?
=============================

Contrast a default-size and a doubled design of the same study to see what extra runs buy.

The same four-factor study is planned two ways: once under a hard
experimental budget that fixes the number of runs, and once at the
default size Mergen recommends. Placing the two designs side by side
shows what a larger budget actually buys, so the decision can be made
before any real experiment is spent.

The honest comparison here is coverage, not a quality percentile. Each
design is optimised to fill the space as well as it can for its size,
so both score well against a random baseline; a small design is not a
"bad" design. What a small design cannot do is occupy as much of the
space: with fewer points, larger gaps are unavoidable. That gap in
coverage, visible in the pairplot, is the real cost of a tight budget.

Parameters
----------
- factor_a, factor_b, factor_c, factor_d (0.0-1.0, continuous,
  20-level grid, rounded to 3 decimals): four generic normalised
  inputs. With four factors the default sample size is 10*d = 40.

What to look at
---------------
- The two saved pairplots, compared directly: the 15-run design leaves
  visibly larger empty regions in several 2D projections, while the
  40-run design places points into those gaps. This difference in
  coverage, not a difference in percentile score, is what a larger
  budget buys.
- ``quality_report()`` for both runs (printed): note that both designs
  score well for their size. A high percentile confirms each design is
  well-spread relative to random designs of the same count; it does
  not mean 15 runs cover the space as fully as 40. Coverage and
  per-size quality are different questions, and the pairplots answer
  the coverage one.

Mergen features used
--------------------
- ``Sampler.set_design(n_samples=...)``: an explicit fixed budget in the
  first run, versus omitting n_samples in the second so the 10*d
  default applies.
- Two independent designs from the same parameter space, compared on
  coverage (pairplots) and per-size quality (quality_report).
- ``Sampler.set_optimizer()``: a modest, shared compute budget so the two
  runs finish quickly for a demonstration.

Estimated runtime: a minute or two (two designs).
"""
from mergen import ParameterSpace, Sampler

# 1. Define a four-factor numeric space. With d = 4 the default sample
#    size is 10*d = 40; a rule of thumb from the computer-experiments
#    literature that a design should scale with dimensionality.
space = ParameterSpace({
    'factor_a': ('continuous', 0.0, 1.0, {'resolution': 20, 'round': 3}),
    'factor_b': ('continuous', 0.0, 1.0, {'resolution': 20, 'round': 3}),
    'factor_c': ('continuous', 0.0, 1.0, {'resolution': 20, 'round': 3}),
    'factor_d': ('continuous', 0.0, 1.0, {'resolution': 20, 'round': 3}),
})

# 2. First study: a hard budget of only 15 runs.
budget_sampler = Sampler(space)
budget_sampler.set_design(n_samples=15)
budget_sampler.set_optimizer('sa', n_restarts=2, max_iter=300)
budget_design = budget_sampler.run()

# 3. Second study: let Mergen size the design (the 10*d default).
default_sampler = Sampler(space)
default_sampler.set_optimizer('sa', n_restarts=2, max_iter=300)
default_design = default_sampler.run()

# 4. Compare per-size quality (printed), then the coverage difference
#    in the pairplots, which is the real cost of a smaller budget.
budget_design.quality_report()
default_design.quality_report()
budget_design.plot('pairplot', save=True)
default_design.plot('pairplot', save=True)
