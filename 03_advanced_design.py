"""
Prescribed points, focus and exclusion
======================================

Compose a reactor design from prescribed runs, a focus region and an exclusion zone.

A pilot-scale continuous reactor is being mapped in temperature and
pressure. Two runs from an earlier campaign are already available and
must be reused as-is. The team also wants extra coverage around a
suspected optimum, and must not run the reactor at the high-T /
high-P corner for safety reasons.

Parameters
----------
- temperature (100-500 degC, 25-degree steps): the reactor operating
  window on an explicit grid so the placement points land on nodes.
- pressure (0.5-5.0 bar, 0.5-bar steps): a matching discrete grid
  from sub-atmospheric to the safe upper limit.

What to look at
---------------
- ``summary()``: the "Prescribed", "Focus" and "Optimised" rows show how
  the design is composed; two prescribed and four focus points
  should be present alongside the optimised runs.
- ``quality_report()``: scores are computed on the union of all included
  points, and every baseline design keeps the prescribed and focus
  points, so the percentiles show what the optimised runs added
  around them.
- ``plot('pairplot')``: the two prescribed runs should appear at the
  requested coordinates, the neighbourhood of (300, 2.0) should be
  visibly denser than elsewhere, and the excluded point (500, 5.0)
  should not appear.

Mergen features used
--------------------
- add_prescribed for runs already carried out.
- add_focus to concentrate additional samples around a critical
  operating point.
- add_exclusion to remove an unsafe operating point from the
  candidates.

Estimated runtime: under half a minute.
"""
from mergen import ParameterSpace, Sampler

# 1. Define a two-parameter engineering-style space on an explicit grid
#    so that prescribed / focus / exclusion points land on grid nodes.
space = ParameterSpace({
    'temperature': range(100, 501, 25),           # 100, 125, ..., 500
    'pressure':    [0.5, 1.0, 1.5, 2.0, 2.5,
                    3.0, 3.5, 4.0, 4.5, 5.0],
})

# 2. Configure the sampler with all four placement mechanisms.
sampler = Sampler(space)

# Prescribed points: two runs that must appear verbatim in the design.
sampler.add_prescribed([[200, 1.0], [400, 3.0]])

# Focus region: extra samples clustered around a critical operating
# point (T = 300 K, P = 2 bar). ``spread`` is the width of the
# neighbourhood in grid steps of each axis.
sampler.add_focus(point=[300, 2.0], spread=1.0, n_samples=4)

# Exclusion: the high-T / high-P corner is never proposed. Only this
# grid node is removed; a constraint would keep a whole region free.
sampler.add_exclusion(point=[500, 5.0])

# 3. Run and inspect.
sampler.set_design(n_samples=20)
result = sampler.run()

result.summary()
result.quality_report()
result.plot('pairplot', save=True)
result.to_csv('advanced_design.csv')
