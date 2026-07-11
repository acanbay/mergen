"""
01_quickstart.py
================
A materials-science lab is preparing a thermal deposition study on a
multilayer coating. Reactor time is limited, so the goal is to pick a
small set of runs that jointly cover the operating envelope as evenly
as possible, ready to feed a first surrogate model.

Parameters
----------
- temperature (100-500 degC, 10-degree steps): sweep the deposition
  window, from the lowest useful onset to the upper safe limit of the
  chamber.
- pressure (0.5-5.0 bar, continuous): span sub-atmospheric to mildly
  pressurised conditions typical of chemical vapour deposition.
- n_layers (2-10, integer): explore thin stacks up to the layer count
  the process can build in one campaign.

What to look at
---------------
- summary(): the "Optimised" and "Validation" counts confirm the
  design was built and that a hold-out set was reserved automatically.
- quality_report(): percentiles against the Monte Carlo baseline
  should be well above 90 for min_distance and near 0 for
  max_absolute_correlation; that pair is the primary evidence of a
  good space-filling design.
- plot('pairplot'): every 2D projection should look evenly
  populated, with no clustered corners or empty bands.

Mergen features used
--------------------
- ParameterSpace with three factor types (discrete, continuous,
  integer).
- Sampler.set_design() and Sampler.run() at their defaults, so the
  example stays minimal and shows the shape of the workflow with no
  configuration to reason about.

Estimated runtime: a few seconds.
"""
from mergen import ParameterSpace, Sampler

# 1. Define the parameter space
space = ParameterSpace({
    'temperature': range(100, 500, 10),        # discrete
    'pressure':    ('continuous', 0.5, 5.0),   # continuous grid
    'n_layers':    ('integer', 2, 10),         # integer grid
})

# 2. Configure and run the sampler
sampler = Sampler(space)
sampler.set_design()
result = sampler.run()

# 3. Inspect, visualise, export
result.summary()
result.quality_report()
result.plot('pairplot', save=True)
result.to_csv('design.csv')
