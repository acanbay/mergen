"""
Mixed parameter types and a constraint
======================================

Mix discrete, continuous, integer, nominal and ordinal factors under one constraint.

A catalysis group is scoping a new packed-column reaction and needs
a first screening design. The chemistry involves a continuous flow
rate, a discrete temperature ladder set by the heater controller, an
integer number of stages, a choice of catalyst material, and a
process grade tier. The whole factor palette lives in a single space,
and a physical column limits the total loading, so a feasibility
constraint is applied before optimisation.

Parameters
----------
- flow_rate (0.1-10.0 mL/min, continuous, 25-level grid, rounded to
  2 decimals): span the operating envelope of the pump; a tighter grid
  gives more resolution along this rate-controlling factor, and the
  rounding keeps the node values at a precision the pump can actually
  be set to.
- temperature (20-100 degC, 5-degree steps): the heater exposes only
  a discrete ladder, so it is entered as an explicit list.
- n_stages (1-20, integer): column length options offered by the
  hardware.
- catalyst ('A', 'B', 'C', nominal): materials with no intrinsic
  ordering; the criterion treats them as unordered labels.
- grade ('low', 'med', 'high', ordinal): quality tiers with a
  meaningful order, so distance along this axis carries information.
- Constraint: flow_rate + n_stages <= 20 discards operating points
  the column cannot support.

What to look at
---------------
- ``summary()``: the "Candidates" count reflects the constraint (fewer
  than the unconstrained Cartesian product); the design size and any
  automatic validation split are also reported here.
- ``quality_report()``: min_distance and max_absolute_correlation
  percentiles against the Monte Carlo baseline show that the mixed
  factor space is well covered despite the categorical columns.
- ``plot('pairplot')``: panels involving the nominal catalyst factor
  should show all three levels visited; panels between numeric
  factors should look evenly spread.

Mergen features used
--------------------
- ``ParameterSpace`` accepting all five factor types simultaneously.
- Per-parameter resolution override via the options dictionary on
  the specification tuple.
- add_constraint to filter the candidate pool by a feasibility
  predicate.
- ``criteria='maxproqq'``, the default-safe choice when the space
  contains any nominal factor.

Estimated runtime: a few seconds.
"""
from mergen import ParameterSpace, Sampler

# 1. Define a mixed-type parameter space.
#    A per-parameter resolution override is passed as an options dict
#    on the specification tuple (see 'flow_rate' below).
space = ParameterSpace({
    'flow_rate':   ('continuous', 0.1, 10.0, {'resolution': 25, 'round': 2}),  # continuous, 25-level grid
    'temperature': range(20, 101, 5),                              # discrete list
    'n_stages':    ('integer', 1, 20),                             # integer interval
    'catalyst':    ('nominal', ['A', 'B', 'C']),                   # unordered labels
    'grade':       ('ordinal', ['low', 'med', 'high']),            # ordered labels
})

# 2. Feasibility constraint: only keep candidates below the diagonal
#    of the flow-rate / stage-count plane.
space.add_constraint(lambda p: p['flow_rate'] + p['n_stages'] <= 20)

# 3. Run the sampler. maxproqq is the default-safe choice when the
#    space contains any nominal factor.
sampler = Sampler(space)
sampler.set_design(n_samples=25)
result = sampler.run(criteria='maxproqq', seed=44)

# 4. Inspect and export.
result.summary()
result.quality_report()
result.plot('pairplot', save=True)
result.to_csv('parameter_types.csv')
