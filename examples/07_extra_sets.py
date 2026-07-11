"""
07_extra_sets.py
================
A machine-learning surrogate workflow needs three disjoint sets of
points drawn from one input space: a space-filling training design, an
automatic validation set, and an external held-out test set whose
points the team has chosen by hand. This example shows how Mergen keeps
all three separate, with no point shared between them.

Parameters
----------
- x1, x2, x3 (discrete grids): three inputs defined on explicit grids
  so the hand-picked test points land exactly on grid nodes. x1 and x2
  are integer ladders; x3 is a stepped list.

What to look at
---------------
- summary(): three distinct groups are reported, 'Optimised' (the
  training design), 'Validation' (the automatic hold-out) and 'test'
  (the user set); their counts should not overlap.
- The saved pairplot: the training points, validation points and the
  user 'test' points appear in three different colours, with the test
  points sitting exactly where they were prescribed and never reused by
  the training design.
- design.csv: all three groups in one file, distinguished by the
  point_type column, ready to feed a training / validation / test split
  downstream.

Mergen features used
--------------------
- Sampler.add_set('test', [...], color=...): attach an external,
  user-chosen point set with its own colour; its nodes are reserved so
  the optimiser cannot select them.
- Sampler.set_design(n_validation=...): request the automatic
  validation hold-out alongside the training design.

Estimated runtime: a few seconds to a minute.
"""
from mergen import ParameterSpace, Sampler

# 1. Define the input space on discrete grids so the hand-picked test
#    points are exact grid nodes.
space = ParameterSpace({
    'x1': range(0, 101, 10),          # 0, 10, ..., 100
    'x2': range(0, 101, 10),
    'x3': [0.0, 0.25, 0.5, 0.75, 1.0],
})

# 2. Attach a hand-picked external test set in its own colour.
sampler = Sampler(space)
sampler.add_set('test',
                [[0, 100, 0.0], [100, 0, 1.0], [50, 50, 0.5]],
                color='#9b5de5')

# 3. Ask for an automatic validation hold-out alongside the training
#    design, then build all groups in one run.
sampler.set_design(n_samples=25, n_validation=5)
result = sampler.run()

# 4. Confirm the three groups are separate, then save.
result.summary()
result.plot('pairplot', save=True)
result.to_csv('design.csv')
