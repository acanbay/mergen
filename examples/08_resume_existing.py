"""
08_resume_existing.py
=====================
A design from a previous campaign already exists and must be reused
exactly as it was run, with Mergen adding only a fresh validation set
around it. No point of the original design is moved or re-optimised;
the optimiser is skipped entirely. This example shows both the default
way to load a design and a variant with a custom label and colour.

Parameters
----------
- x1, x2 (discrete grids): two inputs on explicit grids so the loaded
  campaign points land exactly on grid nodes. Because the design is
  loaded rather than optimised, the choice of criterion and algorithm
  is irrelevant here.

What to look at
---------------
- summary(): the loaded points appear as their own group ('Existing'
  by default, or the custom label in the second variant) alongside the
  automatically generated 'Validation' set; no optimisation step runs.
- The saved pairplot: the loaded design is shown in its group colour,
  with the validation points placed around it in their own colour.
- The guard errors noted in the code: load_design deliberately refuses
  to combine with n_samples or add_prescribed, because the design is
  fixed; growing a design is the job of mergen.sequential.extend
  (see example 09).

Mergen features used
--------------------
- Sampler.load_design(points): reuse an existing design; run() then
  skips optimisation and only builds the validation set.
- load_design(points, name=..., color=...): a second variant with a
  custom label and colour instead of the default 'Existing' / blue.

Estimated runtime: a few seconds.
"""
import pandas as pd
from mergen import ParameterSpace, Sampler

# 1. Define the space on discrete grids so the loaded points are nodes.
space = ParameterSpace({
    'x1': range(0, 101, 10),          # 0, 10, ..., 100
    'x2': range(0, 101, 10),
})

# 2. A design already run in a previous campaign (here, a small array).
previous_campaign = [[0, 0], [50, 50], [100, 100], [0, 100], [100, 0]]

# 3. Load it as-is and add only a validation set. run() skips
#    optimisation because the design is fixed.
sampler = Sampler(space)
sampler.load_design(previous_campaign)
sampler.set_design(n_validation=4)
result = sampler.run()
result.summary()
result.plot('pairplot', save=True)

# 4. Same idea with a custom label and colour, loading from a DataFrame.
prev_df = pd.DataFrame(previous_campaign, columns=['x1', 'x2'])
labelled = Sampler(space)
labelled.load_design(prev_df, name='campaign_2024', color='#ff8800')
labelled.set_design(n_validation=4)
labelled_result = labelled.run()
labelled_result.summary()
labelled_result.plot('pairplot', save=True)
