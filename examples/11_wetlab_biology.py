"""
Wet-lab: an assay with a nominal factor
=======================================

Design an enzyme-activity assay whose buffer type is nominal, scored with the QQ-aware ``maxproqq``.

An enzyme-activity assay is being optimised in a wet lab across four
factors: pH, incubation temperature, buffer type, and substrate
concentration. Buffer type is a nominal (unordered categorical) factor,
which changes how the design must be scored. A held-out validation set
is reserved so the fitted response can be checked on unseen conditions.

Because the space contains a nominal factor, the design is scored with
maxproqq. A QQ-type criterion handles the mix of numeric and
categorical factors correctly; a purely numeric criterion (``phi_p``,
MaxPro, ...) would treat the buffer labels as if they had a numeric
distance, which is meaningless for unordered categories and would
distort the design.

Parameters
----------
- pH (5.0-9.0, continuous, 0.1 steps): the physiological range over
  which the enzyme is active, at the precision a pH meter can be set to.
- temperature (25-45 degC, 5-degree steps): the assay incubator's
  discrete temperature settings.
- buffer (nominal: 'phosphate', 'tris', 'acetate'): three common
  buffers with no intrinsic ordering.
- substrate (integer, 1-10 mM): substrate concentration in whole
  millimolar steps.

What to look at
---------------
- ``summary()`` and ``quality_report()``: the design covers the mixed factor
  space; the percentiles remain meaningful because ``maxproqq`` scores the
  numeric and nominal factors appropriately.
- The saved pairplot: the buffer panels should show all three levels
  visited, and the numeric factors should be evenly spread.
- ``enzyme_design.csv``: the run list for the bench.

Mergen features used
--------------------
- A nominal factor alongside continuous / discrete / integer factors.
- Per-parameter rounding on the continuous pH axis.
- criteria='``maxproqq``' as the correct choice for a space containing a
  nominal factor.
- A validation hold-out via ``set_design(n_validation=...)``.

Estimated runtime: a few seconds to a minute.
"""
from mergen import ParameterSpace, Sampler

# 1. Define the mixed-factor assay space.
space = ParameterSpace({
    'pH':          ('continuous', 5.0, 9.0, {'resolution': 41, 'round': 1}),
    'temperature': range(25, 46, 5),                       # 25, 30, ..., 45
    'buffer':      ('nominal', ['phosphate', 'tris', 'acetate']),
    'substrate':   ('integer', 1, 10),                     # mM
})

# 2. Build the design with maxproqq (correct for a nominal factor) and
#    reserve a validation set for the assay.
sampler = Sampler(space)
sampler.set_design(n_samples=24, n_validation=6)
result = sampler.run(criteria='maxproqq')

# 3. Inspect and save the bench run list.
result.summary()
result.quality_report()
result.plot('pairplot', save=True)
result.to_csv('enzyme_design.csv')
