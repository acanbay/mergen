"""
Plot test script for Mergen.
Runs the sampler once and renders every plot type, with the candidate
pool both off (default) and on, so the pairplot dedupe/alpha fix can be
inspected.
"""
import numpy as np
from mergen import ParameterSpace, Sampler

# 3 parameters on a 20-level grid (fast; pool = 8000 candidates)
space = ParameterSpace({
    'x': ('continuous', 0.0, 1.0, {'resolution': 20}),
    'y': ('continuous', 0.0, 1.0, {'resolution': 20}),
    'z': ('continuous', 0.0, 1.0, {'resolution': 20}),
})

sampler = Sampler(space)
sampler.set_design(n_samples=15, n_validation=3)
sampler.set_optimizer('sa', n_restarts=1, max_iter=1000)
result = sampler.run(criteria='cd2', algorithm='sa', seed=44)

# --- Pairplot: pool OFF (new default) then ON (dedupe + density alpha) ---
# result.plot('pairplot', show=True, save=False)
# result.plot('pairplot', show_pool=True, show=True, save=False)

# --- Other plot types ---
# result.plot('1d',        show=True, save=False)
# result.plot('2d', params=['x', 'y'], show_pool=True, show=True, save=False)
# result.plot('distances', show=True, save=False)
result.plot('quality',   show=True, save=False)