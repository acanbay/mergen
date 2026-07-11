"""
15_beamline_optics.py
=====================
A beam-line tuning study varies the currents of three focusing magnets
and the apertures of two collimators to map how the optics respond.
Each configuration is expensive to simulate or measure, so a compact
space-filling design over the control settings is chosen. The beam
observables (emittance, transmission, spot size) are computed
downstream, not by Mergen.

The design is scored with phi_p (maximin), which maximises the minimum
separation between configurations — the natural objective when each
setting is costly and near-duplicate settings would waste beam time.

Parameters (stepped grids, in engineering units)
----------
- quad1_current, quad2_current, quad3_current (50-150 A, 10 A steps):
  the focusing-magnet currents, at the resolution the supplies are set.
- collimator1_aperture, collimator2_aperture (2-20 mm, 2 mm steps):
  the collimator gaps.

What to look at
---------------
- summary() and quality_report(): the design covers the five-factor
  control space; the min_distance percentile confirms the maximin
  objective pushed the configurations apart.
- The saved distances plot: for a maximin design the pairwise-distance
  distribution should sit well away from zero, meaning no two settings
  are near-duplicates.
- beamline_runs.csv: the control-room run list.

Mergen features used
--------------------
- Five stepped numeric factors in engineering units.
- criteria='phi_p' as the maximin choice for maximally separated,
  expensive configurations.
- A validation hold-out via set_design(n_validation=...).
- Sampler.set_optimizer(): a modest compute budget for a quick demo.
- plot('distances') as the natural view for a maximin design.

Estimated runtime: a few seconds to a minute.
"""
from mergen import ParameterSpace, Sampler

# 1. Define the five-factor beam-line control space on stepped grids.
space = ParameterSpace({
    'quad1_current':        range(50, 151, 10),     # A
    'quad2_current':        range(50, 151, 10),
    'quad3_current':        range(50, 151, 10),
    'collimator1_aperture': range(2, 21, 2),        # mm
    'collimator2_aperture': range(2, 21, 2),
})

# 2. Build a maximin (phi_p) design with a validation hold-out.
sampler = Sampler(space)
sampler.set_design(n_samples=25, n_validation=5)
sampler.set_optimizer('sa', n_restarts=1, max_iter=300)
result = sampler.run(criteria='phi_p')

# 3. Inspect and save the control-room run list.
result.summary()
result.quality_report()
result.plot('distances', save=True)
result.to_csv('beamline_runs.csv')
