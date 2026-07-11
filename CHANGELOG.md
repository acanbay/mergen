# Changelog

All notable changes to mergen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - Unreleased

Initial release.

### Added

- `ParameterSpace` — N-dimensional parameter space supporting discrete,
  continuous (linear/log), and integer (linear/log) parameter types,
  with a memory-efficient bijective grid representation using
  mixed-radix indexing.
- `Sampler` — space-filling design generator with prescribed points,
  focus regions, and exclusion zones, and Kennard-Stone validation set
  construction.
- Three optimisers: Simulated Annealing (SA) with Iterated Local Search
  restarts, Stochastic Coordinate Exchange (SCE), and Enhanced
  Stochastic Evolutionary (ESE).
- Seven space-filling criteria: $\mathrm{UMaxPro}$, $\mathrm{MaxPro}$,
  $\phi_p$, $\mathrm{CD}_2$, $\mathrm{SL}_2$, and the
  qualitative-quantitative pair $\mathrm{MaxPro}_{\mathrm{QQ}}$ and
  $\mathrm{QQD}$.
- Six quality metrics (minimum distance, minimax distance,
  maximum absolute correlation, 2D projection CD2, coefficient of
  variation of distances, mean distance), with a runtime Monte Carlo
  quality assessment reporting each metric's percentile rank against a
  random baseline.
- `Sampler.compare()` — compares criterion and algorithm combinations,
  optimising each several times from independent, reproducible seeds
  (`n_repeats`) and ranking them with a Pareto-frontier and Utopia-point
  rule over the priority metrics, with optional parallelism across cores
  (`n_jobs`).
- `Sampler.run(algorithm=[...])` — runs several algorithms for one
  criterion and reports the best, optionally in parallel.
- Sequential design tools: `augment`, `complement`, `from_dataframe`,
  `subsample`, `n_samples_recommendation`.
- Export formats: CSV, JSON, Markdown, LaTeX, HTML, Excel.
- Visualisation: pairplot, 1D distribution, 2D scatter, pairwise
  distance distribution, correlation heatmap, quality-metrics chart, and
  a per-algorithm comparison chart.
- Convenience properties: `space.centroid`, `space.corners`,
  `space.bounds_as_dict`, `space.random_point()`.
- Continuous integration via GitHub Actions (Python 3.9-3.12).
