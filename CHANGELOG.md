# Changelog

All notable changes to mergen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - Unreleased

Initial release.

### Added

- `ParameterSpace` — N-dimensional parameter space supporting discrete,
  continuous (linear/log), integer (linear/log), nominal, and ordinal
  parameter types, with feasibility constraints and a memory-efficient
  bijective grid representation using mixed-radix indexing.
- `Sampler` — space-filling design generator with prescribed points,
  focus regions, exclusion zones, named extra point sets, resuming from
  an existing design, per-dimension weights, and Kennard-Stone
  validation set construction.
- Three optimisers: Simulated Annealing (SA) with Iterated Local Search
  restarts, Stochastic Coordinate Exchange (SCE), and Enhanced
  Stochastic Evolutionary (ESE).
- Seven space-filling criteria: $\mathrm{UMaxPro}$, $\mathrm{MaxPro}$,
  $\phi_p$, $\mathrm{CD}_2$, $\mathrm{SL}_2$, and the
  qualitative-quantitative pair $\mathrm{MaxPro}_{\mathrm{QQ}}$ and
  $\mathrm{QQD}$ for spaces with nominal or ordinal parameters.
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
- Sequential design tools (`mergen.sequential`): `extend`,
  `fill_around`, `subsample`, `run_order`, `k_fold_split`, and `nested`.
- Export formats: CSV, JSON, Markdown, LaTeX, HTML, Excel.
- Visualisation: pairplot, 1D distribution, 2D scatter, pairwise
  distance distribution, correlation heatmap, quality-metrics chart, and
  comparison charts including the criterion-by-algorithm percentile
  heat map.
- Convenience properties and helpers: `space.centroid`,
  `space.corners`, `space.bounds_as_dict`, `space.random_point()`.
- Documentation built with Sphinx: tutorials, task guides, explanation
  pages, an executed example gallery of fifteen studies, and the full
  API reference.
- Continuous integration via GitHub Actions (Python 3.9-3.12).
