# Changelog

All notable changes to mergen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Documentation URL and Zenodo software DOI in the package metadata,
  so `mergen.info()` prints both and `[project.urls]` carries a
  documentation link for PyPI.
- `mergen.selection`: one documented rule for choosing a design among
  candidates, shared by `Sampler.compare()` and by `Sampler.run()`
  when it picks a repeat or an algorithm on a `priority`.
- `selection`, `tol` and `rtol` arguments to `Sampler.compare()` and
  `Sampler.run()`. `selection='lexicographic'` (default) applies the
  `priority` metrics in order; `selection='pareto'` treats them as
  unordered, competing objectives. `tol` and `rtol` set when two metric
  values count as equal, for all metrics or per metric.
- `ComparisonResult.metrics` (raw metric values), `.ties`,
  `.decided_by`, `.front`, `.selection`, `.tol`, `.rtol`, and
  `ComparisonResult.reselect()` to choose again with another priority,
  rule or tolerance without re-running the sweep.

### Changed

- `ComparisonResult` takes the raw metric table as its second
  argument: `ComparisonResult(table, metrics, results, best, priority,
  ...)`. Code that builds one positionally must be updated.
- `compare()` treats criterion and algorithm names case-insensitively,
  runs a name listed twice, or an alias of a listed criterion such as
  `phip` for `phi_p`, only once (with a warning). It accepts a single
  criterion name as a string, rejects names that are not strings with a
  clear error, and requires `mc_samples` to be a non-negative integer;
  `mc_samples=0` skips the percentile baseline.
- `run()` validates `priority` whenever it is given: an empty
  `priority`, or an unknown or repeated metric name, is an error.
  0.1.0 ignored an empty `priority` and, with `n_repeats=1`, an unknown
  name.

### Fixed

- `Sampler.compare()` chose the winner on Monte Carlo percentile ranks,
  which saturate at 100 for well-optimised designs. Tied rows were then
  decided by the order of the `criteria` and `algorithms` lists, and a
  design worse on every priority metric could win. The winner is now
  chosen on the raw metric values; the percentile columns remain as a
  description.
- The order of `priority` had no effect, although the documentation and
  `summary()` presented it as an order of importance. It now does under
  the default lexicographic rule, and `summary()` states the rule used.
- `ComparisonResult.best` could be reassigned without updating the
  table, the plot or `best_result`; it is now read-only.
- `Sampler.run()` with several algorithms and a `priority` delivered
  the algorithm with the lowest criterion score and ignored the
  priority; the priority now decides between algorithms as well.
- `compare()` and `run()` accept a single metric name as `priority`;
  `run(priority='min_distance')` failed with "Unknown metric 'm'".
  `compare()` rejects an empty or repeated `priority` with a clear
  error. `compare(verbose=False)` no longer prints the winner's run
  summary; warnings are still shown, including a note when a tie was
  decided by the listing order and a warning when `best_result` no
  longer matches the compared design because the sampler was
  reconfigured.

### Results-affecting changes

- `Sampler.compare()` may return a different winner than 0.1.0 for the
  same seed, and `Sampler.run()` with a `priority` may return a
  different repeat or algorithm. Designs from `run()` without
  `priority` are unchanged.
- `compare()` now picks the representative repeat of every combination
  with the selection rule on raw values (0.1.0 used the Utopia distance
  on min-max scaled values), so every row of `comparison.table` and
  `comparison.results` may differ from 0.1.0, not only the winner.
- `selection='pareto'` does not reproduce the 0.1.0 winner: 0.1.0
  applied its Pareto rule to percentile ranks, the new rule works on
  raw values with tolerances.

## [0.1.0] - 2026-07-18

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

[unreleased]: https://github.com/acanbay/mergen/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/acanbay/mergen/releases/tag/v0.1.0
