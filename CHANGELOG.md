# Changelog

All notable changes to mergen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-06-29

### Added

- `ParameterSpace` — N-dimensional discrete parameter space with support for
  discrete, continuous (linear/log), and integer (linear/log) parameter types
- `GridSampler` — memory-efficient bijective grid representation using mixed-radix indexing
- `Sampler` — space-filling design sampler using the Stochastic Coordinate
  Exchange (SCE) algorithm with Iterated Local Search restarts
- Five optimisation criteria: uMaxPro, MaxPro, φ_p (p=15), CD2, Stratified L2
- Six quality metrics: min distance, minimax distance, max |correlation|,
  2D projection CD2, CV of distances, mean distance
- Runtime Monte Carlo quality assessment with percentile rank
- Prescribed points, focus regions, and exclusion zones
- Kennard-Stone validation set construction
- Sequential design tools: `augment`, `complement`, `from_dataframe`,
  `subsample`, `n_samples_recommendation`
- Export formats: CSV, JSON, Markdown, LaTeX, HTML, Excel
- Visualisation: pairplot, 1D distribution, 2D scatter, pairwise distances,
  correlation heatmap, quality metrics chart
- Convenience properties: `space.centroid`, `space.corners`,
  `space.bounds_as_dict`, `space.random_point()`
- CI via GitHub Actions (Python 3.9–3.12)
- 181-test suite with full coverage of all modules