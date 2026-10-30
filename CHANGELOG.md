# Changelog

All notable changes to mergen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.1] - 2026-10-30

### Added

- Documentation URL and Zenodo software DOI in the package metadata,
  so `mergen.info()` prints both and `[project.urls]` carries a
  documentation link for PyPI.
- `mergen.selection`: one documented rule for choosing a design among
  candidates, shared by `Sampler.compare()` and by `Sampler.run()`
  when it picks a repeat or an algorithm on a `priority`.
- `selection`, `tol` and `rtol` arguments to `Sampler.compare()` and
  `Sampler.run()`. `selection='pareto'` (default) treats the `priority`
  metrics as competing objectives and takes the non-dominated design
  closest to the Utopia point; `selection='lexicographic'` applies them
  in order. `tol` and `rtol` set when two metric values count as equal,
  for all metrics or per metric.
- `n_repeats='auto'`, the new default of `Sampler.run()`,
  `Sampler.compare()` and the `sequential` functions. Repeats run in
  rounds of 5 up to a work limit and stop earlier only when the
  selected result has been found by 5 of them.
  While no other repeat has reproduced
  the best result, each round doubles the optimiser's search length,
  unless it was set with `set_optimizer()`; a doubled round that
  would pass the work limit runs at the current length. The work stops at 64
  repeats, or earlier at the work of 32 repeats at the default search
  length, with a warning that the result may still depend on the seed
  and how far the other repeats' criterion scores lie from the
  delivered one. Except on small problems the limit is usually reached:
  on benchmark spaces of 15 to 30 runs, 7 % of runs stopped early, and the
  best of 20 repeats varied across seeds by a median 2.1 % (up to
  11 %). Counting repeats as equal within a tolerance was tested and
  rejected: at 0.1 to 5 %, 11 to 36 % of the runs it declared settled
  were further than the tolerance from the best known score.
  An integer `n_repeats` runs exactly that many repeats.
  `SamplingResult.repeat_info` and `ComparisonResult.repeat_info`
  report whether the repeats stopped early and how far the other
  repeats' scores lie from the delivered one.
- A `cd2` quality metric: the centred L2 discrepancy of the whole
  design, so a `priority` such as `('minimax', 'cd2')` can name it.
- `mergen.distances.maximin_subset()`: the set of candidate points
  that, as a whole, lies farthest from a reference set and from itself.
- `ComparisonResult.metrics` (raw metric values), `.ties`,
  `.decided_by`, `.front`, `.selection`, `.tol`, `.rtol`, and
  `ComparisonResult.reselect()` to choose again with another priority,
  rule or tolerance without re-running the sweep.
- Polishing in `Sampler.compare()` (`polish=True` by default,
  `mergen.polish`): every combination's design is improved on the
  priority metrics by point exchange on the feasible grid, keeping
  prescribed and focus rows; improved designs join the table as
  `'<algorithm>+polish'`, so the winner is never worse on the priority
  under `'lexicographic'` and never dominated by the unpolished winner
  under `'pareto'`. With `min_distance` in the priority, the exchange
  alternates with moves that lower the Morris-Mitchell phi_p. On five
  test spaces the winners within 1 % of the best known first priority
  metric rose from 5 % to 41 %; the best known `minimax` value was
  reached in 21 of 25 `('minimax', 'cd2')` cases, and `cd2` came within
  a median 1 % (at most 5 %). `polish_kicks` adds seeded
  iterated-local-search kicks (default 0).
- Known limit: `compare()` does not reliably reach the best
  `min_distance`. With polishing, the winner lay a median 3 % (up to
  12 %) below the best known value under `('min_distance',)` and 10 %
  (up to 38 %) under the default Pareto priority, where the phi_p moves
  changed no winner in the tests. Targets set before
  the tests (90 % of all cases within 1 %; for `min_distance` a median
  gap of at most 2 % and none above 10 %) were not met, so `compare()`
  warns whenever `min_distance` is a priority metric. This is carried
  to 0.2.0 as an open defect.

### Changed

- `Sampler.run()` and `Sampler.compare()` use every CPU by default
  (`n_jobs=-1`). Results are the same for any `n_jobs`; `n_jobs=1`
  runs on one core. Parallel work uses joblib's loky backend on every
  platform.
- Under `selection='pareto'` each metric is scaled over the Pareto
  front only, so a dominated design can no longer change the winner.
- Repeats with equal scores are now ordered by their designs, not by
  the order in which they were found, and every delivered design is
  given in one canonical form: its optimised rows sorted, and, along an
  axis where the grid, the constraints and the fixed points are
  symmetric, the mirror image that sorts first. Equally good designs
  found from different seeds are then delivered identically.
- Validation and extra sets are the maximin sets of the feasible pool
  (`maximin_subset()`): the points are chosen as a whole, so the
  smallest distance to the design and to each other is as large as
  possible. Kennard-Stone, which adds one point at a time, gives the
  starting bound. The exact search is limited by a count of candidate
  checks, about two seconds on a current desktop; past it, the best set found is used, which is
  never worse than the Kennard-Stone set, and a warning says the set is
  not proven optimal. A holdout set larger than the free feasible points
  is cut to them with a warning.
- The CD2 criterion updates its score in O(n·d) when one point moves
  instead of recomputing it, and grid look-ups are cached; SA runs
  about 1.7 times faster and CD2 on designs of 100 points or more up to
  200 times faster.
- `ComparisonResult` takes the raw metric table as its second
  argument: `ComparisonResult(table, metrics, results, best, priority,
  ...)`. Code that builds one positionally must be updated.
- Continuous integration also tests Python 3.13 and 3.14, and the
  package metadata lists both.
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
- `mean_distance` and `cv_distances` can no longer be `priority`
  metrics: a design can raise its mean distance by crowding points into
  the corners, and a regular lattice has a low CV whatever its spacing.
  Both are still reported.
- Percentile ranks count the design among the baseline designs (B + 1
  values), so they lie strictly between 0 and 100; a NaN metric value
  now gives a NaN rank instead of the 0th percentile.
- `quality_report()` scores the run's criterion with the instance that
  was optimised, so parameters such as `PhiP(p=5)` are respected (0.1.0
  looked the criterion up by name and gave NaN for an instance). Its
  Monte Carlo baseline uses the run's seed, as `compare()` does, instead
  of a fixed 44.
- The Markdown, LaTeX and HTML reports use the same 300 baseline
  designs as `quality_report()` instead of 200. CSV, JSON and Excel
  files keep every digit of the values.
- `set_dimension_weights()` warns that the weights have no effect: no
  criterion or metric reads them in this version.
- `joblib>=1.3` is required (`return_as='generator'` needs it).
- `MERGEN_SILENT=0`, `false`, `no` and `off` no longer silence the import
  banner.
- `compare()` with `n_repeats='auto'` lists every combination whose best
  result was not found five times within the work limit.
- `compare()` gives the cores that the combinations leave idle to their
  repeats, chooses the holdout sets once (for `best_result`) instead of
  once per combination, and `best_result` reuses the winner's repeats
  from the sweep instead of optimising them again on one core. The
  default sweep on a 9 x 16 grid with 20 points takes about 35 s
  instead of 79 s on 16 cores (38 s instead of 89 s with 20 validation
  points); the result is the same.
- `run()` with `verbose=False` no longer prints its configuration and
  final-design banners, also for a loaded design; warnings are still
  printed.
- `run()` no longer reseeds the global `random` and `numpy.random`
  generators; focus neighbours are drawn from a generator seeded by the
  run's seed.
- `run()` refuses a list of several criteria; 0.1.0 used the first and
  ignored the rest without a word. `compare()` tries several.
- `set_design()` refuses a negative or non-integer `n_samples`,
  `n_validation` or extra-set size, and an extra set named like a
  built-in or an `add_set()` set (0.1.0 replaced that set).
- Percentile ranks are printed between the 1st and the 99th, as the
  estimator never reaches 0 or 100; 0.1.0 rounded 99.8 to "100th".
- `plot('quality')` uses the same 300 baseline designs as
  `quality_report()` instead of 200.

### Fixed

- Criteria that departed from their published definitions:
  - uMaxPro normalised each axis so that its lowest and highest level
    were the same point on the periodic scale, so every design had a
    clamped tie on each axis and the score was not invariant under
    cyclic shifts. The levels now sit at their stratum midpoints,
    level k of m at (k + 1/2)/m (Vorechovsky & Masek).
  - MaxPro and uMaxPro clamped a coordinate tie at 1e-10; on an axis
    with fewer levels than runs, where ties are forced, these ties made
    up more than 99.99 % of the score. Such an axis now contributes
    (|Δ| + 1/m)², the form of Joseph, Gul & Ba (2020, Eq. 9); axes with
    at least n levels keep the published term. MaxProQQ applies the
    same rule to a continuous column whose grid has fewer than n
    levels. MaxProQQ and QQD also count a factor's levels in the
    feasible pool, as MaxPro and uMaxPro do, so a constraint that
    removes a level changes the 1/m weight and QQD's constant.
  - QQD placed quantitative levels at the ends of [0, 1], so the first
    and last level coincided in its wrap-around kernel. They now sit
    at (2x + 1)/(2s), as in Zhang, Yang & Zhou (2021); the paper's
    example value QQD² = 0.0213 is reproduced.
  - φ_p's fast update lost most of its digits when one pair held most
    of the score (94 % error in one case), and with a large p a
    duplicate point overflowed to infinity. A dominant pair now
    triggers a full recomputation, also when the optimiser rejected an
    earlier recomputed move and continued from an older design. `PhiP` accepts only 0 < p <= 30: there the
    sum stays finite with a floor that only exact duplicates reach,
    while a larger p would need a floor at which close pairs score like
    duplicates. MaxPro, uMaxPro and MaxProQQ use the same safeguard
    when a near-duplicate pair is moved apart.
  - A parameter with a single feasible value changed the order of
    designs under CD2, stratified L2 and QQD (11 % of pairs in one
    test), although Mergen warns that it does not contribute. It is
    now left out of every criterion and quality metric.
- The `projection_cd2` metric averaged CD2 over the 2D projections;
  Sun, Wang & Xu (2019) define the uniform projection criterion as the
  mean of CD2². It now follows the paper.
- Nominal and ordinal levels came out of Mergen as level indices
  (0, 1, 2) in `result.samples`, every export, the report tables and
  the plot axes, although the labels are kept for round-tripping. They
  now come out as the labels, kept as given: a label 32 stays the
  number 32 instead of becoming the text '32'. Every function that
  takes points (`add_prescribed`, `add_focus`, `add_exclusion`,
  `add_set`, `load_design` and the `sequential` functions) accepts the
  labels, also as text read back from a CSV file, and a DataFrame by
  column name; `run_order` and `subsample` return the labels too; `add_prescribed`, `add_focus` and `add_exclusion`
  rejected labels, and ordinal labels were rejected everywhere. A
  level index is still accepted unless a label of that parameter is
  itself a number. Two labels naming the same number (`1` and `1.0`),
  and NaN or infinite labels, are refused; a number label is found
  however it is written (`'1.0'` for 1). Labels, parameter names, set
  names and the label of a loaded design are escaped in the LaTeX,
  Markdown and HTML reports, in their tables as well. A `$...$` span
  that matplotlib reads as math, such as `$\tan\beta$`, stays math in
  the reports as in the plots; any other `$` is shown literally, and in
  plots no longer makes them fail.
- A constraint received the level index of a nominal or ordinal
  parameter, so `lambda p: p['solver'] != 'adam'` silently removed
  nothing. Constraints now receive the label.
- `add_prescribed()` hid the prescribed points from the criterion by
  default (`in_optim=False`), although the documentation says they are
  visible, so optimised runs could be placed one grid step from an
  existing run. The default is now `in_optim=True`.
- A point excluded with `add_exclusion()` could stay in a set added
  with `add_set()`. A point that appears twice among the `add_set()`
  sets or in a loaded design, or in both a loaded design and a set,
  was accepted; `load_design(name='Validation')` was too. All are now
  refused with a clear error.
- Two overlapping focus regions could deliver fewer points than
  requested, without a warning, when a centre with
  `include_center=True` had already been drawn by the other region.
- `set_optimizer()` accepted out-of-range values such as
  `max_iter=-5` or `n_restarts=2.5`; some failed only when `run()`
  started, others not at all. They are now refused at once.
  `k_fold_split` and `subsample` cut a count such as 2.7; they now
  refuse it. `set_dimension_weights()` accepted negative,
  non-finite or all-zero weights and unknown names.
- A discrete parameter accepted `nan` values, and a range `nan`
  bounds. A generator used as a
  parameter's values broke `set_resolution()`. A resolution below 2
  was accepted by the constructor and refused by `set_resolution()`.
  An ordinal or nominal spec with an extra item was reported as an
  unknown type, and a grid point that violates a constraint as "not
  on the parameter grid". All now give the right check and message.
- `run(seed=None)` and `compare(seed=None)` use seed 44, as before;
  the docstrings now say so.
- Sets added with `add_set()` without a colour, and every set from
  `set_design(extra_sets=...)`, were left out of the plots and their
  legends. Every set is now drawn, with a fallback colour when it has
  none.
- The Markdown, LaTeX and HTML reports of a loaded design had no table
  of its points; they now have one under the design's label. Every
  report printed "Restarts: ?"; it now gives the number of repeats.
  The LaTeX report left `_` and other special characters in parameter
  names unescaped, and its summary left out the extra sets. CSV and
  Excel exports had an unnamed index column; it is now `id`.
- Mergen's messages, warnings and errors, and `quality_report()`,
  wrote terminal colour codes into redirected output and logs; colours
  are now used only on a terminal and in Jupyter. `quality_report()`
  also shifted its Rank column by the width of the colour codes,
  marked both the criterion's own metrics and the criteria
  evaluated after the run with `*` (the latter now `+`), and treated
  `mc_samples=-1` as 0; it now refuses a negative count.
- `plot('2d')` on a one-parameter space failed with a
  `ZeroDivisionError`, and an unknown parameter name with a
  `KeyError`; both now raise a clear `ValueError`. The
  `SamplingResult.plot` docstring now lists every plot kind.
- `SamplingResult.comparison()` and `designs` were described as
  covering several criteria, although `run()` optimises one.
- `Sampler.run()` failed with its default criterion on a space with a
  nominal parameter, since uMaxPro does not support one. The default
  is now `criteria=None`: MaxProQQ when the space has a nominal or
  ordinal parameter, uMaxPro otherwise (`mergen.sampler.default_criterion`).
- `sequential.extend`, `fill_around` and `nested` defaulted to `cd2`,
  although their docstrings said the defaults match `run()`. They now
  use the same default criterion as `run()`; `extend` and
  `fill_around` still add no validation points, and say so. A count
  such as `n_new=2.7` is refused instead of being cut to 2.
- `ComparisonResult.reselect()` with `n_repeats='auto'` built
  `best_result` from fewer repeats than the table under the new
  priority, so the two could hold different designs, and it warned
  that the sampler had been reconfigured. `best_result` now uses all
  the stored repeats, and holds the design the table describes.
- `Sampler.compare()` gave a tie on every priority metric to the
  combination listed first, so reordering `criteria` or `algorithms`
  could change the winner. Ties now go to the design that is better on
  the other metrics that may serve as a priority, then to the
  criterion and algorithm names. `run()` with several algorithms
  breaks equal scores on the algorithm name in the same way.
- The Monte Carlo baseline of `quality_report()` and `compare()` drew
  every point of its random designs at random, also for designs with
  prescribed or focus points. A dense focus region or clustered
  prescribed points then pulled the rank down whatever the optimiser
  did. Every baseline design now keeps the points the optimiser did
  not choose and draws only the rest, from the candidates the
  optimiser could use (without `add_set()` points and focus centres
  with `include_center=False`, and without the reference points of
  `sequential.fill_around`), and the report footer says so. Designs
  without any of these points are not affected by this change.
- `compare(seed=None)` drew its baseline from fresh entropy, so its
  percentiles differed from `best_result.quality_report()`, which uses
  seed 44. Both now use 44.
- `quality_report()` on a `load_design()` result printed a criterion
  row named "LOADED" with no values; it now has no criterion row.
- `compare()` after `load_design()` failed with an `AttributeError`;
  it now stops with a clear error.
- With out-of-budget prescribed or focus points, the `compare()` table
  left those points out while `quality_report()` counted them, so the
  two showed different values for the same design. The table, the
  choice of the winner and the choice among repeats now include them.
- `Sampler.compare()` chose the winner on Monte Carlo percentile ranks,
  which saturate at 100 for well-optimised designs. Tied rows were then
  decided by the order of the `criteria` and `algorithms` lists, and a
  design worse on every priority metric could win. The winner is now
  chosen on the raw metric values; the percentile columns remain as a
  description.
- The order of `priority` had no effect, although the documentation and
  `summary()` presented it as an order of importance. It now does under
  the lexicographic rule, and `summary()` states the rule used.
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
  broken and a warning when `best_result` no
  longer matches the compared design because the sampler was
  reconfigured.
- The Monte Carlo baseline of `quality_report()` and `compare()` drew
  random designs from the full grid, ignoring constraints, and could
  include exclusion centres. It now draws from the feasible candidate
  pool without exclusion centres, as the documentation stated.
- `minimax` was estimated on a regular grid over the unit cube. That
  grid did not match the candidate points and ignored constraints and
  nominal factors: on a triangular constrained space, a design holding
  every feasible point scored 0.71 instead of 0. It is now the exact
  maximum over the feasible candidate pool (a fixed subset of 50,000
  points for larger pools). `comparison()` now passes the space to
  every metric, so nominal factors are handled there as in
  `quality_report()`.
- The validation set and the generated extra sets ignored constraints,
  so they could contain infeasible points, and on grids above 500,000
  points they were drawn with Python's global random generator. They
  are now chosen deterministically over the feasible candidate pool
  (see Changed), with a Kennard-Stone routine shared with
  `sequential.subsample`, `run_order`, `k_fold_split` and `nested`. It
  measures HEOM distance, so nominal factors count as unordered, and
  gives a tie (distances equal to a relative 1e-9) to the lowest
  candidate index. The reference set also includes the out-of-budget
  prescribed and focus points and the sets added with `add_set()`.
- `Sampler.run()` could deliver a point that violates the constraints:
  without prescribed or focus points the optimisation starts from a
  corner of the grid, which was used even when infeasible. It now
  starts from the nearest free feasible candidate in that case.
- SA and ESE column swaps could move a design point onto an exclusion
  centre, a point of an `add_set()` set or an out-of-budget prescribed
  point. Such swaps are now rejected.
- On a small feasible pool that the design nearly fills, `run()` and
  `compare()` could take minutes: every random replacement tried up to
  100,000 grid draws. After 1,000 failed draws the free feasible
  points are now listed and one is drawn from them.
- Focus neighbours were drawn from the whole grid, so they could be
  infeasible or an exclusion centre. They are now drawn from feasible
  points that are not excluded.
- With `in_optim` mixed between prescribed and focus points, the
  `point_type` labels of those rows were swapped, and a focus region
  that yielded fewer than `n_samples` points shifted the `in_optim`
  flags onto the wrong rows.
- The seed corner stayed reserved after the optimiser moved it, so the
  validation and extra sets could never use that grid node.
- SA never cooled. Its starting temperature was calibrated on raw
  score changes while the acceptance test uses changes of log(score),
  so the temperature came out near 1e20 and SA accepted almost every
  move to the end. The temperature is now calibrated on log-score
  changes (Ben-Ameur, 2004), and on the moves the chain makes from its
  actual start: the restart's start design with its frozen rows, the
  same mix of swaps and replacements, and only free, feasible nodes.
  0.1.0 probed a separate random design and also moves to reserved or
  banned nodes, so the early acceptance missed the 0.80 target (0.25
  to 0.49 in one test). On 37 benchmark combinations of space and
  criterion with 10 seeds each, the final score is within the
  seed-to-seed spread of the old calibration in 31, better in 2 and
  worse in 4 (by 1.3 to 9.2 %). With a short user-set `max_iter` the
  hotter start can cost more: four single-seed regression designs with
  a few hundred iterations scored 1 to 22 % worse. A given `T_start`
  now skips the probe. SA results carry `early_acceptance` in their
  metadata: the share of accepted moves in the first 5 % of each
  restart.
- ESE did not follow Jin, Chen & Sudjianto (2005). It used a
  Metropolis test on log(score) instead of the paper's threshold rule,
  its starting threshold had the scale error described for SA, its
  acceptance and improvement counts accumulated over the whole run
  instead of the last outer iteration, its exploration phase had no
  memory of whether it was raising or lowering the threshold, it drew
  each candidate exchange from a random column, and with the default
  `Q=1` the threshold never adapted. ESE now accepts a candidate when
  its score change is at most `Th × U(0, 1)` with `Th0 = 0.005 ×` the
  starting score, tries `J` distinct exchanges that change the design
  in column `i mod d` at inner iteration `i`, raises the threshold in
  exploration until more than 80 % of the moves are accepted and then
  lowers it until fewer than 10 % are, and sizes `M` and `J` from the number of possible
  exchanges as the paper does. `Q` defaults to enough outer iterations
  for about 10,000 candidate exchanges. The paper itself could not be
  consulted; the rules follow two sources that restate it.
- SCE stopped its iterated local search at the first kick that did not
  improve the design and kicked from the current design. It now stops
  after `max_ils_kicks` failed kicks in a row and kicks from the best
  design (Lourenço, Martin & Stützle, 2003).
- SA and ESE scored a swap between two rows that differ only on the
  swapped axis, which leaves the design unchanged, as a large
  improvement: the two-step score update passed through a duplicate
  point and the cancellation that followed gave a false value. ESE then
  kept choosing such swaps and could stop improving. They are now
  skipped.
- SA passed the next restart the occupied grid nodes of its last
  design instead of its best design, so some free nodes could not be
  drawn as replacements.
- A constraint written as a lambda failed with "Can't pickle local
  object" when `run()` or `compare()` ran on several cores.
- `add_focus()` documented `spread` in grid steps, the how-to guide in
  normalised coordinates, and the code divided it by the mean of the
  raw parameter ranges, so the neighbourhood depended on the units. It
  also drew candidates from the whole grid, so on a fine grid the
  "focus" points could land anywhere. `spread` is now the width of the
  Gaussian kernel in grid steps of each axis, the neighbours come from
  the grid nodes around the centre, a nominal factor keeps the centre's
  category, and `spread` must be positive.
- A prescribed point or a focus centre that is also a point of a set
  added with `add_set()` is now an error; it was accepted although a
  set point can never be in the design.
- Percentile ranks were printed as "72th", "1th" and "22th"; they now
  read 72nd, 1st and 22nd.
- SCE drew its restart kicks from Python's global random generator, so
  `run(n_jobs=...)` and `compare(n_jobs=...)` could give different SCE
  designs for the same seed. It now uses the run's seeded generator.
- Factors measured in very small units did not work. Grid values were
  matched with an absolute tolerance of 1e-9, so for levels around
  1e-12 every level matched the first one and duplicate and reservation
  checks failed; the quality metrics treated a range below 1e-10 as
  zero; SCE skipped levels closer than 1e-12. Matching now uses a
  tolerance scaled to the gap between neighbouring levels (never below
  a few units in the last place), and the design, score and metrics no
  longer depend on the unit.
- `sequential.extend()` failed with "appears in add_prescribed() more
  than once" when the sampler had prescribed or focus points, and on a
  sampler with a loaded design. It now adds the existing rows as the
  only fixed points, keeps their `point_type` labels, and restores the
  sampler afterwards, including its `extra_sets` (0.1.0 lost them after
  `extend`, `fill_around` and `nested`). `sequential.fill_around()`
  failed in the same cases; it now treats the sampler's prescribed
  points as further reference points and returns exactly `n_new`
  points.
- With every feasible point taken, `run()` now raises a clear error
  instead of starting from a point of a user-supplied set.
- When a quality report failed during an export, `sys.stdout` stayed
  redirected and all later output was lost; the error was hidden. The
  export now continues without the table and warns.
- JSON export wrote values with 10 decimal places, so 1e-12 became 0.0;
  CSV and Excel kept 10 significant digits.
- The documentation said an exclusion keeps the design `spread` away
  from the excluded point. Only the point itself is removed, and
  `spread` has no effect; the documentation and example 03 now say so
  and show a constraint for excluding a region.
- The Monte Carlo baseline warns when fewer feasible points are left
  than the design has, and returns NaN when none are left.
- The balanced Latin hypercube that starts every optimiser chose the
  levels used once more at random. With fewer points than levels (for
  example 20 points on a 100-level axis) the design could miss whole
  parts of the range, and ESE, which only swaps levels, could never
  reach them. The levels are now drawn one from each of equal strata of
  the axis, as in a Latin hypercube (McKay, Beckman & Conover, 1979),
  and where fixed points already use a level, the level given up is
  the one nearest them, so no stratum is left empty.
- ESE counted draws that could not change the design (the same row,
  the same level, an infeasible or duplicate result) towards its `J`
  candidate exchanges; it now evaluates `J` exchanges that change the
  design, as the method describes.
- SA with every row fixed returned an infinite score; it returns the
  design's score. SA and SCE refuse `max_iter` below 1, and SA refuses
  `T_start` ≤ 0 and `cooling`, `T_end_ratio` or `target_accept` outside
  (0, 1); 0.1.0 accepted them or failed with a division by zero.
- `('integer', 1.5, 3.7)` gave the levels 2, 3 and 4; an integer
  parameter now holds exactly the integers between its bounds.
- `ParameterSpace.set_resolution()` did not rebuild the grids it
  promised to; parameters without their own `resolution` now get the
  new one.
- Prescribed, focus, exclusion, set and loaded points are stored as the
  grid node they match, so `0.3` on a grid whose node is
  `0.30000000000000004` no longer misses exact look-ups.
- A focus region with fewer free neighbours than requested left the
  design short (for example 17 of 20 rows); the missing slots now go to
  the optimiser.
- `sequential.extend()` generates the sampler's extra sets around the
  extended design (0.1.0 returned none) and warns when a prescribed
  point of the sampler is missing from `existing_design`. A row that is
  an out-of-budget prescribed point of the sampler stays out of the
  budget instead of taking a design slot, and the existing rows come
  first in their input order.
  `fill_around()` reports its true design size in the result metadata.
- The `projection_cd2` metric cited Liu & Liu (2023); its definition is
  that of Sun, Wang & Xu (2019), which average the squared discrepancy
  where Mergen averages the discrepancy. The `cv_distances` reference
  named the wrong journal.
- The quality report marked with `*` metrics that a criterion was said
  to optimise (for example minimax and maximum correlation for
  `umaxpro`), with no source; a default `umaxpro` design is no better
  than a random one on correlation. A metric is now marked only when it
  follows from the criterion's definition, which leaves the minimum
  distance for `phi_p` (maximin as p grows; Morris & Mitchell 1995).
- A prescribed or focus point outside the budget (`in_design=False`)
  was never seen by the criterion, even with `in_optim=True`; only its
  grid node was reserved, so new runs could sit right next to it. Such
  points are now frozen rows the optimiser sees, and they are still
  delivered after the budget rows. This also makes
  `sequential.fill_around()` work as documented: its references were
  ignored by the criterion, so moving a reference left the design
  unchanged.
- `sequential.fill_around()` refused a reference point that is not on
  the parameter grid, although measured literature or pilot points are
  its use case. A reference point now has to lie within the range of a
  continuous parameter and on a level of a discrete or categorical one.
- `sequential.k_fold_split()` said that each fold covers the space, but
  its Kennard-Stone round-robin folds beat only 3 to 85 % of random
  splits in coverage. The folds now minimise the largest fold coverage
  radius (the largest distance from a design point to the nearest point
  of the fold), then the sum of the radii, by pairwise swaps between
  folds from several Kennard-Stone starts. On 240 test designs (2 to 5
  factors, 20 to 300 rows, 2 to 10 folds), at most 11 of 200 random
  splits covered better, and at most 3 on designs without repeated
  rows. A design of 1,000 rows takes a few seconds, up to about 15 s
  with 10 folds.
- The algorithm guide gives the measured cost of ESE keeping the
  levels of its starting design: a median 4 to 11 % above the best
  design found on benchmark spaces (1 to 5 % for SA), and 47 % on an
  axis with many more levels than runs.
- Documentation that contradicted the code: the user guide said
  `fill_around` places new runs near the reference points (it places
  them away from them); the `nested` docstring gave wrong shapes and
  row order (and `nested` could pick an out-of-budget prescribed point,
  which is never run, for the inner design; it now picks from the
  budget rows only); the `sequential` module listed four of its six functions;
  `get_criterion()` listed five of the seven criteria; the SA docstring
  called random replacement the Morris-Mitchell move (their move swaps
  two values within a column, as Joseph, Gul & Ba 2020 describe it); `list_optimizers()` was documented as
  returning an empty list; the `Sampler` docstring named SCE as its
  engine although SA is the default; `GridSampler` claimed 48 bytes for
  a 10^10-point grid while a `Sampler` holds over 100 bytes per
  feasible candidate; `maximin_subset()` returns sorted indices, not
  the selection order of `kennard_stone()`. Tian & Xu are now cited as
  2026 and Joseph, Gul & Ba (MaxProQQ) as 2020 throughout.
- Documentation builds: Read the Docs used whatever gallery the
  docs-gallery branch held, so a build that started before the gallery
  of new code was ready executed the changed examples and ran into the
  time limit, and rebuilding an older version used the gallery of newer
  code. The gallery now records the code it was built from; Read the
  Docs builds only when it matches, cancels the build otherwise (the
  pages online stay), and never executes an example. The docs-gallery
  workflow starts the waiting builds once the gallery is ready. Each
  release keeps its own documentation version, also downloadable as
  zipped HTML.
- The estimated runtimes of six examples were wrong: those of 04, 12
  and 14 were too short for the default repeats of 0.1.1, those of 05,
  06 and 09 too long. They now give times measured on four cores.

### Results-affecting changes

- uMaxPro and QQD designs change on every space; MaxPro designs change
  on spaces with an axis of fewer levels than runs, MaxProQQ designs on
  spaces with a continuous grid coarser than the design. CD2,
  stratified L2 and QQD results change on spaces with a single-valued
  parameter. `projection_cd2` values are now squared discrepancies, so
  their scale changes, and its percentile ranks can change.
- Prescribed points are now seen by the criterion by default: every
  run that uses `add_prescribed()` with default arguments gives a
  different design. Pass `in_optim=False` for the previous behaviour.
- Constraints on nominal or ordinal parameters now receive labels. A
  constraint written against level indices (`p['solver'] != 0`)
  changes meaning; rewrite it with the label.
- Nominal and ordinal columns of `result.samples`, `validation`, `sets`
  and every export hold labels instead of numbers, and
  `ParameterSpace.category_labels()` returns the labels as given
  instead of as text.
- The default criterion of `run()` on a space with an ordinal
  parameter is now MaxProQQ instead of uMaxPro. Default calls of
  `sequential.extend`, `fill_around` and `nested` use uMaxPro (or
  MaxProQQ) instead of CD2.
- Every design changes. The SA, ESE and SCE fixes change the search,
  `n_repeats='auto'` runs more repeats and may lengthen the search,
  ties between repeats are broken by the design, and the delivered rows
  are sorted and, where the problem is symmetric, mirrored. To repeat a
  0.1.0 result, use Mergen 0.1.0.
- `compare()` polishes by default, so its winner usually changes and
  its table gains `'+polish'` rows. Pass `polish=False` to compare the
  optimiser designs only.
- `Sampler.compare()` may return a different winner than 0.1.0 for the
  same seed, and every row of `comparison.table` and
  `comparison.results` may differ: the representative repeat of each
  combination is chosen on raw metric values, and the default rule is
  now `'pareto'`. `selection='pareto'` does not reproduce the 0.1.0
  winner either: 0.1.0 applied its Pareto rule to percentile ranks.
- Percentile ranks change for designs with prescribed or focus points,
  `add_set()` sets or focus centres with `include_center=False`, and
  for `sequential.extend` and `fill_around` results. `compare(seed=None)`
  percentiles no longer change from call to call. The
  `compare()` winner does not change, as it is chosen on raw values.
- With out-of-budget prescribed or focus points, `run()` with a
  `priority` and `compare()` may select another repeat or winner, as
  the priority metrics now include those points. Exact ties in
  `compare()` may go to another combination.
- `minimax` values can change for any design, and the baseline medians
  and percentiles change for constrained spaces and spaces with
  exclusions. Because `compare()` selects on these metrics, a
  `priority` that contains `minimax` may pick a different winner.
  `comparison()` values change for spaces with nominal factors.
- Validation and extra sets change wherever the maximin set differs
  from the Kennard-Stone points, and for constrained spaces, spaces
  with nominal factors, grids above 500,000 points, designs with
  out-of-budget points or `add_set()` sets. `sequential` selections can
  change where candidates tie.
- Percentile ranks shift slightly (B + 1 estimator), and the baseline
  of `quality_report()` changes for every run whose seed is not 44.
- `sequential.k_fold_split()` assigns different rows to the folds.
- Designs with prescribed or focus points outside the budget and
  `in_optim=True` (the default) change, and so does every
  `sequential.fill_around()` result. `PhiP` refuses a p that is not a
  number in (0, 30].
- `run()`, `compare()` and the `sequential` functions take longer by
  default, as they repeat up to a work limit (stopping earlier only if
  the best result is found five times); pass an integer `n_repeats`
  for a fixed number of repeats.

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

[unreleased]: https://github.com/acanbay/mergen/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/acanbay/mergen/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/acanbay/mergen/releases/tag/v0.1.0
