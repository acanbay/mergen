# Objects and settings at a glance

Every object you can create and every setting you can change, on one
page. Each entry links to the page that explains it in depth; the
full signatures live in the {doc}`API reference <../api/index>`.

## Defining the space

{doc}`ParameterSpace <../api/space>` holds the parameters of your
study. Five parameter kinds are supported, freely mixed in one
space: an explicit list of values (discrete), a `('continuous', low,
high)` range sampled on a fine grid (add `'log'` for
logarithmically spaced values), an `('integer', low, high)` range,
and `('nominal', ...)` or `('ordinal', ...)` categories. The
{doc}`parameter types guide <how-to/parameter_types>` walks
through all of them.

## Shaping the design (before running)

All of these are methods of {doc}`Sampler <../api/sampler>`, called
before `run()`:

`set_design(n_samples, n_validation, extra_sets)`
: How many runs to place, how many validation points to reserve, and
  any additional named point sets. Guidance on choosing the size is
  in {doc}`How large should a design be? <explanation/sample_size>`.

`add_prescribed(points)`
: Runs that already exist and must be kept in the design, for
  example from an earlier campaign.

`load_design(points)`
: Resume from a complete previous design; `run()` then builds only
  what is missing around it.

`add_focus(point, spread, n_samples)`
: Extra points around a grid point that deserves them.

`add_exclusion(point)` and feasibility constraints
: A single grid point, or whole combinations, that must never be
  proposed. See
  {doc}`constraints and exclusions <how-to/constraints_exclusions>`.

`add_set(name, points)`
: Extra labelled point sets, such as a hand-picked test set.

`set_optimizer(name, **options)`
: Tune the optimisation budget (iterations, restarts) of the chosen
  algorithm.

`set_dimension_weights(weights)`
: Meant to make some parameters count more than others in the
  distance calculations. In this version the weights are stored but
  not used, and the call warns about it.

## Running

`Sampler.run(criteria, algorithm, seed, n_repeats, n_jobs)`
: Builds the design. Seven criteria and three algorithms are
  available; the {doc}`criterion guide
  <explanation/choosing_criterion>` and the {doc}`algorithm guide
  <explanation/choosing_algorithm>` explain when to pick which. By
  default the optimisation is repeated up to a work limit, stopping
  earlier only if its best result has been found five times, and the
  repeats run on every core. Except on small problems the limit is usually reached
  first; a warning then says how far the other repeats' scores lie from
  the delivered one, since another seed may give a different design.

`Sampler.compare(criteria, algorithms, n_repeats)`
: When you are unsure, run several combinations under a shared
  baseline and get the design that is best on the metrics you
  prioritise among those combinations, polished on those metrics to a
  local optimum (not a proven best); see
  {doc}`comparing designs <how-to/compare_designs>`.

## Working with the result

`run()` returns a result object with everything in one place:

`summary()` and `quality_report()`
: The design counts, and the statistical quality evidence against a
  Monte Carlo baseline. How to read the report is covered in
  {doc}`quality metrics <explanation/quality_metrics>`.

`plot(kind)`
: Seven plot kinds: `'pairplot'`, `'1d'`, `'2d'`, `'distances'`,
  `'correlation'`, `'quality'` and `'comparison'`, or `'all'` for
  every one. The criterion-by-algorithm heat map is
  `ComparisonResult.plot()`.

`to_csv()`, `to_excel()`, `to_json()`, `to_markdown()`,
`to_latex()`, `to_html()`
: Exports for running, sharing and publishing; see
  {doc}`exporting results <how-to/export_reports>`.

`compare()` returns a comparison object of its own, with
`summary()`, `plot()` (the percentile heat map), `to_markdown()` and
`best_result`.

## Growing a design over time

The {doc}`sequential module <../api/sequential>` extends finished
designs: `extend` adds new well-placed runs, `fill_around`
places new runs away from points that already have results (from
the literature or a pilot) and are not run again, `subsample` picks a
representative pilot subset, `run_order` orders runs so early stops
still leave a balanced design, `k_fold_split` builds
cross-validation folds, and `nested` creates designs within designs.
The workflow is shown in {doc}`sequential designs
<how-to/sequential_designs>`.
