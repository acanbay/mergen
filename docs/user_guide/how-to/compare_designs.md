# Compare criteria and algorithms

When you are unsure which criterion or optimiser suits your problem,
`compare()` settles it empirically: it builds a design for each
candidate combination on your actual space and selects one on the
raw values of the priority metrics.

## Comparing criteria

```python
from mergen import ParameterSpace, Sampler

space = ParameterSpace({
    'factor_a': ('continuous', 0.0, 1.0, {'resolution': 15, 'round': 3}),
    'factor_b': ('continuous', 0.0, 1.0, {'resolution': 15, 'round': 3}),
    'factor_c': ('continuous', 0.0, 1.0, {'resolution': 15, 'round': 3}),
})

sampler = Sampler(space)
sampler.set_design(n_samples=30)

comparison = sampler.compare(
    priority=('min_distance', 'max_abs_correlation'),
)
```

With `criteria` left unset, `compare()` evaluates every criterion
compatible with the space. Each combination is optimised several times
and is represented by the repeat that is best on the priority metrics.
By default (`n_repeats='auto'`) the repeats continue until the best
result has been found by five of them or a work limit is reached.
Except on small problems the limit is usually reached first; a warning
then lists those combinations with how far their other repeats' scores
lie from the delivered one, since another seed may give a different
design for them. An integer `n_repeats` runs
exactly that many.

The winner is chosen on the raw values of the `priority` metrics. By
default (`selection='pareto'`) they are competing objectives: designs
beaten on every metric by another are set aside, and among the rest
the one closest to the ideal (each metric at its best) wins; when two
designs share the shortest distance to the ideal, the priority order
breaks the tie, and the summary says so. With
`selection='lexicographic'` the metrics are applied in order of
importance: the first decides, and each later metric only decides
among designs tied on the earlier ones. Two values count as tied when
they differ by at most `tol`, which defaults to floating-point noise.
Designs still tied go to the one that is better on the other metrics
that may serve as a priority (compared exactly, in the order
`min_distance`, `minimax`, `max_abs_correlation`, `projection_cd2`,
`cd2`), then to the criterion and algorithm names, so the order of the
`criteria` and `algorithms` lists does not change the winner. The
metrics include out-of-budget prescribed and focus points, as in
`quality_report()`. Closely spaced out-of-budget points can therefore
set the same `min_distance` for every design, and the later priority
metrics decide. Metrics are computed on normalised coordinates, so
`tol={'min_distance': 0.01}` treats designs whose minimum distances
differ by less than 1 % of the parameter range as equally good. With
`selection='lexicographic'`, a priority such as `('minimax', 'cd2')`
asks for the best coverage first and the most uniform design second.
`comparison.reselect(...)` applies a different priority, rule or
tolerance without re-running the sweep.

## Reading and saving the result

```python
comparison.to_markdown('comparison_table.md')

best = comparison.best_result
best.quality_report()
best.plot('pairplot', save=True)
best.to_csv('winner_design.csv')
```

`comparison.best_result` is an ordinary result object, so the full
reporting and export interface is available on the winner.

The comparison itself has a plot: a heat map of the percentile-rank
table, with one row per criterion and algorithm combination, one
column per quality metric, and the winning row starred.

```python
comparison.plot(save=True)
```

```{figure} ../../_static/img/comparison_matrix.png
:width: 95%
:alt: Heat map of percentile ranks for criterion and algorithm combinations.

A small sweep (three criteria, two optimisers, two repeats each) on a
two-parameter space. Each cell is the percentile rank of that
combination's design on that metric against random designs of the same
size (random completions of any prescribed or focus points); the
starred row won under the requested priority. The
percentiles are descriptive: the selection uses the raw metric values,
available as `comparison.metrics`.
```

## Comparing algorithms

To hold the criterion fixed and compare optimisers instead, pass a
list of algorithms to `run`:

```python
result = sampler.run(criteria='phi_p', algorithm=['sa', 'sce', 'ese'])
```

The result carries the per-algorithm outcomes (`algorithm_results`)
and reports the best. This is the quickest way to trade optimisation
quality against runtime for your specific problem.

## Parallelism

Comparisons repeat many optimisations, so they parallelise well. By
default (`n_jobs=-1`) every core is used; pass a count such as
`n_jobs=4`, or `n_jobs=1` for a single core. The result is identical
for every setting, only the run time changes.
