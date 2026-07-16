# Compare criteria and algorithms

When you are unsure which criterion or optimiser suits your problem,
`compare()` settles it empirically: it builds a design for each
candidate combination on your actual space and ranks them on all
quality metrics at once.

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
(`n_repeats`, default 5) and ranked on the mean metric percentile, so
the winner reflects typical rather than lucky single-run performance.
The `priority` argument names the metrics that break ties, in order of
importance.

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

```{figure} ../_static/img/comparison_matrix.png
:width: 95%
:alt: Heat map of percentile ranks for criterion and algorithm combinations.

A small sweep (three criteria, two optimisers, two repeats each) on a
two-parameter space. Each cell is the mean percentile rank of that
combination on that metric; the starred row won under the requested
priority.
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

Comparisons repeat many optimisations, so they parallelise well. Pass
`n_jobs=-1` to use all cores, or a specific count such as `n_jobs=4`;
the result is identical to the single-core run, only faster. The
default stays on one core.
