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

### Polishing on the priority

Each criterion optimises its own score, not the priority metrics, so
the compared designs are usually some way from the best design for the
priority. With `polish=True` (the default) `compare()` therefore moves
the rows of every combination's design, one at a time, to the free
feasible grid node that improves the priority most, until no single
move helps. Prescribed and focus rows stay, exclusions stay free. A
polished design is never worse than its start (beyond `tol`) on the
first priority metric under `selection='lexicographic'`, and on every
priority metric under `selection='pareto'`. A polished design that
differs from its start joins the table as `'<algorithm>+polish'` and
competes under the same rule. Under `'lexicographic'` the winner is
therefore never worse on the priority than without polishing. Under
`'pareto'` it is never dominated by the winner without polishing, but
the added designs move the ideal point, so the choice can shift along
the front. `best_result.summary()` names the combination a polished
winner started from, and `run()` cannot reproduce a polished winner;
`polish=False` gives winners that it does. `reselect()` polishes again
for the new priority.

The result is a local optimum. In tests on five spaces (5 seeds, four
priority settings), the share of winners whose first priority metric
lay within 1 % of the best known value rose from 5 % to 39 %. With
`('minimax', 'cd2')` the median gap to the best known `minimax` value
(proven optimal on three of the spaces) fell from 18 % to 0 % (at most
10 %), and with `('cd2',)` from 5 % to under 1 %. `min_distance` is improved but not reliably to its best known
value: under the default priority the median gap fell from 27 % to
12 %, with up to 15 % left on continuous spaces and 41 % on a mixed
one. A target set before the tests, within 1 % in 90 % of all cases,
was not met by any setting. Polishing cost a median of under a second and at most 13 % of
the sweep time. The cost grows with the candidate set and runs on one
core: with three continuous factors at the default resolution of 100
(a million candidates) and 30 runs, one pass over the rows took about
30 s and over 1 GB of memory per combination, and a design may need
several passes.

`polish_kicks=k` adds k iterated-local-search kicks: three random rows
are moved and the exchange runs again, and the result is kept only if
it is better. Kicks follow the seed. With 10 kicks the cases within
1 % rose to 52 % and the best known `minimax` value was reached in
every case, but on a 3-factor space with 30 runs polishing took up to
1.1 times as long as the sweep. `polish=False` compares the optimiser designs only.

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
