# Constrain, exclude, prescribe, and focus

Four mechanisms let you shape a design beyond a plain box: feasibility
constraints, exclusion zones, prescribed points, and focus regions.
They compose freely.

## Feasibility constraints

A constraint removes infeasible candidates before optimisation. Pass a
predicate that receives a parameter dictionary and returns `True` for
allowed points:

```python
space.add_constraint(lambda p: p['flow_rate'] + p['n_stages'] <= 20)
```

Only candidates satisfying every constraint enter the design, and the
Monte Carlo baseline in the quality report is drawn from the same
feasible region, so quality percentiles stay honest inside a
constrained space. Prescribed and focus points are kept in every
baseline design, so the percentiles measure what the optimiser added
around them.

## Prescribed points

Prescribed points are runs that must appear in the design verbatim,
for example experiments you have already carried out:

```python
sampler.add_prescribed([[200, 1.0], [400, 3.0]])
```

By default these count toward the `n_samples` budget and are visible
to the criterion, so the optimised points are placed to complement
them. Both behaviours are configurable (`in_design`, `in_optim`).

## Focus regions

A focus region concentrates extra runs around a critical operating
point, on top of the space-filling design:

```python
sampler.add_focus(point=[300, 2.0], spread=1.0, n_samples=4)
```

`n_samples` is the number of focus runs. They are drawn from the grid
nodes around the point, with a Gaussian weight whose width `spread` is
counted in grid steps of each axis, so `spread=1.0` keeps them mostly
within one or two steps of the point whatever the units; a nominal
factor keeps the point's category. Use it when one region of the space warrants denser
sampling than the rest.

## Exclusion zones

An exclusion removes one unsafe or uninteresting point from the
candidates:

```python
sampler.add_exclusion(point=[500, 5.0])
```

No design, validation or extra-set point is placed at the excluded
point. Its neighbours stay available: in this version the `spread`
argument has no effect. To keep a whole region free, add a constraint
instead:

```python
space.add_constraint(
    lambda p: not (p['temperature'] > 450 and p['pressure'] > 4.0))
```

## Grid alignment

Prescribed, focus, and exclusion points are interpreted on the
candidate grid, so define parameters on an explicit grid when you rely
on these mechanisms, to be sure your reference points land on grid
nodes:

```python
space = ParameterSpace({
    'temperature': range(100, 501, 25),
    'pressure':    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
})
```
