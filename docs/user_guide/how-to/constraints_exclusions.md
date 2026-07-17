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
constrained space.

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
sampler.add_focus(point=[300, 2.0], spread=0.15, n_samples=4)
```

`spread` is the neighbourhood radius and `n_samples` the number of
focus runs. Use it when one region of the space warrants denser
sampling than the rest.

## Exclusion zones

An exclusion zone keeps the design away from an unsafe or
uninteresting region:

```python
sampler.add_exclusion(point=[500, 5.0], spread=0.20)
```

No design point will be placed within `spread` of the excluded point.

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
