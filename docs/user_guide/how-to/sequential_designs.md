# Extend and reuse designs

The `mergen.sequential` module works on existing designs: it adds
points, reuses previous campaigns, orders runs, and partitions designs
for cross-validation. Every function looks only at parameter-space
geometry, so nothing here depends on your measured responses.

## Extend an existing design

Add space-filling points to a design while keeping every original run
unchanged:

```python
from mergen import sequential

extended = sequential.extend(sampler, base.best_design, n_new=10)
```

The first rows of `extended` are exactly your original points; the
`n_new` additions are optimised to fill the space around them. This is
the right tool when a pilot design is to be grown into a larger one
without wasting completed runs.

## Reuse a previous campaign

To rebuild a result object around a design you already have, for
example to re-report or extend a design loaded from a file, use
`load_design`:

```python
sampler.load_design(previous_campaign)
```

`load_design` deliberately refuses to combine with `n_samples` or
`add_prescribed`: the design is taken as given rather than
re-optimised, and mixing the two intents silently would be a trap. A
labelled variant records a name and colour for reporting:

```python
sampler.load_design(prev_df, name='campaign_2024', color='#ff8800')
```

## Order runs for sequential execution

Assign an execution order in which every leading subset of the design
is itself space-filling, so an early stop still leaves a balanced
design:

```python
ordered = sequential.run_order(sampler, extended.best_design)
```

This matters whenever runs are executed serially and the campaign
might be halted early, on a budget cut or an interim analysis.

## Sub-sample and split

Pick a small representative subset of a larger design with
Kennard-Stone selection:

```python
pilot = sequential.subsample(sampler, extended.best_design, n_select=6)
```

Or partition a design into space-filling folds for cross-validation,
so that each fold and each training set covers the whole space:

```python
folds = sequential.k_fold_split(sampler, extended.best_design, k=3)
```

A `nested` constructor builds an inner design that is a subset of a
larger outer design, for multi-fidelity studies where two budget tiers
share one space.
