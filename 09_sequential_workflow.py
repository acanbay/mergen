"""
A staged, sequential campaign
=============================

Extend, reorder and subsample a base design as an experimental campaign grows in stages.

An experimental campaign that unfolds in stages: an initial design is
run, results come back, and the design is then grown and reorganised
for the next batch. This example walks through Mergen's sequential
toolkit end to end, printing the row count at each step so the staging
is easy to follow.

Parameters
----------
- x1, x2 (discrete grids): two inputs on explicit grids so every
  design point is an exact grid node throughout the staged workflow.

What to look at
---------------
- The printed row count after each step: the base design, the extended
  design, the reordered design, and the small subsampled pilot. These
  counts are how the staging is made legible.
- The two saved pairplots (base and extended): the extended design
  keeps the original points and fills the remaining gaps, rather than
  starting over.
- ``ordered_design.csv``: the final design with an added run_order column,
  ready as an execution list for the next batch.

Mergen features used
--------------------
- mergen.sequential.extend: augment an existing design with new,
  space-filling points without discarding the originals.
- mergen.sequential.run_order: assign an execution order that is
  robust to drift, so early runs already cover the space.
- mergen.sequential.subsample: pick a small representative subset (a
  pilot) from a larger design.
- mergen.sequential.k_fold_split: partition the design into folds for
  cross-validation.

Estimated runtime: a minute or two.
"""
from mergen import ParameterSpace, Sampler
from mergen import sequential

# 1. Build a base design for the first batch.
space = ParameterSpace({
    'x1': range(0, 101, 5),
    'x2': range(0, 101, 5),
})
sampler = Sampler(space)
sampler.set_design(n_samples=15)
base = sampler.run()
print(f"base design           : {len(base.best_design)} points")

# 2. Results came back; grow the design with 10 more points. extend()
#    keeps the original points and adds space-filling ones around them.
extended = sequential.extend(sampler, base.best_design, n_new=10)
print(f"after extend          : {len(extended.best_design)} points")

# 3. Assign a drift-robust execution order to the combined design.
ordered = sequential.run_order(sampler, extended.best_design)
print(f"ordered design        : {len(ordered)} rows (run_order column added)")

# 4. Pick a small representative pilot from the full design.
pilot = sequential.subsample(sampler, extended.best_design, n_select=6)
print(f"subsampled pilot      : {len(pilot)} points")

# 5. Partition the design into folds for cross-validation.
folds = sequential.k_fold_split(sampler, extended.best_design, k=3)
print(f"k-fold split          : {len(folds)} folds")

# 6. Save the two coverage views and the final ordered run list.
base.plot('pairplot', save=True)
extended.plot('pairplot', save=True)
ordered.to_csv('outputs/ordered_design.csv', index=False)
