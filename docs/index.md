# Mergen

**M**ulti-dimensional **E**xperimental **R**un **GEN**erator: a Python
module for space-filling Design of Experiments.

Every experimental study works under a fixed budget: only a limited
number of runs can be afforded, and each one should be as informative
as possible. Mergen selects those runs. Given a parameter space (its
bounds, resolution, constraints and critical regions), it computes
the coordinates of *n* points that spread evenly through the space,
so that no region is left unexplored and no run is wasted.

The result is not a random sample but a mathematically optimised
design, delivered together with a statistical quality report suitable
for the methods section of a paper. Mergen handles the spaces real
studies actually have: mixed factor types, forbidden regions, runs
that already exist, and zones that deserve extra attention. The full
reasoning lives in {doc}`Why space-filling designs?
<user_guide/explanation/why_space_filling>`.

{doc}`Installation <getting_started/installation>` ·
{doc}`Quickstart tutorial <getting_started/first_design>` ·
{doc}`Examples <auto_examples/index>` ·
{doc}`API reference <api/index>`

```python
import mergen

space = mergen.ParameterSpace({
    'temperature': ('continuous', 300, 500),
    'pressure':    ('continuous', 1.0, 5.0),
    'catalyst':    [0.1, 0.2, 0.5, 1.0],
})
sampler = mergen.Sampler(space)
sampler.set_design(n_samples=30)
result = sampler.run(criteria='umaxpro', algorithm='sa')

result.quality_report()      # statistical quality evidence
result.plot('pairplot')      # visual check of the design
result.to_csv()              # coordinates, ready to run
```

Under the hood: seven optimisation criteria (including recent
developments such as uMaxPro and the stratified L2-discrepancy),
three optimisers over a discrete-grid Latin-hypercube structure,
mixed parameter types, constraints, focus regions, prescribed points,
sequential extension, and a `compare()` sweep for when you are unsure
what to pick.

## Documentation

::::{grid} 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install the package and build your first design in a short, guided
tutorial; then learn to read what Mergen gives you back.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc

Task recipes and the reasoning behind the choices: which criterion,
which optimiser, how many runs, and what every object and setting
does.
:::

:::{grid-item-card} Examples
:link: auto_examples/index
:link-type: doc

Fifteen complete studies from different domains, executed end to end,
each downloadable as a script or a notebook.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every public class and function, generated from the numpy-style
docstrings.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
user_guide/index
auto_examples/index
api/index
about/index
```
