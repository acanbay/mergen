# Mergen

**M**ulti-dimensional **E**xperimental **R**un **GEN**erator: a Python
module for space-filling Design of Experiments.

You have a system with adjustable settings and a budget of *n*
experiments. Mergen chooses which *n* combinations of settings to run,
spreads them so the whole range of possibilities is represented, and
proves the quality of the result.

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

result.quality_report()      # publication-ready quality evidence
result.plot('pairplot')      # visual check of the design
result.to_csv()              # coordinates, ready to run
```

## Why Mergen

Suppose you study a system with several adjustable settings: a reactor
with a temperature, a pressure, and a choice of catalyst, or a machine
learning model with a learning rate and a batch size. Every
combination of settings is one possible experiment, and together the
combinations form the *parameter space* of the study. Trying them all
is impossible; a realistic budget is a few dozen runs against millions
of combinations.

A few dozen can nevertheless be enough, because the goal is not to
test every combination but to learn how the system behaves. From the
results of well-placed runs one fits a model of the response, whether
a simple fit, a Gaussian process, or a machine learning regressor, and
that model then predicts the outcome at all the combinations that were
never run. How trustworthy those predictions are is decided largely
before any experiment happens: by where the runs were placed. Runs
that cluster teach the model the same thing twice; regions with no
runs are regions where the model has no information.

Which runs, then? The intuitive answers all waste budget. Varying
one setting at a time never reveals how settings interact. A regular
grid spends most of its runs repeating the same few values of each
setting. Random picking clusters some runs together and leaves other
regions untouched. The established remedy is a *space-filling design*:
choose the runs so that they spread evenly through the parameter
space, every region is represented, and no two runs duplicate each
other's information.

Mergen builds such designs, and it builds them for the spaces real
studies actually have: with combinations that are infeasible or
forbidden, with runs that were already performed and must be kept,
with critical regions that deserve extra attention, and with settings
that are categories rather than numbers. And because a design you
cannot defend is not finished, every design leaves the package with a
statistical quality report: your experiment plan, and the evidence
that it was well chosen, as one object.

## What Mergen offers

Seven optimisation criteria, including recent developments from the
literature such as uMaxPro (Vorechovsky & Masek, 2026) and the
stratified L2-discrepancy (Tian & Xu, 2025). Three optimisers (simulated annealing,
stochastic coordinate exchange, enhanced stochastic evolutionary) over
a discrete-grid Latin-hypercube structure. Mixed parameter types:
discrete, continuous (linear or log), integer, nominal, ordinal.
Feasibility constraints, exclusion zones, focus regions, and prescribed
points. A `compare()` sweep that ranks criterion × algorithm
combinations by Pareto/Utopia ordering when you are unsure what to
pick. Sequential utilities to extend, sub-sample, reorder, and nest
existing designs.

## Documentation

::::{grid} 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Learning-oriented. Start here if you are new to Mergen or to design of
experiments: build your first design and learn to read its output.
:::

:::{grid-item-card} How-to guides
:link: how-to/index
:link-type: doc

Task-oriented. Recipes for concrete goals: parameter types,
constraints, comparing designs, sequential workflows, exports.
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc

Understanding-oriented. Which criterion for which problem, which
optimiser, what the quality metrics mean, how large a design to build.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Information-oriented. Every public class and function, generated from
the numpy-style docstrings.
:::

::::

## Installation

Installation options and requirements are described on the
{doc}`installation` page.

## References

Tian, Y., & Xu, H. (2025). A stratified L2-discrepancy with application
to space-filling designs. *Journal of the Royal Statistical Society,
Series B*.

Vorechovsky, M., & Masek, J. (2026). Uniform maximum projection designs
for computer experiments. *Computers & Structures*.

```{toctree}
:hidden:
:maxdepth: 2

installation
tutorials/index
how-to/index
explanation/index
api/index
```
