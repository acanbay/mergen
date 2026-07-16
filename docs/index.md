# Mergen

**M**ulti-dimensional **E**xperimental **R**un **GEN**erator: a Python
module for space-filling Design of Experiments.

You own the parameter space: its ranges, its constraints, its critical
regions. Mergen finds the *n* experiment coordinates that represent that
space best, and proves the quality of the result.

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

Every design ships with a quality report benchmarked against a runtime
Monte-Carlo baseline, so the Methods section of your paper can state,
with numbers, how good the design is.

Mergen is currently installed from source
(`git clone` the repository, then `pip install -e .`); PyPI and
conda-forge releases will follow the first public version.

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

## References

Tian, Y., & Xu, H. (2025). A stratified L2-discrepancy with application
to space-filling designs. *Journal of the Royal Statistical Society,
Series B*.

Vorechovsky, M., & Masek, J. (2026). Uniform maximum projection designs
for computer experiments. *Computers & Structures*.

```{toctree}
:hidden:
:maxdepth: 2

tutorials/index
how-to/index
explanation/index
api/index
```
