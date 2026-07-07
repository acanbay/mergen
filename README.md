# Mergen

**Multi-dimensional Experimental Run GENerator** is a space-filling Design of Experiments library for Python.

[![Tests](https://github.com/acanbay/mergen/actions/workflows/tests.yml/badge.svg)](https://github.com/acanbay/mergen/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## What is Mergen?

Every experimental study works under a fixed budget: only a limited number of runs can be afforded, and each one should be as informative as possible. Mergen selects those runs. Given a parameter space (its bounds, resolution, constraints, and critical regions), it computes the coordinates of *n* points that best cover the space, so that no region is left unexplored and no run is wasted.

The result is not a random sample but a mathematically optimised design, delivered together with a quality report suitable for the methods section of a paper. Mergen supports discrete, continuous, integer, and categorical factors in a single space, three optimisation algorithms, seven space-filling criteria, and a full quality-assessment and export pipeline.

**Typical applications:**

- Physics and engineering (design points for simulations and experiments)
- Chemistry and materials science (synthesis and characterisation points)
- Biology and pharmaceutics (dose-response and formulation studies)
- Machine learning (hyperparameter and benchmark point selection)
- Uncertainty quantification and surrogate modelling (training designs for emulators)
- Data science (representative subset selection from large datasets)

---

## Installation

```bash
pip install git+https://github.com/acanbay/mergen.git
```

Optional Excel export support:

```bash
pip install "git+https://github.com/acanbay/mergen.git#egg=mergen[excel]"
```

**Requirements:** Python 3.9 or newer, NumPy, SciPy, pandas, matplotlib, and Jinja2.

---

## Quick Start

```python
from mergen import ParameterSpace, Sampler

# 1. Define the parameter space
space = ParameterSpace({
    'temperature': range(100, 500, 10),        # discrete
    'pressure':    ('continuous', 0.5, 5.0),   # continuous grid
    'n_layers':    ('integer', 2, 10),         # integer grid
})

# 2. Configure and run the sampler
sampler = Sampler(space)
sampler.set_design()
result = sampler.run()

# 3. Inspect, visualise, export
result.summary()
result.quality_report()
result.plot('pairplot', save=True)
result.to_csv('design.csv')
```

---

## Features

**Parameter types**

- Discrete, continuous (linear or log grid), and integer (linear or log grid) factors
- Categorical factors, both nominal (unordered) and ordinal (ordered)
- Mixed spaces combining any of the above
- Feasibility constraints supplied as user predicates

**Design control**

- Prescribed points: fixed coordinates always included in the design (e.g. runs already carried out)
- Focus regions: a neighbourhood around a critical point where sampling is made denser
- Exclusion zones: regions the design must avoid (e.g. infeasible or unsafe settings)
- User-defined sets: externally chosen points attached as a named set (e.g. a held-out test set) with an optional custom colour
- Existing designs: a previously generated design can be loaded as-is, and only the validation and extra sets are generated around it

**Optimisation**

- Three algorithms and seven space-filling criteria (see the selection guide below)
- Iterated local search restarts for robustness
- Optional multi-algorithm runs executed in parallel

**Sequential design**

- Extend an existing design with new points
- Fill in around external data
- Ordered run sequences, k-fold splits, and nested designs

**Export**

- CSV, JSON, Markdown, LaTeX, HTML, and Excel

---

## Examples

Worked examples are coming soon in the [`examples/`](examples/) directory.

---

## Choosing an Algorithm

All three algorithms optimise the same criteria and return a design of the requested size; they differ in how they search and in the trade-off between speed and quality. Select one through `set_optimizer`.

| Algorithm | Key     | When to use |
|-----------|---------|-------------|
| SA        | `'sa'`  | General-purpose default. Simulated annealing accepts occasional worsening moves to escape local optima, giving a good balance of speed and quality for most designs. |
| SCE       | `'sce'` | Small to medium designs where fast convergence matters. Stochastic coordinate exchange improves one coordinate at a time and reaches a good design quickly. |
| ESE       | `'ese'` | Large or difficult designs where quality matters more than runtime. The enhanced stochastic evolutionary search adapts its acceptance threshold and explores more thoroughly. |

---

## Choosing a Criterion

The criterion defines what "well spread" means. The right choice depends on your factor types and on whether you care about coverage of the full space, spread within projections, or point separation. Select one through the `criteria` argument of `run`.

| Criterion     | Key            | Factor types     | When to use |
|---------------|----------------|------------------|-------------|
| uMaxPro       | `'umaxpro'`    | Numerical        | Default choice. Keeps points well spread in every projection and improves behaviour near the boundaries with a periodic distance. |
| MaxPro        | `'maxpro'`     | Numerical        | Good spread in all lower-dimensional projections is the priority (e.g. when some factors may turn out inactive). |
| $\phi_p$      | `'phi_p'`      | Numerical        | The goal is maximum separation between points (a maximin design). |
| CD2           | `'cd2'`        | Numerical        | Uniform coverage of the whole space matters most; based on the centred L2 discrepancy. |
| Stratified L2 | `'stratified'` | Numerical        | Balanced uniformity across nested strata as well as overall. |
| MaxProQQ      | `'maxproqq'`   | Numerical and nominal | Projection quality is the priority. |
| QQD           | `'qqd'`        | Numerical and nominal | Uniform coverage (a discrepancy measure) is the priority. |

The numerical criteria (uMaxPro, MaxPro, $\phi_p$, CD2, Stratified L2) also accept ordinal factors, since ordered levels carry a meaningful distance. Nominal (unordered) factors require `maxproqq` or `qqd`.

---

## Interpreting the Quality Report

`result.quality_report()` scores the design on six standard space-filling metrics and, optionally, against a Monte Carlo baseline of random designs of the same size. The metrics fall into three groups.

**Separation and coverage**

- Minimum distance: the smallest distance between any two points. Higher is better; it measures how well points are separated and guards against clustering.
- Minimax distance: the largest distance from any location in the space to its nearest design point. Lower is better; it measures how well the design covers empty regions.

**Spacing uniformity**

- Coefficient of variation of distances: the spread of pairwise distances relative to their mean. Lower is better; it indicates more regular, even spacing.
- Mean distance: the average pairwise distance. Higher values indicate a more dispersed design; read it alongside the coefficient of variation rather than on its own.

**Projection quality**

- Maximum absolute correlation: the largest absolute correlation between any two factor columns. Lower is better; near zero means the factors are close to orthogonal.
- 2D projection CD2: the average centred L2 discrepancy over all two-factor projections. Lower is better; it measures uniformity within projections.

**Monte Carlo baseline**

When `mc_samples` is set, each metric is also reported as a percentile against random designs of the same size. A percentile near 100 means the optimised design outperforms almost all random designs on that metric, which is the outcome to expect from a good design.

---

## References

- Hickernell, F. J. (1998). A generalized discrepancy and quadrature error bound. *Mathematics of Computation*, 67(221), 299-322.
- Jin, R., Chen, W., & Sudjianto, A. (2005). An efficient algorithm for constructing optimal design of computer experiments. *Journal of Statistical Planning and Inference*, 134(1), 268-287.
- Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs for computer experiments. *Biometrika*, 102(2), 371-380.
- Joseph, V. R., Gul, E., & Ba, S. (2019). Designing computer experiments with multiple types of factors: The MaxPro approach. *Journal of Quality Technology*, 52(4), 343-354.
- Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments. *Technometrics*, 11(1), 137-148.
- Meyer, R. K., & Nachtsheim, C. J. (1995). The coordinate exchange algorithm for constructing exact optimal experimental designs. *Technometrics*, 37(1), 60-69.
- Morris, M. D., & Mitchell, T. J. (1995). Exploratory designs for computational experiments. *Journal of Statistical Planning and Inference*, 43(3), 381-402.
- Tian, Y., & Xu, H. (2025). A minimum aberration-type criterion for space-filling designs. *Journal of the Royal Statistical Society: Series B*, 88(2).
- Vořechovský, M., & Eliáš, J. (2026). Uniform maximum projection designs. *Computers & Structures*.
- Wilson, D. R., & Martinez, T. R. (1997). Improved heterogeneous distance functions. *Journal of Artificial Intelligence Research*, 6, 1-34.
- Zhang, M., Yang, F., & Zhou, Y.-D. (2021). Uniformity criterion for designs with both qualitative and quantitative factors. *arXiv:2101.02416*.

---

## Citation

If you use Mergen in your research, please cite it using the information in [`CITATION.cff`](CITATION.cff).

---

## License

MIT © Ali Can Canbay
