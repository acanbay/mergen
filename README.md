# mergen

**Multi-dimensional Experimental Run GENerator**

Space-filling Design of Experiments for Python.

[![Tests](https://github.com/acanbay/mergen/actions/workflows/tests.yml/badge.svg)](https://github.com/acanbay/mergen/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Why Mergen?

Every experimental study faces the same constraint: limited resources. Which parameter combinations should be tested? How many experiments are feasible?

Mergen answers this question: it finds the coordinates of the limited number of points from which the maximum information about the entire parameter space can be obtained. Not random sampling — a mathematically optimised design.

**Use cases:**

- **Physics** — sampling point selection for simulations and experiments
- **Engineering** — sampling point selection in numerical simulation and experimental studies
- **Chemistry & materials science** — selection of synthesis and characterisation points
- **Biology & pharmaceutics** — sampling point selection in dose-response and formulation studies
- **Machine learning** — selection of hyperparameter combinations and benchmark points
- **Uncertainty quantification & surrogate modelling** — sampling point selection for emulator training
- **Data science** — representative subset selection from large datasets

---

## Installation

```bash
pip install git+https://github.com/acanbay/mergen.git
```

For Excel export support:

```bash
pip install "git+https://github.com/acanbay/mergen.git#egg=mergen[excel]"
```

**Requirements:** Python ≥ 3.9, NumPy, SciPy, pandas, matplotlib, Jinja2

---

## Quick Start

```python
import numpy as np
from mergen import ParameterSpace, Sampler

# Define the parameter space
space = ParameterSpace({
    'temperature': range(100, 500, 10),       # discrete
    'pressure':    ('continuous', 0.5, 5.0),  # linear grid
    'n_layers':    ('integer', 2, 10),        # integer
})

# Optional: add constraints
space.add_constraint(lambda p: p['temperature'] * p['pressure'] < 1500)

# Create sampler and run
sampler = Sampler(space)
sampler.set_design(n_samples=30)
result = sampler.run(seed=44)

# Inspect results
result.summary()
result.quality_report()

# Visualise
result.plot('pairplot')
result.plot('distances')

# Export
result.to_csv('design.csv')
result.to_markdown('report.md')
```

---

## Features

**Parameter types**
- Discrete, continuous (linear/log), and integer (linear/log) parameters
- Mixed parameter spaces
- Feasibility constraints

**Optimisation**
- Simulated Annealing with multiple restarts
- Five space-filling criteria: uMaxPro, MaxPro, φ_p, CD2, Stratified L2
- Prescribed points, focus regions, and exclusion zones

**Quality assessment**
- Six standard space-filling metrics: min distance, minimax distance, max |correlation|, 2D projection CD2, CV of distances, mean distance
- Runtime Monte Carlo baseline comparison

**Sequential design**
- Augment an existing design with new points
- Complement design around existing data
- Load designs from CSV/DataFrame

**Export**
- CSV, JSON, Markdown, LaTeX, HTML, Excel

---

## Advanced Usage

```python
from mergen import ParameterSpace, Sampler
from mergen.sequential import augment, complement, n_samples_recommendation

# Get a sample size recommendation
rec = n_samples_recommendation(space, budget=50)

# Prescribed points (always included)
sampler.add_prescribed([[300, 2.5, 5]], in_design=True, in_sa=False)

# Focus region (denser sampling near a critical point)
sampler.add_focus([400, 4.5, 8], spread=1.5, in_design=True, in_sa=True)

# Exclusion zone (avoid a problematic region)
sampler.add_exclusion([100, 0.5, 2], spread=1.0)

# Multiple restarts for better optimisation
sampler.set_sa(n_restarts=5)
result = sampler.run(criteria='umaxpro', seed=44)

# Augment an existing design
result2 = augment(result, n_add=10, seed=44)

# Complement around external data
import numpy as np
existing = np.array([[200, 2.0, 4], [400, 4.0, 8]])
result3 = complement(space, existing, n_samples=20, seed=44)
```

---

## References

| Algorithm | Reference |
|-----------|-----------|
| uMaxPro | Vorechovsky & Elias (2026), *Computers & Structures* |
| MaxPro | Joseph, Gul & Ba (2015), *Biometrika* 102(2) |
| SA, φ_p | Morris & Mitchell (1995), *J. Statist. Plan. Infer.* 43 |
| CD2 | Hickernell (1998), *Math. Comp.* 67 |
| Stratified L2 | Tian & Xu (2025), *JRSS-B* 88(2) |
| Validation set | Kennard & Stone (1969), *Technometrics* 11(1) |

---

## Citation

If you use Mergen in your research, please cite it using the information in [`CITATION.cff`](CITATION.cff).

---

## License

MIT © Ali Can Canbay