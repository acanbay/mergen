# Installation

Mergen is installed from its source repository:

```bash
git clone https://github.com/acanbay/mergen.git
cd mergen
pip install .
```

For development, install in editable mode with the extras you need:

```bash
pip install -e .            # the package itself
pip install -e ".[excel]"   # openpyxl, for result.to_excel()
pip install -e ".[dev]"     # pytest, coverage, ruff, black
pip install -e ".[docs]"    # Sphinx toolchain
```

## Package name and import name

The distribution name is `mergen-doe`; the import name is `mergen`:

```python
import mergen
```

The distinction between a distribution name and an import name is
common in the Python ecosystem; the `scikit-learn` distribution, for
example, is imported as `sklearn`.

## Requirements

Python 3.9 or newer. The core dependencies (NumPy, SciPy, pandas,
matplotlib, joblib, tabulate, Jinja2) are installed automatically.

## Verify the installation

```bash
python -c "import mergen; print(mergen.__version__)"
```

A version string confirms the package is importable and ready.
