# Installation

::::{tab-set}

:::{tab-item} pip
```bash
pip install mergen-doe
```

Optional Excel export support:

```bash
pip install "mergen-doe[excel]"
```
:::

:::{tab-item} conda
A conda-forge package is planned. Until it is published, pip can be
used inside conda environments:

```bash
pip install mergen-doe
```

Optional Excel export support:

```bash
pip install "mergen-doe[excel]"
```
:::

:::{tab-item} source
For the latest development state, or to contribute:

```bash
git clone https://github.com/acanbay/mergen.git
cd mergen
pip install -e .
```

Development and documentation extras:

```bash
pip install -e ".[excel]"  # openpyxl, for result.to_excel()
pip install -e ".[dev]"    # pytest, coverage, ruff, black
pip install -e ".[docs]"   # Sphinx toolchain
```
:::

::::

## Package name and import name

The distribution name is `mergen-doe` on both PyPI and conda-forge.
The import name is `mergen`, regardless of the installation method:

```python
import mergen
```

The distinction between a distribution name and an import name is
common in the Python ecosystem; the `scikit-learn` distribution, for
example, is imported as `sklearn`.

## Requirements

Python 3.9 or newer. The core dependencies (NumPy, SciPy, pandas,
matplotlib, Jinja2) are installed automatically.

## Verify the installation

```bash
python -c "import mergen; print(mergen.__version__)"
```

A version string confirms the package is importable and ready.
