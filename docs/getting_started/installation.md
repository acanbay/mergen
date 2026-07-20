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

`excel` is the only extra meant for installed packages; the `dev`
and `docs` extras belong to a source checkout (source tab), since the
tests and the documentation sources ship with the repository rather
than the package.

:::

:::{tab-item} conda

```bash
conda install -c conda-forge mergen-doe
```

The conda-forge package installs the core dependencies automatically.
Excel export support requires `openpyxl`, which can be installed from
conda-forge as well:

```bash
conda install -c conda-forge mergen-doe openpyxl
```

The `dev` and `docs` extras are not part of the conda-forge package;
they are only needed to work on Mergen itself, from a source checkout
(source tab).

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
pip install -e ".[excel]"   # openpyxl, for result.to_excel()
pip install -e ".[dev]"     # pytest, coverage, ruff, black
pip install -e ".[docs]"    # Sphinx toolchain
```

:::

::::

## Package name and import name

The distribution name is `mergen-doe`; the import name is `mergen`,
regardless of the installation method:

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
