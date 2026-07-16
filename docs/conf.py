# Configuration file for the Sphinx documentation builder.
#
# Design notes:
# - No sys.path hacks and no OS-specific paths: the package is expected
#   to be installed (``pip install .[docs]``), as ReadTheDocs and local
#   builds both do. Metadata is read via importlib.metadata with a safe
#   fallback so the config never crashes on a partial environment.
# - Only lower-bounded dependencies are assumed (see pyproject.toml);
#   nothing here relies on version-specific Sphinx behaviour.
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# ── Project information ─────────────────────────────────────────────
project = "Mergen"
author = "Ali Can Canbay"
copyright = "2026, Ali Can Canbay"  # noqa: A001 (Sphinx convention)

try:
    release = _pkg_version("mergen-doe")
except PackageNotFoundError:  # e.g. docs linting without the package
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# ── General configuration ───────────────────────────────────────────
extensions = [
    "myst_nb",                     # Markdown + executable notebooks (jupytext)
    "sphinx.ext.autodoc",          # API reference from docstrings
    "sphinx.ext.autosummary",      # summary tables for the API pages
    "sphinx.ext.napoleon",         # numpy-style docstrings
    "sphinx.ext.intersphinx",      # cross-links to numpy/scipy/pandas docs
    "sphinx.ext.viewcode",         # [source] links
    "sphinx_copybutton",           # copy button on code blocks
    "sphinx_design",               # grids/cards for the landing page
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute"]

# ── MyST / myst-nb ──────────────────────────────────────────────────
myst_enable_extensions = [
    "colon_fence",   # ::: directives
    "dollarmath",    # $...$ and $$...$$
    "deflist",
]
# Notebook execution is disabled while the skeleton is being built.
# It will be switched to "cache" when the example pages are wired in,
# so ReadTheDocs re-executes only changed notebooks and stays within
# its build time limits.
nb_execution_mode = "off"

# ── autodoc / autosummary / napoleon ────────────────────────────────
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

# ── intersphinx ─────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# ── HTML output ─────────────────────────────────────────────────────
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"Mergen {release}"
html_theme_options = {
    "github_url": "https://github.com/acanbay/mergen",
    "navbar_align": "content",
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
