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
    "myst_nb",                     # Markdown pages (MyST) via myst-nb
    "sphinx.ext.autodoc",          # API reference from docstrings
    "sphinx.ext.autosummary",      # summary tables for the API pages
    "sphinx.ext.napoleon",         # numpy-style docstrings
    "sphinx.ext.intersphinx",      # cross-links to numpy/scipy/pandas docs
    "sphinx.ext.viewcode",         # [source] links
    "sphinx_copybutton",           # copy button on code blocks
    "sphinx_design",               # grids/cards for the landing page
    "sphinx_gallery.gen_gallery",  # executed example gallery
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", "jupyter_execute",
    # sphinx-gallery writes .rst pages next to generated .ipynb/.py
    # download artefacts; only the .rst is a Sphinx source. Excluding
    # the artefacts prevents "multiple files found" source clashes.
    "auto_examples/*.ipynb",
    "auto_examples/*.py",
]

# ── MyST / myst-nb ──────────────────────────────────────────────────
myst_enable_extensions = [
    "colon_fence",   # ::: directives
    "dollarmath",    # $...$ and $$...$$
    "deflist",
]
# Handwritten pages contain no executable notebooks; all example
# execution is handled by sphinx-gallery below, so myst-nb stays off.
nb_execution_mode = "off"

# ── sphinx-gallery ──────────────────────────────────────────────────
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    # Execute only the light examples on documentation builds (their
    # measured total stays within the ReadTheDocs build limit); the
    # heavy studies (04, 05, 12, 14) are rendered without execution
    # and the full set is exercised weekly in CI.
    "filename_pattern": r"/(0[1236789]|1[0135])_",
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
}

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
html_css_files = ["custom.css"]
html_title = f"Mergen {release}"
html_favicon = "_static/favicon.png"
# The landing page belongs to no section, so its section navigation is
# empty; hide the primary sidebar there explicitly for a clean,
# full-width landing (the pandas convention) instead of an empty gutter.
html_sidebars = {"index": []}

html_theme_options = {
    # Brand logo: the theme swaps the image automatically when the
    # user toggles between light and dark mode. When a logo is set,
    # the theme hides the "Mergen <release>" navbar text in its place.
    "logo": {
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
        "alt_text": "Mergen documentation - Home",
    },
    "github_url": "https://github.com/acanbay/mergen",
    "navbar_align": "content",
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
