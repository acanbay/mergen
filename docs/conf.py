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
def _example_outputs_dirs(gallery_conf):
    """Resolve the ``outputs/`` directories the examples write into.

    Sphinx-gallery executes each example with the examples source
    directory as the working directory, so mergen's ``save=True``
    artefacts land in ``<examples_dir>/outputs``. The directories are
    resolved from the gallery configuration because the reset hook
    below runs in a different working directory than the scraper.
    """
    import os

    dirs = gallery_conf.get("examples_dirs")
    if isinstance(dirs, (str, os.PathLike)):
        dirs = [dirs]
    src = str(gallery_conf.get("src_dir", ""))
    resolved = []
    for d in dirs:
        d = str(d)
        p = d if os.path.isabs(d) else os.path.normpath(os.path.join(src, d))
        resolved.append(os.path.join(p, "outputs"))
    return resolved


def _snapshot_outputs(gallery_conf):
    import glob
    import os

    snap = {}
    for out_dir in _example_outputs_dirs(gallery_conf):
        for p in glob.glob(os.path.join(out_dir, "*.png")):
            snap[os.path.abspath(p)] = os.stat(p).st_mtime_ns
    return snap


def _mergen_png_scraper(block, block_vars, gallery_conf):
    """Embed the PNGs that mergen's ``plot(save=True)`` wrote just now.

    Mergen's plot functions save their figure and then close it, so
    sphinx-gallery's matplotlib scraper never sees an open figure.
    This scraper embeds every PNG created or updated since the current
    example started (the baseline snapshot is taken by ``_mergen_reset``
    below), which keeps files left behind by other examples or earlier
    builds out of the page.
    """
    import glob
    import os
    import shutil

    from sphinx_gallery.scrapers import figure_rst

    baseline = _mergen_png_scraper._baseline
    image_names = []
    path_iterator = block_vars["image_path_iterator"]
    for out_dir in _example_outputs_dirs(gallery_conf):
        for png in sorted(glob.glob(os.path.join(out_dir, "*.png"))):
            abspath = os.path.abspath(png)
            mtime = os.stat(png).st_mtime_ns
            if baseline.get(abspath) != mtime:
                baseline[abspath] = mtime
                target = next(path_iterator)
                shutil.copy(png, target)
                image_names.append(target)
    return figure_rst(image_names, gallery_conf["src_dir"])


_mergen_png_scraper._baseline = {}


def _mergen_reset(gallery_conf, fname):
    """Snapshot the outputs directories at the start of each example."""
    _mergen_png_scraper._baseline = _snapshot_outputs(gallery_conf)


# The scraper above is a function object, which Sphinx cannot pickle
# into its config cache; the resulting "config.cache" warning is
# expected and harmless, so it is suppressed explicitly.
suppress_warnings = ["config.cache"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    # Every example executes on every documentation build, so each
    # page shows the full output of the code it displays.
    "filename_pattern": r"/\d{2}_",
    "within_subsection_order": "FileNameSortKey",
    "image_scrapers": ("matplotlib", _mergen_png_scraper),
    # "matplotlib"/"seaborn" keep sphinx-gallery's default per-example
    # cleanup; _mergen_reset primes the PNG scraper (see above).
    "reset_modules": ("matplotlib", "seaborn", _mergen_reset),
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

# ── example page order overrides ────────────────────────────────────
# Sphinx-gallery hard-codes two layout choices this project overrides:
# single-code-block scripts render output above the code, and figures
# render above the captured stdout. Mergen's examples are deliberately
# single-block scripts, and the documented page order here is
# text, full code, full output, figures. Both overrides are narrow,
# verified against sphinx-gallery 0.15-0.21 (see the pin in
# pyproject.toml), and fall back to the default layout instead of
# breaking the build if internals change.
import sphinx_gallery.gen_rst as _sg_gen_rst
import sphinx_gallery.py_source_parser as _sg_parser

try:
    _orig_split = _sg_parser.split_code_and_text_blocks

    def _split_single_code_as_notebook(source_file, return_node=False):
        """Append an empty text block to single-code-block scripts.

        With more than two blocks sphinx-gallery switches to its
        notebook-like layout, which places the code before its output;
        the empty block renders as nothing.
        """
        result = _orig_split(source_file, return_node)
        file_conf, blocks = result[0], list(result[1])
        if len(blocks) == 2 and blocks[1].type == "code":
            blocks.append(_sg_parser.Block("text", "", blocks[-1].lineno))
        return (file_conf, blocks, *result[2:])

    _sg_parser.split_code_and_text_blocks = _split_single_code_as_notebook
    _sg_gen_rst.split_code_and_text_blocks = _split_single_code_as_notebook

    _orig_gco = _sg_gen_rst._get_code_output

    def _stdout_before_images(is_last_expr, example_globals, gallery_conf,
                              logging_tee, images_rst, capture_repr):
        """Emit the captured stdout first and the figures after it."""
        out = _orig_gco(is_last_expr, example_globals, gallery_conf,
                        logging_tee, "", capture_repr)
        if images_rst:
            out = f"{out}\n{images_rst}\n\n"
        return out

    _sg_gen_rst._get_code_output = _stdout_before_images
except (AttributeError, TypeError):  # pragma: no cover
    pass  # future sphinx-gallery: keep its default layout
