"""
mergen
======
Multi-dimensional Experimental Run GENerator

Space-filling Design of Experiments for Python.

Mergen generates optimal sampling coordinates for any parameter space —
discrete, continuous, integer, or mixed — and provides rigorous quality
assessment via standard space-filling metrics.

This is the algorithm-independent core. Optimisation algorithms (SA,
SCE, ESE) live in the ``mergen.algorithms`` subpackage (added later).

Quick start
-----------
::

    from mergen import ParameterSpace, Sampler

    space = ParameterSpace({
        'temperature': range(100, 400, 10),
        'pressure':    ('continuous', 0.5, 5.0),
    })
    sampler = Sampler(space)
    sampler.set_design(n_samples=30)
    result = sampler.run()

References
----------
Vorechovsky & Masek (2026), Computers & Structures.       [uMaxPro]
Joseph, Gul & Ba (2015), Biometrika 102(2).               [MaxPro]
Morris & Mitchell (1995), J. Statist. Plan. Infer. 43.    [phi_p]
Hickernell (1998), Math. Comp. 67.                        [CD2]
Tian & Xu (2025), JRSS-B 88(2).                           [Stratified L2]
Kennard & Stone (1969), Technometrics 11(1).              [Validation set]
"""

# ── Version (single source of truth — pyproject.toml reads from here) ────
__version__ = "0.1.0"

# ── Package identity ──────────────────────────────────────────────────────
__fullname__    = "Multi-dimensional Experimental Run GENerator"
__description__ = "Space-filling Design of Experiments for Python"
__license__     = "MIT"

# ── Authors ───────────────────────────────────────────────────────────────
__authors__ = [
    {
        "name" : "Ali Can Canbay",
        "email": "acanbay@ankara.edu.tr",
        "orcid": "https://orcid.org/0000-0003-4602-473X",
    },
]

# ── URLs ──────────────────────────────────────────────────────────────────
__github__ = "https://github.com/acanbay/mergen"
__docs__   = None   # ReadTheDocs URL — add when available

# ── DOI ───────────────────────────────────────────────────────────────────
__doi__ = {
    "software": None,    # Zenodo DOI — add after first release
    "papers"  : [],      # JOSS and other publications
}

# ── Public API (algorithm-independent core + algorithm registry + Sampler) ─
from .space      import ParameterSpace
from .criteria   import get_criterion, list_criteria
from .algorithms import (
    BaseOptimizer,
    OptimisationResult,
    get_optimizer,
    list_optimizers,
    register_optimizer,
)
from .sampler    import (
    Sampler,
    SamplingResult,
    ComparisonResult,
    FocusPoint,
    ExclusionPoint,
)
from . import sequential
from . import distances

__all__ = [
    "ParameterSpace",
    "get_criterion",
    "list_criteria",
    "BaseOptimizer",
    "OptimisationResult",
    "get_optimizer",
    "list_optimizers",
    "register_optimizer",
    "Sampler",
    "SamplingResult",
    "ComparisonResult",
    "FocusPoint",
    "ExclusionPoint",
    "sequential",
    "distances",
]


# ── Info banner ───────────────────────────────────────────────────────────
def _banner() -> str:
    """Return a formatted package info string for terminal and text output."""
    lines = [
        f"mergen v{__version__} — {__fullname__}",
        f"Authors : {', '.join(a['name'] for a in __authors__)}",
        f"License : {__license__}",
        f"GitHub  : {__github__}",
    ]
    if __docs__:
        lines.append(f"Docs    : {__docs__}")
    if __doi__["software"]:
        lines.append(f"DOI     : {__doi__['software']}")
    for paper in __doi__["papers"]:
        if paper.get("doi"):
            title = paper.get("title", "")
            lines.append(f"Paper   : {paper['doi']}"
                         + (f"  ({title})" if title else ""))
    return "\n".join(lines) + "\n"


def info() -> None:
    """Print package information to stdout."""
    print(_banner())


# Print banner on import.
# To suppress: set environment variable MERGEN_SILENT=1 before importing.
import os as _os  # noqa: E402
if not _os.environ.get('MERGEN_SILENT'):
    print(_banner())
