"""
mergen.criteria
===============
Optimisation criteria for space-filling design.

Each criterion exposes four methods:

    evaluate(X, space)
        Full computation on a normalised design matrix. Used once
        per algorithm restart to obtain the initial score.

    incremental(X, i, new_pt, space, current_score)
        O(n*d) update when the entire point *i* is replaced by
        ``new_pt``. Used by single-point swap operators.

    begin_1d(X, i, current_score)
        Initialise a per-point cache for fast 1D coordinate-exchange
        trials.

    try_1d(cache, axis, new_value, space)
        O(n) update when only one coordinate changes. Inner-loop
        operation of the SCE algorithm.

All criteria operate on **normalised** coordinates in [0, 1]^d.

Available criteria
------------------
    'umaxpro'     UMaxPro       — Vorechovsky & Masek (2026)  (default)
    'maxpro'      MaxPro        — Joseph, Gul & Ba (2015)
    'phi_p'       PhiP (p=15)   — Morris & Mitchell (1995)
    'cd2'         CD2           — Hickernell (1998)
    'stratified'  StratifiedL2  — Tian & Xu (2025)

Usage
-----
    from mergen.criteria import get_criterion

    crit  = get_criterion('umaxpro')
    score = crit.evaluate(X_norm, space)
"""

from __future__ import annotations

from .base       import BaseCriterion, _EPS  # noqa: F401
from .umaxpro    import UMaxPro
from .maxpro     import MaxPro
from .maxproqq   import MaxProQQ
from .phi_p      import PhiP
from .cd2        import CD2
from .qqd        import QQD
from .stratified import StratifiedL2


__all__ = [
    "BaseCriterion",
    "UMaxPro",
    "MaxPro",
    "MaxProQQ",
    "PhiP",
    "CD2",
    "QQD",
    "StratifiedL2",
    "get_criterion",
    "list_criteria",
    "criterion_latex",
]


# ── Registry of built-in criterion names -> class ────────────────────
_REGISTRY: dict = {
    'umaxpro'       : UMaxPro,
    'maxpro'        : MaxPro,
    'maxproqq'      : MaxProQQ,
    'maxpro_qq'     : MaxProQQ,     # alias
    'phi_p'         : PhiP,
    'phip'          : PhiP,         # alias
    'cd2'           : CD2,
    'qqd'           : QQD,
    'stratified'    : StratifiedL2,
    'stratified_l2' : StratifiedL2,  # alias
}


# LaTeX math labels for the criteria, for use in rendering targets
# (plots, LaTeX / Markdown / HTML exports). Plain-text targets
# (terminal, CSV, Excel) keep the registry name. Notation follows the
# source papers: phi_p (Morris & Mitchell 1995); MaxPro (Joseph, Gul &
# Ba 2015); centered L2-discrepancy CD_2 (Hickernell 1998); stratified
# L2-discrepancy SL_2 (Tian & Xu 2025).
_CRITERION_LATEX = {
    'umaxpro'    : r'$\mathrm{UMaxPro}$',
    'maxpro'     : r'$\mathrm{MaxPro}$',
    'maxproqq'   : r'$\mathrm{MaxPro}_{\mathrm{QQ}}$',
    'phi_p'      : r'$\phi_p$',
    'cd2'        : r'$\mathrm{CD}_2$',
    'qqd'        : r'$\mathrm{QQD}$',
    'stratified' : r'$\mathrm{SL}_2$',
}


def criterion_latex(name: str) -> str:
    """Return the LaTeX math label for a criterion, or the name itself.

    Aliases resolve to their canonical label. Plain-text output targets
    (terminal, CSV) should use the raw name instead of this label.
    """
    aliases = {'maxpro_qq': 'maxproqq', 'phip': 'phi_p',
               'stratified_l2': 'stratified'}
    key = aliases.get(name, name)
    return _CRITERION_LATEX.get(key, name)


def get_criterion(name: str) -> BaseCriterion:
    """
    Instantiate a criterion by name.

    Parameters
    ----------
    name : str
        One of: ``'umaxpro'``, ``'maxpro'``, ``'phi_p'``, ``'cd2'``,
        ``'stratified'`` (or ``'stratified_l2'``).
        Case-insensitive.

    Returns
    -------
    BaseCriterion instance

    Raises
    ------
    ValueError if *name* is not recognised.

    Examples
    --------
    >>> crit = get_criterion('umaxpro')
    >>> type(crit).__name__
    'UMaxPro'
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = ', '.join(
            f"'{k}'" for k in sorted(
                set(_REGISTRY.keys())
                - {'phip', 'stratified_l2', 'maxpro_qq'}
            )
        )
        raise ValueError(
            f"\n\033[0;31m[MERGEN ERROR]\033[0m  "
            f"Unknown criterion '{name}'.\n"
            f"  Available: {available}"
        )
    return _REGISTRY[key]()


def list_criteria() -> list:
    """Return the list of available criterion names (canonical, no aliases)."""
    return sorted(set(_REGISTRY.keys())
                  - {'phip', 'stratified_l2', 'maxpro_qq'})


def nominal_supporting_criteria() -> list:
    """
    Return the canonical names of criteria that support nominal
    (unordered categorical) factors.

    Used by :class:`~mergen.sampler.Sampler` to build the diagnostic
    message when a user requests a criterion incompatible with a
    space that contains nominal columns.
    """
    seen = set()
    out  = []
    for name, cls in _REGISTRY.items():
        if name in {'phip', 'stratified_l2', 'maxpro_qq'}:
            continue                       # aliases
        if cls in seen:
            continue
        if getattr(cls, 'supports_nominal', False):
            out.append(name)
            seen.add(cls)
    return sorted(out)
