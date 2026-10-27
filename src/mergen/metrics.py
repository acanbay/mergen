"""
mergen.metrics
==============
Design quality metrics and optional Monte Carlo baseline comparison.

Usage
-----
    result.quality_report()
    result.quality_report(metrics=['min_distance', 'cd2'])
    result.quality_report(mc_samples=500)
    result.quality_report(criteria_metrics=['maxpro', 'phi_p'], mc_samples=500)

Metrics (always computed from the design, no MC needed)
--------------------------------------------------------
    min_distance         Johnson et al. (1990) — worst-case separation
    minimax              Johnson et al. (1990) — fill distance / coverage
    max_abs_correlation  — orthogonality between parameters
    projection_cd2       Sun et al. (2019) — 2D projection uniformity
    cd2                  Hickernell (1998) — whole-design uniformity (priority)
    cv_distances         Lekivetz & Jones (2015) — clustering index
    mean_distance        — average pairwise distance

Plus the criterion used in sampler.run() is always included.

criteria_metrics (optional, computed from the design)
------------------------------------------------------
    Any SA criterion name: 'umaxpro', 'maxpro', 'phi_p', 'cd2', 'stratified'
    Useful when the user wants to evaluate the design under a different
    criterion than the one used for optimisation.

MC baseline (optional, mc_samples > 0)
---------------------------------------
    Draws mc_samples random designs from the feasible space.
    For each metric and criteria_metric, computes:

    - baseline median
    - percentile rank of the design value among random designs

References
----------
Johnson, Moore & Ylvisaker (1990), J. Statist. Plan. Infer. 26(2).
Hickernell (1998), Math. Comp. 67.
Lekivetz & Jones (2015), Qual. Reliab. Eng. Int. 31(5).
Sun, Wang & Xu (2019), Annals of Statistics 47(1).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

_EPS = 1e-10

# ── Terminal colours ──────────────────────────────────────────────────────
_GREEN  = "\033[0;32m"
_YELLOW = "\033[1;33m"
_RED    = "\033[0;31m"
_RESET  = "\033[0m"

# ── Metric names ──────────────────────────────────────────────────────────
_DEFAULT_METRICS = [
    'min_distance',
    'minimax',
    'max_abs_correlation',
    'projection_cd2',
    'cv_distances',
    'mean_distance',
]

_METRIC_LABELS = {
    'min_distance'       : 'Min distance',
    'minimax'            : 'Minimax distance',
    'max_abs_correlation': 'Max |correlation|',
    'projection_cd2'     : '2D projection CD2',
    'cd2'                : 'Centred L2 discrepancy',
    'cv_distances'       : 'CV distances',
    'mean_distance'      : 'Mean distance',
}

# LaTeX math labels for the metrics, for use in rendering targets
# (plots, LaTeX / Markdown / HTML exports). Plain-text targets
# (terminal, CSV, Excel) keep the names above. References for the
# notation: min/mean pairwise distance and maximin d_min (Johnson,
# Moore & Ylvisaker 1990); minimax d_mM (Johnson et al. 1990);
# centered L2-discrepancy CD_2 (Hickernell 1998).
_METRIC_LATEX = {
    'min_distance'       : r'$d_{\min}$',
    'minimax'            : r'$d_{\mathrm{mM}}$',
    'max_abs_correlation': r'$|\rho|_{\max}$',
    'projection_cd2'     : r'$\mathrm{CD}_2^{\mathrm{proj}}$',
    'cd2'                : r'$\mathrm{CD}_2$',
    'cv_distances'       : r'$\mathrm{CV}_d$',
    'mean_distance'      : r'$\bar{d}$',
}


def metric_latex(name: str) -> str:
    """Return the LaTeX math label for a metric, or the name itself."""
    return _METRIC_LATEX.get(name, name)

# Higher is better for these
_HIGHER_BETTER = {'min_distance', 'mean_distance'}

# What each criterion primarily optimises — maps to metric keys
# References:
#   phi_p      : Morris & Mitchell (1995), Johnson et al. (1990)
#   MaxPro     : Joseph, Gul & Ba (2015), Biometrika 102(2)
#   uMaxPro    : Vorechovsky & Masek (2026), arxiv 2603.19778
#   MaxProQQ   : Joseph, Gul & Ba (2020), J. Qual. Technol. 52(4)
#   cd2        : Hickernell (1998)
#   qqd        : Zhang, Yang & Zhou (2021); a discrepancy, like cd2
#   stratified : Tian & Xu (2025), JRSS-B
_CRIT_OPTIMISES = {
    'umaxpro'    : {'projection_cd2', 'minimax', 'max_abs_correlation'},
    'maxpro'     : {'projection_cd2', 'minimax', 'max_abs_correlation'},
    'maxproqq'   : {'projection_cd2', 'minimax', 'max_abs_correlation'},
    'phi_p'      : {'min_distance', 'mean_distance', 'cv_distances'},
    'cd2'        : {'projection_cd2', 'cv_distances', 'minimax'},
    'qqd'        : {'projection_cd2', 'cv_distances', 'minimax'},
    'stratified' : {'projection_cd2', 'cv_distances'},
}


# ======================================================================
# Metric functions — all operate on normalised [0,1]^d coordinates
# ======================================================================

from .distances import heom_pairwise, _embed_heom  # noqa: E402


def _pairwise_distances(
    X:     np.ndarray,
    space=None,
) -> np.ndarray:
    """
    Flat vector of unique upper-triangle pairwise distances.

    Applies HEOM (Wilson & Martinez 1997) when *space* contains any
    nominal factor; otherwise reduces to normalised Euclidean, which
    lets the numerical values reported for all-numerical spaces stay
    unchanged.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2:
        return np.empty(0)
    if space is not None and space.has_nominal:
        D = heom_pairwise(X, space=space, squared=False)
        iu, ju = np.triu_indices(n, k=1)
        return D[iu, ju]
    # Fast path: identical to the pre-categorical implementation.
    parts = []
    for i in range(n - 1):
        parts.append(np.sqrt(np.sum((X[i] - X[i + 1:]) ** 2, axis=1)))
    return np.concatenate(parts) if parts else np.empty(0)


def min_distance(X: np.ndarray, space=None) -> float:
    """
    Minimum normalised pairwise distance (maximin criterion).

    Higher values indicate better point separation. When *space*
    contains nominal factors the pairwise distance is HEOM (Wilson
    & Martinez 1997); otherwise the standard normalised Euclidean
    distance is used.

    References
    ----------
    Johnson, M. E., Moore, L. M. & Ylvisaker, D. (1990).
        *Journal of Statistical Planning and Inference*, 26(2), 131-148.
    Wilson, D. R. & Martinez, T. R. (1997). Improved heterogeneous
        distance functions. *J. Artif. Intell. Res.*, 6, 1-34.
    """
    dists = _pairwise_distances(X, space=space)
    if dists.size == 0:
        return 0.0
    return float(dists.min())


def mean_distance(X: np.ndarray, space=None) -> float:
    """
    Mean normalised pairwise distance.

    Higher values indicate better overall spread. Uses HEOM in the
    presence of nominal factors; normalised Euclidean otherwise.
    """
    dists = _pairwise_distances(X, space=space)
    if dists.size == 0:
        return 0.0
    return float(dists.mean())


def cv_distances(X: np.ndarray, space=None) -> float:
    """
    Coefficient of variation of pairwise distances: std / mean.

    Lower values indicate more uniform spacing. Uses HEOM in the
    presence of nominal factors; normalised Euclidean otherwise.

    References
    ----------
    Lekivetz, R. & Jones, B. (2015). Fast flexible space-filling
        designs for nonrectangular regions. *Quality and Reliability
        Engineering International*, 31(5), 829-837.
    """
    dists = _pairwise_distances(X, space=space)
    if dists.size == 0:
        return 0.0
    mu = dists.mean()
    return float(dists.std() / mu) if mu > _EPS else 0.0


#: Feasible pools up to this size are evaluated exactly by minimax().
_MINIMAX_EXACT_MAX = 50_000


def _minimax_eval_set(space) -> np.ndarray:
    """
    Normalised feasible candidate pool used by minimax(): the whole
    pool when it has at most ``_MINIMAX_EXACT_MAX`` points, otherwise a
    fixed random subset of that size (the same for every design).
    """
    source = space.candidate_pool
    cached = getattr(space, '_minimax_eval', None)
    if cached is not None and cached[0] is source:
        return cached[1]
    pool = np.asarray(source, dtype=float)
    if len(pool) > _MINIMAX_EXACT_MAX:
        idx  = np.random.default_rng(44).choice(
            len(pool), size=_MINIMAX_EXACT_MAX, replace=False)
        pool = pool[np.sort(idx)]
    granges = space.granges
    E = (pool - space.gmins) / granges
    # Keyed on the pool object, which the space rebuilds whenever its
    # parameters or constraints change.
    space._minimax_eval = (source, E)
    return E


def minimax(X: np.ndarray, space=None, n_eval: int = 5_000) -> float:
    """
    Minimax (fill / coverage) distance.

    The largest distance from a point of the space to its nearest
    design point. Lower values indicate better space coverage. Uses
    HEOM when *space* has nominal factors so that coverage is defined
    consistently on mixed spaces; without *space* nominal factors are
    not recognised.

    With *space* given, the points to cover are the feasible candidate
    pool (constraints applied), so the value is exact for the discrete
    problem Mergen solves. Pools larger than ``_MINIMAX_EXACT_MAX``
    points are represented by a fixed random subset; the value is then
    an estimate from below. Without *space*, a regular grid on
    :math:`[0, 1]^d` (``n_eval`` points; random points for d > 6) is
    used, which ignores constraints and nominal factors and is also
    an estimate from below.

    References
    ----------
    Johnson, M. E., Moore, L. M. & Ylvisaker, D. (1990).
        *Journal of Statistical Planning and Inference*, 26(2), 131-148.
    """
    X    = np.asarray(X, dtype=float)
    n, d = X.shape
    chunk = 1_000

    if space is not None:
        from scipy.spatial import cKDTree
        E = _minimax_eval_set(space)
        if space.has_nominal:
            mask   = space.is_mask
            levels = [np.unique(np.concatenate([E[:, j], X[:, j]]))
                      if mask[j] else None for j in range(d)]
            E, X = _embed_heom(E, levels), _embed_heom(X, levels)
        dist, _ = cKDTree(X).query(E, k=1)
        return float(dist.max())

    if d <= 6:
        n_grid   = max(3, int(round(n_eval ** (1 / d))))
        axes     = [np.linspace(0, 1, n_grid)] * d
        eval_pts = np.column_stack(
            [g.ravel() for g in np.meshgrid(*axes, indexing='ij')]
        )
    else:
        eval_pts = np.random.default_rng(44).uniform(0, 1, (n_eval, d))

    max_min = 0.0
    for start in range(0, len(eval_pts), chunk):
        ep = eval_pts[start:start + chunk]
        dists = np.sqrt(
            ((ep[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        max_min = max(max_min, float(dists.min(axis=1).max()))
    return float(max_min)


def max_abs_correlation(X: np.ndarray, space=None) -> float:
    """
    Maximum absolute pairwise correlation between parameter columns.

    Lower values indicate better orthogonality between parameters.
    A value of 0 means parameters are completely uncorrelated.

    Correlation is defined only on ordered columns, so nominal
    columns (when *space* is provided and lists any) are excluded
    from the pairing pool. If fewer than two ordered columns remain
    the metric returns 0.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n < 2:
        return 0.0
    if space is not None and space.has_nominal:
        keep = ~space.is_mask
        X = X[:, keep]
        d = X.shape[1]
    if d < 2:
        return 0.0
    Xc    = X - X.mean(axis=0)
    norms = np.sqrt((Xc ** 2).sum(axis=0))
    norms[norms < _EPS] = 1.0
    Xn   = Xc / norms
    corr = Xn.T @ Xn
    np.fill_diagonal(corr, 0.0)
    return float(np.abs(corr).max())


def projection_cd2(X: np.ndarray, space=None) -> float:
    """
    Mean centred L2 discrepancy across all 2D projections.

    Evaluates uniformity in every pair of parameter dimensions.
    Lower values indicate better 2D projection coverage. Sun, Wang &
    Xu (2019) average the squared discrepancy, CD2²; this metric
    averages CD2 itself, which ranks designs on the same projections
    but is not the same number.

    Nominal columns are excluded from the projection sweep because
    the CD2 kernel of Hickernell (1998) is defined on ordered
    :math:`[0, 1]` axes; averaging it over pairs that involve a
    nominal column would give a value with no interpretation.

    References
    ----------
    Sun, F., Wang, Y. & Xu, H. (2019). Uniform projection designs.
        *Annals of Statistics*, 47(1), 641-661.
    Hickernell, F. J. (1998). *Mathematics of Computation*, 67(221),
        299-322.
    """
    X = np.asarray(X, dtype=float)
    if space is not None and space.has_nominal:
        keep = ~space.is_mask
        X = X[:, keep]
    d = X.shape[1]
    if d < 2:
        return 0.0
    from .criteria import CD2 as _CD2
    _cd2 = _CD2()
    vals = [_cd2.evaluate(X[:, [i, j]], space=None)
            for i, j in combinations(range(d), 2)]
    return float(np.mean(vals))


def cd2(X: np.ndarray, space=None) -> float:
    """
    Centred L2 discrepancy of the whole design.

    The same quantity the ``'cd2'`` criterion minimises, here as a
    quality metric, so it can be named in a ``priority``. Lower values
    indicate a design closer to uniform on the unit cube. Nominal
    columns are left out, as in :func:`projection_cd2`; with no ordered
    column the value is NaN.

    References
    ----------
    Hickernell, F. J. (1998). *Mathematics of Computation*, 67(221),
        299-322.
    """
    X = np.asarray(X, dtype=float)
    if space is not None and space.has_nominal:
        X = X[:, ~space.is_mask]
    if X.shape[1] == 0:
        return float('nan')
    from .criteria import CD2 as _CD2
    return float(_CD2().evaluate(X, space=None))


# ── Metric dispatcher ─────────────────────────────────────────────────────
_METRIC_FN = {
    'min_distance'       : min_distance,
    'minimax'            : minimax,
    'max_abs_correlation': max_abs_correlation,
    'projection_cd2'     : projection_cd2,
    'cd2'                : cd2,
    'cv_distances'       : cv_distances,
    'mean_distance'      : mean_distance,
}


def _compute_metric(name: str, X: np.ndarray, space=None) -> float:
    """Compute a metric by name on normalised design X."""
    if name not in _METRIC_FN:
        raise ValueError(f"Unknown metric '{name}'. "
                         f"Available: {list(_METRIC_FN.keys())}")
    return float(_METRIC_FN[name](X, space=space))


def _compute_criterion(name, X: np.ndarray, space=None) -> float:
    """
    Compute a criterion score on normalised design X. *name* is a
    criterion name or a criterion instance (with its own parameters).
    """
    from .criteria import get_criterion
    crit = get_criterion(name) if isinstance(name, str) else name
    return float(crit.evaluate(X, space=space))


# ======================================================================
# Monte Carlo baseline
# ======================================================================

def _drop_rows(pool: np.ndarray, rows, tol) -> np.ndarray:
    """
    The pool without the rows that match any of *rows* within *tol* (a
    scalar or one value per axis) on every axis, in the original order.
    Each coordinate is mapped to its nearest level of the pool on that
    axis, so the cost is one sort per axis rather than one pass over the
    pool per row. Exact when *tol* is below half the gap between
    neighbouring levels, as ``grid_tol`` is.
    """
    R = np.asarray(rows, dtype=float).reshape(-1, pool.shape[1])
    if not len(R) or not len(pool):
        return pool
    tol = np.broadcast_to(np.asarray(tol, dtype=float), (pool.shape[1],))
    pool_key = np.zeros(len(pool), dtype=np.int64)
    row_key  = np.zeros(len(R), dtype=np.int64)
    matched  = np.ones(len(R), dtype=bool)
    for j in range(pool.shape[1]):
        levels, inv = np.unique(pool[:, j], return_inverse=True)
        pos  = np.clip(np.searchsorted(levels, R[:, j]), 1, len(levels) - 1) \
            if len(levels) > 1 else np.zeros(len(R), dtype=np.int64)
        if len(levels) > 1:
            left = np.abs(R[:, j] - levels[pos - 1]) <= np.abs(
                R[:, j] - levels[pos])
            pos  = np.where(left, pos - 1, pos)
        matched &= np.abs(R[:, j] - levels[pos]) <= tol[j]
        pool_key = pool_key * len(levels) + inv.ravel()
        row_key  = row_key * len(levels) + pos
    return pool[~np.isin(pool_key, row_key[matched])]


def _mc_baseline(
    gs,
    gmins:        np.ndarray,
    granges:      np.ndarray,
    n:            int,
    metric_names: List[str],
    crit_names:   List[str],
    mc_samples:   int,
    seed:         int = 44,
    space         = None,
    exclude:      Optional[Sequence] = None,
    crit_objs:    Optional[Dict] = None,
    fixed:        Optional[Sequence] = None,
) -> Dict[str, np.ndarray]:
    """
    Draw mc_samples random designs of size n from the feasible pool
    and evaluate all requested metrics and criteria.

    With *space* given, the pool is its feasible candidate pool
    (constraints applied) minus the points in *exclude* (exclusion
    centres, add_set() points and focus centres with
    include_center=False, which the optimiser can never choose). Without *space*, the
    full Cartesian product of *gs* is used.

    *fixed* are the delivered rows the optimiser did not choose
    (prescribed and focus points). Each random design is then those k
    rows plus n - k points drawn from the pool without them, so the
    rank measures what the optimiser added. Without fixed rows the
    draws are unchanged.

    Returns
    -------
    dict {name: np.ndarray of shape (mc_samples,)}
    """
    rng     = np.random.default_rng(seed)
    results = {m: [] for m in metric_names + crit_names}

    if space is not None:
        pool = np.asarray(space.candidate_pool, dtype=float)
        if exclude is not None and len(exclude):
            pool = _drop_rows(pool, exclude, space.grid_tol)
    else:
        pool = None
    F = (np.empty((0, len(gmins))) if fixed is None
         else np.asarray(fixed, dtype=float).reshape(-1, len(gmins)))
    if len(F):
        if pool is None:
            pool = np.array([gs.index_to_point(i)
                             for i in range(gs.n_candidates)], dtype=float)
        pool = _drop_rows(pool, F,
                          space.grid_tol if space is not None else 1e-9)
    n_free = max(n - len(F), 0)
    n_pool = len(pool) if pool is not None else gs.n_candidates
    if n_pool == 0 and n_free > 0:
        return {k: np.full(mc_samples, np.nan) for k in results}
    if n_pool < n_free:
        import warnings
        warnings.warn(
            f"Only {n_pool} feasible points are left for the baseline, so "
            f"its random designs have {len(F) + n_pool} points instead "
            f"of {n}.", UserWarning, stacklevel=3)

    for _ in range(mc_samples):
        indices  = rng.choice(n_pool, size=min(n_free, n_pool),
                              replace=False)
        if pool is not None:
            pts_raw = pool[indices]
        else:
            pts_raw = np.array([gs.index_to_point(int(i)) for i in indices])
        if len(F):
            pts_raw = np.vstack([F, pts_raw.reshape(-1, len(gmins))])
        X_norm   = (pts_raw - gmins) / granges

        for m in metric_names:
            try:
                results[m].append(_compute_metric(m, X_norm, space=space))
            except Exception:
                results[m].append(np.nan)

        for c in crit_names:
            try:
                results[c].append(_compute_criterion(
                    (crit_objs or {}).get(c, c), X_norm, space=space))
            except Exception:
                results[c].append(np.nan)

    return {k: np.array(v) for k, v in results.items()}


def _percentile_rank(design_val: float, mc_vals: np.ndarray,
                     higher_better: bool) -> float:
    """
    Percentile rank of ``design_val`` among the ``B`` baseline values
    and the design itself (``B + 1`` values), with midrank tie handling:
    values strictly beaten count fully, ties count half, and the design
    ties with itself. On a discrete grid many random designs share the
    same metric value (e.g. an adjacent-cell minimum distance); midrank
    places a design that ties the whole baseline mid-distribution
    instead of at the 0th percentile. Counting the design keeps the rank
    strictly between 0 and 100, since ``B`` random designs cannot show
    that no design beats it. The rank describes the design; it does not
    take part in choosing one.
    """
    if np.isnan(design_val):
        return np.nan
    mc_clean = mc_vals[~np.isnan(mc_vals)]
    if len(mc_clean) == 0:
        return np.nan
    equal = np.isclose(mc_clean, design_val, rtol=1e-9, atol=1e-12)
    if higher_better:
        beaten = (mc_clean < design_val) & ~equal
    else:
        beaten = (mc_clean > design_val) & ~equal
    B = len(mc_clean)
    return float((beaten.sum() + 0.5 * (equal.sum() + 1)) / (B + 1) * 100)


# ======================================================================
# quality_report
# ======================================================================

def quality_report(
    result,
    metrics:          Union[str, List[str]] = 'default',
    criteria_metrics: Optional[List[str]]   = None,
    mc_samples:       int                   = 0,
    verbose:          bool                  = True,
) -> dict:
    """
    Compute and print design quality metrics.

    Parameters
    ----------
    result           : SamplingResult
    metrics          : 'default' | list of metric names
        Which metrics to compute. Default: all 6 standard metrics.
    criteria_metrics : list of criterion names or None
        Additional criteria to evaluate the design against
        (e.g. ['maxpro', 'phi_p']). Computed from the design directly,
        no SA involved. Useful to see how the design scores under
        criteria other than the one used for optimisation.
    mc_samples       : int
        Number of random designs drawn from the feasible space for
        baseline comparison. 0 = disabled (default).
        When > 0, each metric and criterion is given a percentile rank:
        what fraction of random designs does this design outperform?

    Returns
    -------
    dict with all computed values and (if mc_samples > 0) baseline stats.
    """
    space   = result.space
    names   = space.names
    gmins   = space.gmins
    granges = space.granges
    gs      = space.grid_sampler()

    # Normalise design (all point types included)
    pts    = result.samples[names].values.astype(float)
    n, d   = pts.shape
    X_norm = (pts - gmins) / granges

    # Resolve metrics
    if metrics == 'default':
        metric_names = list(_DEFAULT_METRICS)
    else:
        metric_names = list(metrics)
        unknown = [m for m in metric_names if m not in _METRIC_FN]
        if unknown:
            raise ValueError(
                f"\n{_RED}[MERGEN ERROR]{_RESET}  "
                f"Unknown metric(s): {unknown}.\n"
                f"  Available: {list(_METRIC_FN.keys())}"
            )

    # Criteria: always include the one used in run()
    run_criteria = result._meta.get('criteria', None)
    crit_names   = []
    # A loaded design was not optimised on any criterion.
    if run_criteria and run_criteria != 'loaded':
        if isinstance(run_criteria, str):
            crit_names = [run_criteria]
        else:
            crit_names = list(run_criteria)
    if criteria_metrics:
        for c in criteria_metrics:
            if c not in crit_names:
                crit_names.append(c)
    # A name that is both a metric and a scored criterion ('cd2') is
    # reported once, as the criterion.
    metric_names = [m for m in metric_names if m not in crit_names]

    # ── Compute design values ─────────────────────────────────────────
    design_metrics: Dict[str, float] = {}
    for m in metric_names:
        try:
            design_metrics[m] = _compute_metric(m, X_norm, space=space)
        except Exception:
            design_metrics[m] = np.nan

    # The run's criterion is scored with the instance it optimised, so
    # non-default parameters (e.g. phi_p's p) are respected.
    crit_objs = {}
    if result._meta.get('criterion_obj') is not None and run_criteria:
        crit_objs[crit_names[0]] = result._meta['criterion_obj']

    design_criteria: Dict[str, float] = {}
    for c in crit_names:
        try:
            design_criteria[c] = _compute_criterion(
                crit_objs.get(c, c), X_norm, space=space)
        except Exception:
            design_criteria[c] = np.nan

    # ── MC baseline ───────────────────────────────────────────────────
    # Fixed rows that are part of the delivered design (fill_around
    # drops its reference rows from the samples).
    fixed = np.asarray(result._meta.get('fixed_points') or [],
                       dtype=float).reshape(-1, d)
    if len(fixed):
        tol   = getattr(space, 'grid_tol', 1e-9)
        fixed = fixed[[bool(np.any(np.all(np.abs(pts - f) <= tol, axis=1)))
                       for f in fixed]]
    n_fixed = len(fixed)
    mc_vals: Dict[str, np.ndarray] = {}
    if mc_samples > 0:
        if verbose:
            print(f"  {_YELLOW}[METRICS]{_RESET}  "
                  f"Computing MC baseline ({mc_samples} designs)...")
        # Same baseline seed as compare() for the run's seed.
        seed = result._meta.get('seed')
        mc_vals = _mc_baseline(
            gs, gmins, granges, n,
            metric_names, crit_names,
            mc_samples, seed=44 if seed is None else seed, space=space,
            exclude=(list(result._meta.get('exclusions') or [])
                     + list(result._meta.get('baseline_reserved') or [])),
            crit_objs=crit_objs,
            fixed=fixed,
        )
        if verbose:
            print(f"  {_GREEN}[METRICS]{_RESET}  "
                  f"MC baseline complete ({mc_samples} designs).")

    # ── Print table ───────────────────────────────────────────────────
    if verbose:
        _print_table(
            n, d, result._meta.get('criteria', '?'),
            metric_names, design_metrics,
            crit_names, design_criteria,
            mc_vals, n_fixed,
        )

    # ── Return dict ───────────────────────────────────────────────────
    out = {
        'n_design'  : n,
        'n_params'  : d,
        'criteria'  : result._meta.get('criteria', '?'),
        'mc_samples': mc_samples,
        'baseline_fixed_points': n_fixed,
    }
    for m in metric_names:
        out[m] = design_metrics[m]
        if m in mc_vals:
            hb = m in _HIGHER_BETTER
            out[f'{m}_baseline_median'] = float(np.nanmedian(mc_vals[m]))
            out[f'{m}_percentile_rank'] = _percentile_rank(
                design_metrics[m], mc_vals[m], hb)

    for c in crit_names:
        out[f'criterion_{c}'] = design_criteria[c]
        if c in mc_vals:
            # Criteria: lower is better
            out[f'criterion_{c}_baseline_median'] = float(np.nanmedian(mc_vals[c]))
            out[f'criterion_{c}_percentile_rank'] = _percentile_rank(
                design_criteria[c], mc_vals[c], higher_better=False)

    return out


# ── Formatting helpers ────────────────────────────────────────────────────

def _fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '     n/a'
    if abs(v) < 0.01 or abs(v) > 9999:
        return f'{v:.4e}'
    return f'{v:.4f}'


def _ordinal(p) -> str:
    """A percentile as an English ordinal: 1st, 2nd, 3rd, 11th, 72nd."""
    k = int(round(float(p)))
    if 10 <= k % 100 <= 20:
        return f'{k}th'
    return f"{k}{({1: 'st', 2: 'nd', 3: 'rd'}).get(k % 10, 'th')}"


def _pct_ordinal(p) -> str:
    """A percentile rank as an ordinal, within 1st..99th.

    The rank never reaches 0 or 100 (see :func:`_percentile_rank`), so
    rounding must not print it as either.
    """
    return _ordinal(min(max(int(round(float(p))), 1), 99))


def _fmt_pct(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '      n/a'
    colour = _GREEN if p >= 75 else (_YELLOW if p >= 50 else _RED)
    text = f'{_pct_ordinal(p)} pct'
    return f'{colour}{text}{_RESET}'


def _print_table(
    n, d, run_crit,
    metric_names, design_metrics,
    crit_names, design_criteria,
    mc_vals, n_fixed=0,
):
    W   = 72
    has_mc = len(mc_vals) > 0
    sep = '─' * W

    print()
    print('═' * W)
    print(f'  MERGEN Design Metrics  (n={n}, d={d})')
    print('═' * W)

    if has_mc:
        hdr = f"  {'Metric':<22} {'Value':>10}  {'Baseline':>10}  {'Better when':<12}  {'Rank':>9}"
    else:
        hdr = f"  {'Metric':<22} {'Value':>10}  {'Better when':<12}"
    print(hdr)
    print(sep)

    # Standard metrics
    opt_metrics = _CRIT_OPTIMISES.get(run_crit, set())
    for m in metric_names:
        label  = _METRIC_LABELS.get(m, m)
        val    = design_metrics.get(m, np.nan)
        direc  = 'higher' if m in _HIGHER_BETTER else 'lower'
        mark   = '*' if m in opt_metrics else ' '
        if has_mc and m in mc_vals:
            med = float(np.nanmedian(mc_vals[m]))
            pct = _percentile_rank(val, mc_vals[m], m in _HIGHER_BETTER)
            pct_str = f'{_pct_ordinal(pct)} pct' if pct is not None else 'n/a'
            colour  = _GREEN if (pct or 0) >= 75 else (_YELLOW if (pct or 0) >= 50 else _RED)
            print(f"  {label:<22} {_fmt_val(val):>10}  "
                  f"{_fmt_val(med):>10}  {direc:<12}  "
                  f"{colour}{pct_str:>9}{_RESET}  {mark}")
        else:
            print(f"  {label:<22} {_fmt_val(val):>10}  {direc:<12}  {mark}")

    # Criteria scores
    if crit_names:
        print(sep)
        print(f"  {'Criterion scores':<26}")
        print(sep)
        for c in crit_names:
            label  = c.upper() if c == run_crit else f'{c.upper()} *'
            val    = design_criteria.get(c, np.nan)
            if has_mc and c in mc_vals:
                med = float(np.nanmedian(mc_vals[c]))
                pct = _percentile_rank(val, mc_vals[c], higher_better=False)
                print(f"  {label:<26} {_fmt_val(val):>10}  "
                      f"{_fmt_val(med):>10}  {_fmt_pct(pct):>10}  lower")
            else:
                print(f"  {label:<26} {_fmt_val(val):>10}  lower")

        if any(c != run_crit for c in crit_names):
            print("  * evaluated post-hoc (not used for optimisation)")

    # Criterion note
    if run_crit and run_crit in _CRIT_OPTIMISES:
        print(sep)
        print(f"  * = primarily optimised by '{run_crit}'")
        print("  For other priorities, see: mergen.criteria.list_criteria()")

    if has_mc:
        print(sep)
        n_mc = len(mc_vals[next(iter(mc_vals))])
        if n_fixed:
            print(f"  Baseline: {n_mc} random completions of the {n_fixed} "
                  f"fixed points  |  Rank = percentile among them")
        else:
            print(f"  Baseline: {n_mc} MC designs from feasible space  |  Rank = percentile among baseline designs")
    print('═' * W)
    print()


# ======================================================================
# comparison — multiple criteria
# ======================================================================

def comparison(result) -> pd.DataFrame:
    """
    Compare design quality metrics across all criteria in result.designs.

    Each row is a criterion, each column is a metric.
    Only the standard metrics are included (no MC).

    Parameters
    ----------
    result : SamplingResult

    Returns
    -------
    pd.DataFrame — rows = criteria, columns = metrics
    """
    space   = result.space
    names   = space.names
    gmins   = space.gmins
    granges = space.granges

    rows = {}
    for crit_name, df in result.designs.items():
        pts    = df[names].values.astype(float)
        X_norm = (pts - gmins) / granges
        row    = {}
        for m in _DEFAULT_METRICS:
            try:
                row[m] = _compute_metric(m, X_norm, space=space)
            except Exception:
                row[m] = np.nan
        # Also include the criterion score
        try:
            row[f'{crit_name}_score'] = _compute_criterion(crit_name, X_norm,
                                                           space=space)
        except Exception:
            row[f'{crit_name}_score'] = np.nan
        rows[crit_name] = row

    df_out = pd.DataFrame(rows).T
    df_out.index.name = 'criterion'

    # Print
    W = 72
    print()
    print('═' * W)
    print('  MERGEN Criteria Comparison')
    print('═' * W)
    print(df_out.to_string(float_format=lambda x: f'{x:.4f}'))
    print('═' * W)
    print()

    return df_out
