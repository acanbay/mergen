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
    projection_cd2       Liu & Liu (2023) — 2D projection uniformity
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
Lekivetz & Jones (2015), Quality Engineering 27(1).
Liu & Liu (2023), Canadian Journal of Statistics 51(1).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Union

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
    'cv_distances'       : 'CV distances',
    'mean_distance'      : 'Mean distance',
}

# Higher is better for these
_HIGHER_BETTER = {'min_distance', 'mean_distance'}

# What each criterion primarily optimises — maps to metric keys
# References:
#   phi_p      : Morris & Mitchell (1995), Johnson et al. (1990)
#   MaxPro     : Joseph, Gul & Ba (2015), Biometrika 102(2)
#   uMaxPro    : Vorechovsky & Elias (2026), arxiv 2603.19778
#   cd2        : Hickernell (1998)
#   stratified : Tian & Xu (2025), JRSS-B
_CRIT_OPTIMISES = {
    'umaxpro'    : {'projection_cd2', 'minimax', 'max_abs_correlation'},
    'maxpro'     : {'projection_cd2', 'minimax', 'max_abs_correlation'},
    'phi_p'      : {'min_distance', 'mean_distance', 'cv_distances'},
    'cd2'        : {'projection_cd2', 'cv_distances', 'minimax'},
    'stratified' : {'projection_cd2', 'cv_distances'},
}


# ======================================================================
# Metric functions — all operate on normalised [0,1]^d coordinates
# ======================================================================

def min_distance(X: np.ndarray) -> float:
    """
    Minimum normalised pairwise Euclidean distance (maximin criterion).

    Higher values indicate better point separation.

    References
    ----------
    Johnson, M. E., Moore, L. M. & Ylvisaker, D. (1990).
        *Journal of Statistical Planning and Inference*, 26(2), 131-148.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2:
        return 0.0
    min_d = np.inf
    for i in range(n - 1):
        dists = np.sqrt(np.sum((X[i] - X[i + 1:]) ** 2, axis=1))
        min_d = min(min_d, float(dists.min()))
    return float(min_d)


def mean_distance(X: np.ndarray) -> float:
    """
    Mean normalised pairwise Euclidean distance.
    Higher values indicate better overall spread.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n - 1):
        dists.extend(np.sqrt(np.sum((X[i] - X[i + 1:]) ** 2, axis=1)).tolist())
    return float(np.mean(dists))


def cv_distances(X: np.ndarray) -> float:
    """
    Coefficient of variation of pairwise distances: std / mean.

    Lower values indicate more uniform spacing.

    References
    ----------
    Lekivetz, R. & Jones, B. (2015).
        *Quality Engineering*, 27(1), 46-52.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n - 1):
        dists.extend(np.sqrt(np.sum((X[i] - X[i + 1:]) ** 2, axis=1)).tolist())
    dists = np.array(dists)
    mu = dists.mean()
    return float(dists.std() / mu) if mu > _EPS else 0.0


def minimax(X: np.ndarray, n_eval: int = 5_000) -> float:
    """
    Minimax (fill / coverage) distance.

    The maximum over all evaluation points of the minimum distance to
    any design point. Lower values indicate better space coverage.

    References
    ----------
    Johnson, M. E., Moore, L. M. & Ylvisaker, D. (1990).
        *Journal of Statistical Planning and Inference*, 26(2), 131-148.
    """
    X    = np.asarray(X, dtype=float)
    n, d = X.shape
    chunk = 1_000

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
        ep      = eval_pts[start:start + chunk]
        dists   = np.sqrt(
            ((ep[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        max_min = max(max_min, float(dists.min(axis=1).max()))
    return float(max_min)


def max_abs_correlation(X: np.ndarray) -> float:
    """
    Maximum absolute pairwise correlation between parameter columns.

    Lower values indicate better orthogonality between parameters.
    A value of 0 means parameters are completely uncorrelated.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if d < 2 or n < 2:
        return 0.0
    Xc    = X - X.mean(axis=0)
    norms = np.sqrt((Xc ** 2).sum(axis=0))
    norms[norms < _EPS] = 1.0
    Xn   = Xc / norms
    corr = Xn.T @ Xn
    np.fill_diagonal(corr, 0.0)
    return float(np.abs(corr).max())


def projection_cd2(X: np.ndarray) -> float:
    """
    Mean centred L2 discrepancy across all 2D projections.

    Evaluates uniformity in every pair of parameter dimensions.
    Lower values indicate better 2D projection coverage.

    References
    ----------
    Liu, H. & Liu, M.-Q. (2023).
        *Canadian Journal of Statistics*, 51(1), 293-311.
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    if d < 2:
        return 0.0
    from .criteria import CD2 as _CD2
    _cd2 = _CD2()
    vals = [_cd2.evaluate(X[:, [i, j]], space=None)
            for i, j in combinations(range(d), 2)]
    return float(np.mean(vals))


# ── Metric dispatcher ─────────────────────────────────────────────────────
_METRIC_FN = {
    'min_distance'       : min_distance,
    'minimax'            : minimax,
    'max_abs_correlation': max_abs_correlation,
    'projection_cd2'     : projection_cd2,
    'cv_distances'       : cv_distances,
    'mean_distance'      : mean_distance,
}


def _compute_metric(name: str, X: np.ndarray) -> float:
    """Compute a metric by name on normalised design X."""
    if name not in _METRIC_FN:
        raise ValueError(f"Unknown metric '{name}'. "
                         f"Available: {list(_METRIC_FN.keys())}")
    return float(_METRIC_FN[name](X))


def _compute_criterion(name: str, X: np.ndarray) -> float:
    """Compute a criterion score by name on normalised design X."""
    from .criteria import get_criterion
    return float(get_criterion(name).evaluate(X, space=None))


# ======================================================================
# Monte Carlo baseline
# ======================================================================

def _mc_baseline(
    gs,
    gmins:        np.ndarray,
    granges:      np.ndarray,
    n:            int,
    metric_names: List[str],
    crit_names:   List[str],
    mc_samples:   int,
    seed:         int = 44,
) -> Dict[str, np.ndarray]:
    """
    Draw mc_samples random designs of size n from the feasible pool
    and evaluate all requested metrics and criteria.

    The feasible pool is the full candidate set (constraints already
    applied via GridSampler). Prescribed and focus reserved indices
    are NOT excluded — they are part of the feasible space.

    Returns
    -------
    dict {name: np.ndarray of shape (mc_samples,)}
    """
    rng     = np.random.default_rng(seed)
    results = {m: [] for m in metric_names + crit_names}

    for _ in range(mc_samples):
        indices  = rng.choice(gs.n_candidates,
                              size=min(n, gs.n_candidates),
                              replace=False)
        pts_raw  = np.array([gs.index_to_point(int(i)) for i in indices])
        X_norm   = (pts_raw - gmins) / np.where(granges > _EPS, granges, 1.0)

        for m in metric_names:
            try:
                results[m].append(_compute_metric(m, X_norm))
            except Exception:
                results[m].append(np.nan)

        for c in crit_names:
            try:
                results[c].append(_compute_criterion(c, X_norm))
            except Exception:
                results[c].append(np.nan)

    return {k: np.array(v) for k, v in results.items()}


def _percentile_rank(design_val: float, mc_vals: np.ndarray,
                     higher_better: bool) -> float:
    """
    Fraction of mc_vals that the design_val outperforms, as a percentage.

    higher_better=True  → pct of mc_vals < design_val
    higher_better=False → pct of mc_vals > design_val
    """
    mc_clean = mc_vals[~np.isnan(mc_vals)]
    if len(mc_clean) == 0:
        return np.nan
    if higher_better:
        return float(np.mean(mc_clean < design_val) * 100)
    else:
        return float(np.mean(mc_clean > design_val) * 100)


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
    X_norm = (pts - gmins) / np.where(granges > _EPS, granges, 1.0)

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
    if run_criteria:
        if isinstance(run_criteria, str):
            crit_names = [run_criteria]
        else:
            crit_names = list(run_criteria)
    if criteria_metrics:
        for c in criteria_metrics:
            if c not in crit_names:
                crit_names.append(c)

    # ── Compute design values ─────────────────────────────────────────
    design_metrics: Dict[str, float] = {}
    for m in metric_names:
        try:
            design_metrics[m] = _compute_metric(m, X_norm)
        except Exception:
            design_metrics[m] = np.nan

    design_criteria: Dict[str, float] = {}
    for c in crit_names:
        try:
            design_criteria[c] = _compute_criterion(c, X_norm)
        except Exception:
            design_criteria[c] = np.nan

    # ── MC baseline ───────────────────────────────────────────────────
    mc_vals: Dict[str, np.ndarray] = {}
    if mc_samples > 0:
        if verbose:
            print(f"  {_YELLOW}[METRICS]{_RESET}  "
                  f"Computing MC baseline ({mc_samples} designs)...")
        mc_vals = _mc_baseline(
            gs, gmins, granges, n,
            metric_names, crit_names,
            mc_samples
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
            mc_vals
        )

    # ── Return dict ───────────────────────────────────────────────────
    out = {
        'n_design'  : n,
        'n_params'  : d,
        'criteria'  : result._meta.get('criteria', '?'),
        'mc_samples': mc_samples,
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


def _fmt_pct(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '      n/a'
    colour = _GREEN if p >= 75 else (_YELLOW if p >= 50 else _RED)
    text = f'{p:.0f}th pct'
    return f'{colour}{text}{_RESET}'


def _print_table(
    n, d, run_crit,
    metric_names, design_metrics,
    crit_names, design_criteria,
    mc_vals,
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
            pct_str = f'{pct:.0f}th pct' if pct is not None else 'n/a'
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
            print(f"  * evaluated post-hoc (not used for optimisation)")

    # Criterion note
    if run_crit and run_crit in _CRIT_OPTIMISES:
        print(sep)
        print(f"  * = primarily optimised by '{run_crit}'")
        print(f"  For other priorities, see: mergen.criteria.list_criteria()")

    if has_mc:
        print(sep)
        print(f"  Baseline: {len(mc_vals[next(iter(mc_vals))])} MC designs from feasible space  |  Rank = percentile among baseline designs")
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
        X_norm = (pts - gmins) / np.where(granges > _EPS, granges, 1.0)
        row    = {}
        for m in _DEFAULT_METRICS:
            try:
                row[m] = _compute_metric(m, X_norm)
            except Exception:
                row[m] = np.nan
        # Also include the criterion score
        try:
            row[f'{crit_name}_score'] = _compute_criterion(crit_name, X_norm)
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