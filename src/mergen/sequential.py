"""
mergen.sequential
=================
Convenience tools for sequential and augmented space-filling design.

All functions are wrappers around the core Sampler API.  They do not
introduce new optimisation logic — they simplify common workflows.

Functions
---------
augment(result, n_add, ...)
    Add points to an existing design, filling the gaps left by the
    current design.  Existing points are frozen (prescribed, in_sa=True).

complement(space, existing_points, n_samples, ...)
    Generate a design that fills the gaps around a given set of points
    that are NOT part of the new design.

from_dataframe(space, df, n_add, ...)
    Load an existing design from a DataFrame or CSV file, then augment it.

subsample(data, space, n, ...)
    Select a space-filling subset of size n from a large dataset using
    Kennard-Stone farthest-point selection.

n_samples_recommendation(space, budget, verbose)
    Recommend the number of design points based on the Loeppky et al.
    (2009) 10p rule and the feasible space size.

References
----------
Loeppky, Sacks & Welch (2009), Technometrics 51(4).    [10p rule]
Kennard & Stone (1969), Technometrics 11(1).            [farthest-point]
Joseph, Gul & Ba (2015), Biometrika 102(2).             [MaxPro]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .space   import ParameterSpace
from .sampler import Sampler, SamplingResult

# Terminal colours
_YELLOW = "\033[1;33m"
_RESET  = "\033[0m"

def _warn(msg: str) -> None:
    print(f"  {_YELLOW}[WARNING]{_RESET}  {msg}")


# ======================================================================
# augment
# ======================================================================

def augment(
    result:       SamplingResult,
    n_add:        int,
    focus:        Optional[List] = None,
    exclusion:    Optional[List] = None,
    criteria:     str            = 'umaxpro',
    n_restarts:   int            = 5,
    seed:         int            = 44,
    **sampler_kwargs,
) -> SamplingResult:
    """
    Add *n_add* points to an existing design.

    Existing design points are frozen (prescribed with ``in_sa=True``),
    so new points are placed to fill the gaps left by the current design.

    Parameters
    ----------
    result : SamplingResult
        The existing design to augment.
    n_add : int
        Number of new points to add.
    focus : list or None
        Optional focus region: ``[point, spread]`` or
        ``[point, spread, n_samples]``.
    exclusion : list or None
        Optional exclusion region: ``[point, spread]``.
    criteria : str
        SA optimisation criterion. Default: ``'umaxpro'``.
    n_restarts : int
        Number of SA restarts. Default: 5.
    seed : int
        Random seed. Default: 44.

    Returns
    -------
    SamplingResult
        Combined result: existing points (Prescribed) + new points (Optimised).

    Examples
    --------
    >>> result2 = augment(result, n_add=10)
    >>> result2 = augment(result, n_add=10, focus=[[350, 4.5], 1.5])

    References
    ----------
    Loeppky, J. L., Sacks, J. & Welch, W. J. (2009).
        *Technometrics*, 51(4), 366-376.
    """
    space = result.space
    names = space.names
    existing_pts = result.samples[names].values

    sampler = Sampler(space)
    sampler.add_prescribed(existing_pts, in_design=False, in_sa=True)

    if focus is not None:
        point  = focus[0]
        spread = focus[1] if len(focus) > 1 else 1.0
        n_samp = focus[2] if len(focus) > 2 else None
        sampler.add_focus(point, spread=spread, n_samples=n_samp,
                          in_design=True, in_sa=True)

    if exclusion is not None:
        point  = exclusion[0]
        spread = exclusion[1] if len(exclusion) > 1 else 1.0
        sampler.add_exclusion(point, spread=spread)

    sampler.set_design(n_samples=n_add)
    sampler.set_sa(n_restarts=n_restarts, **sampler_kwargs)
    new_result = sampler.run(criteria=criteria, seed=seed)

    # Combine existing + new
    existing_df = result.samples.copy()
    existing_df['point_type'] = 'Prescribed'
    new_df = new_result.samples[
        new_result.samples["point_type"].isin(["Optimised", "Focus"])
    ].copy()
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.index.name = 'id'

    out = SamplingResult(combined, new_result.validation, space)
    out.sets  = new_result.sets
    out._meta = new_result._meta
    out._meta['augmented_from'] = len(existing_pts)
    out._meta['n_added']        = n_add
    return out


# ======================================================================
# complement
# ======================================================================

def complement(
    space:           ParameterSpace,
    existing_points: Union[np.ndarray, pd.DataFrame, List],
    n_samples:       int,
    focus:           Optional[List] = None,
    exclusion:       Optional[List] = None,
    criteria:        str            = 'umaxpro',
    n_restarts:      int            = 5,
    seed:            int            = 44,
    **sampler_kwargs,
) -> SamplingResult:
    """
    Generate a design that complements an existing set of points.

    Existing points are reserved and visible to SA (new points avoid
    their vicinity), but they are NOT included in the output design.

    Useful when another group has already sampled certain regions and
    you want to cover the rest of the parameter space.

    Parameters
    ----------
    space : ParameterSpace
    existing_points : array-like or DataFrame, shape (n, d)
        Points to complement. If DataFrame, must contain columns
        matching ``space.names``.
    n_samples : int
        Number of new design points to generate.
    focus : list or None
        Optional focus region: ``[point, spread]``.
    exclusion : list or None
        Optional exclusion region: ``[point, spread]``.
    criteria : str
        Default: ``'umaxpro'``.
    n_restarts : int
        Default: 5.
    seed : int
        Default: 44.

    Returns
    -------
    SamplingResult

    Examples
    --------
    >>> competitor = np.array([[100, 0.5], [400, 5.0], [200, 2.5]])
    >>> result = complement(space, competitor, n_samples=20)

    >>> df = pd.read_csv('prior_experiment.csv')
    >>> result = complement(space, df, n_samples=15)
    """
    names = space.names

    if isinstance(existing_points, pd.DataFrame):
        pts = existing_points[names].values.astype(float)
    else:
        pts = np.asarray(existing_points, dtype=float)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]

    if pts.shape[1] != space.n_parameters:
        raise ValueError(
            f"existing_points has {pts.shape[1]} columns but space has "
            f"{space.n_parameters} parameters."
        )

    sampler = Sampler(space)
    sampler.add_prescribed(pts, in_design=False, in_sa=True)

    if focus is not None:
        point  = focus[0]
        spread = focus[1] if len(focus) > 1 else 1.0
        n_samp = focus[2] if len(focus) > 2 else None
        sampler.add_focus(point, spread=spread, n_samples=n_samp,
                          in_design=True, in_sa=True)

    if exclusion is not None:
        point  = exclusion[0]
        spread = exclusion[1] if len(exclusion) > 1 else 1.0
        sampler.add_exclusion(point, spread=spread)

    sampler.set_design(n_samples=n_samples)
    sampler.set_sa(n_restarts=n_restarts, **sampler_kwargs)
    result = sampler.run(criteria=criteria, seed=seed)

    # Remove the prescribed (existing) points from the output —
    # they are not part of the complement design
    mask = result.samples["point_type"].isin(["Optimised", "Focus"])
    result.samples = result.samples[mask].reset_index(drop=True)
    result.samples.index.name = "id"
    return result


# ======================================================================
# from_dataframe
# ======================================================================

def from_dataframe(
    space:      ParameterSpace,
    data:       Union[pd.DataFrame, str],
    n_add:      int,
    col_map:    Optional[Dict[str, str]] = None,
    criteria:   str                      = 'umaxpro',
    n_restarts: int                      = 5,
    seed:       int                      = 44,
    **sampler_kwargs,
) -> SamplingResult:
    """
    Load an existing design from a DataFrame or CSV, then augment it.

    Parameters
    ----------
    space : ParameterSpace
    data : DataFrame or str
        Existing design. If str, treated as a CSV file path.
    n_add : int
        Number of new points to add.
    col_map : dict or None
        Column name mapping ``{csv_column: space_parameter}``.
        Example: ``{'Temp': 'temperature', 'P': 'pressure'}``
    criteria : str
        Default: ``'umaxpro'``.
    n_restarts : int
        Default: 5.
    seed : int
        Default: 44.

    Returns
    -------
    SamplingResult

    Examples
    --------
    >>> result = from_dataframe(space, 'prior_design.csv', n_add=10)
    >>> result = from_dataframe(space, df, n_add=5,
    ...                         col_map={'Temp': 'temperature'})
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    if col_map:
        df = df.rename(columns=col_map)

    names   = space.names
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not found in data.\n"
            f"  Available: {list(df.columns)}\n"
            f"  Use col_map to rename columns."
        )

    pts = df[names].values.astype(float)

    valid_pts, n_skipped = [], 0
    for i, pt in enumerate(pts):
        if space.on_grid(pt) >= 0:
            valid_pts.append(pt)
        else:
            n_skipped += 1

    if n_skipped:
        _warn(f"{n_skipped} row(s) are off-grid and were skipped.")

    if not valid_pts:
        raise ValueError(
            "No valid points found after grid validation.\n"
            "Check that data values match the parameter grid."
        )

    existing_pts = np.array(valid_pts)

    existing_df = pd.DataFrame(existing_pts, columns=names)
    existing_df['point_type'] = 'Prescribed'
    existing_df.index.name = 'id'

    tmp_result = SamplingResult(
        existing_df,
        pd.DataFrame(columns=names + ['point_type']),
        space,
    )

    return augment(
        tmp_result, n_add=n_add,
        criteria=criteria, n_restarts=n_restarts, seed=seed,
        **sampler_kwargs,
    )


# ======================================================================
# subsample
# ======================================================================

def subsample(
    data:  Union[np.ndarray, pd.DataFrame],
    space: ParameterSpace,
    n:     int,
    seed:  int = 44,
) -> pd.DataFrame:
    """
    Select a space-filling subset of size *n* from a large dataset.

    Uses Kennard-Stone farthest-point selection to maximise the minimum
    distance between selected points.

    Useful for:
    - Reducing large simulation outputs while preserving coverage
    - Selecting representative training subsets from big datasets
    - Subsampling observational data for surrogate modelling

    Parameters
    ----------
    data : array-like or DataFrame, shape (N, d)
        Dataset to subsample. If DataFrame, must contain columns
        matching ``space.names``.
    space : ParameterSpace
        Used for normalisation.
    n : int
        Number of points to select.
    seed : int
        Random seed for the initial anchor point. Default: 44.

    Returns
    -------
    pd.DataFrame, shape (n, n_parameters)
        Selected subset with original index preserved.

    Examples
    --------
    >>> subset = subsample(large_df, space, n=50)
    >>> subset = subsample(simulation_results, space, n=100)

    References
    ----------
    Kennard, R. W. & Stone, L. A. (1969).
        *Technometrics*, 11(1), 137-148.
    """
    names   = space.names
    gmins   = space.gmins
    granges = space.granges

    if isinstance(data, pd.DataFrame):
        pts      = data[names].values.astype(float)
        orig_idx = np.array(data.index)
    else:
        pts      = np.asarray(data, dtype=float)
        orig_idx = np.arange(len(pts))

    N = len(pts)
    if n >= N:
        _warn(
            f"subsample: n={n} >= dataset size N={N}. "
            f"Returning full dataset."
        )
        if isinstance(data, pd.DataFrame):
            return data[names].copy()
        return pd.DataFrame(pts, columns=names)

    norm = (pts - gmins) / np.where(granges > 1e-10, granges, 1.0)

    rng   = np.random.default_rng(seed)
    start = int(rng.integers(0, N))

    selected = [start]

    for _ in range(n - 1):
        remaining = [i for i in range(N) if i not in selected]
        sel_norm  = norm[selected]
        dists     = np.array([
            np.min(np.linalg.norm(sel_norm - norm[i], axis=1))
            for i in remaining
        ])
        best = remaining[int(np.argmax(dists))]
        selected.append(best)

    result_df = pd.DataFrame(pts[selected], columns=names,
                              index=orig_idx[selected])
    result_df.index.name = 'id'
    return result_df


# ======================================================================
# n_samples_recommendation
# ======================================================================

def n_samples_recommendation(
    space:   ParameterSpace,
    budget:  Optional[int] = None,
    verbose: bool          = True,
) -> dict:
    """
    Recommend the number of design points for a given parameter space.

    Combines the Loeppky et al. (2009) 10p rule with feasible space
    size to give a practical recommendation.

    Rules applied
    -------------
    1. **10p rule**: ``n_min = 10 x n_parameters``
    2. **Coverage**: ``n_coverage = ceil(n_candidates ^ (1/d))``
    3. **Budget cap**: capped at *budget* if provided.

    Parameters
    ----------
    space : ParameterSpace
    budget : int or None
        Maximum number of experiments.
    verbose : bool
        Print the recommendation table. Default: True.

    Returns
    -------
    dict
        n_min, n_coverage, n_recommended, budget, warning.

    Examples
    --------
    >>> rec = n_samples_recommendation(space)
    >>> rec = n_samples_recommendation(space, budget=50)

    References
    ----------
    Loeppky, J. L., Sacks, J. & Welch, W. J. (2009).
        *Technometrics*, 51(4), 366-376.
    """
    d          = space.n_parameters
    n_cand     = space.n_candidates
    n_min      = 10 * d
    n_coverage = math.ceil(n_cand ** (1.0 / d)) if n_cand > 0 else n_min
    n_rec      = max(n_min, min(n_coverage, n_min * 3))

    warning = None
    if budget is not None:
        if budget < n_min:
            warning = (
                f"Budget ({budget}) is below the 10p minimum ({n_min}). "
                f"Design quality may be limited."
            )
        n_rec = min(n_rec, budget)

    result = {
        'n_min'        : n_min,
        'n_coverage'   : n_coverage,
        'n_recommended': n_rec,
        'budget'       : budget,
        'warning'      : warning,
    }

    if verbose:
        W = 58
        print()
        print('=' * W)
        print('  MERGEN — Sample Size Recommendation')
        print('=' * W)
        print(f'  Parameters         : {d}')
        print(f'  Feasible candidates: {n_cand:,}')
        print('-' * W)
        print(f'  10p rule (min)     : {n_min}')
        print(f'  Coverage estimate  : {n_coverage}')
        if budget is not None:
            print(f'  Budget             : {budget}')
        print('-' * W)
        print(f'  Recommended        : {n_rec}')
        if warning:
            print(f'  {_YELLOW}Warning{_RESET}: {warning}')
        print('=' * W)
        print()

    return result