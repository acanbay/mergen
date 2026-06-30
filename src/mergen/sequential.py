"""
Sequential and augmented design utilities for Mergen.

This module collects design-time operations that work on **existing**
designs: extending them with new points, filling in around reference
locations, sub-selecting space-filling subsets from a pool, and
assigning execution orders.

All functions in this module are *surrogate-free* — they look only at
parameter-space geometry, never at simulation outputs. Adaptive
sampling, active learning, expected improvement, and batch Bayesian
optimisation are intentionally out of scope; they belong in a
surrogate-modelling package.

Public API
----------
extend(sampler, existing_design, n_new, ...)
    Append n_new space-filling points to an existing design while
    preserving every input row (Wang 2003; Qian 2009).

fill_around(sampler, reference_points, n_new, ...)
    Generate n_new points that space-fill *around* a set of reference
    points; the reference points are criterion-visible but excluded
    from the output.

subsample(sampler, pool, n_select, anchor='center')
    Pick n_select space-filling points from an arbitrary candidate
    pool via Kennard-Stone selection in normalised coordinates
    (Kennard & Stone 1969).

run_order(sampler, design, anchor='center', column='run_order')
    Assign an execution rank so that every cumulative prefix of the
    design remains space-filling — useful for serial execution,
    early stopping, interim analyses.

Notes
-----
The first argument of every public function is a ``Sampler``. The
sampler supplies the parameter-space metadata (axis ranges,
parameter names, constraints) but its existing state — prescribed
points, focus regions, exclusions, ``n_samples``, ``n_validation``
— is preserved across the call via the private snapshot/restore
hooks ``_snapshot_state`` / ``_restore_state``.

References
----------
Kennard, R. W. & Stone, L. A. (1969). Computer aided design of
    experiments. *Technometrics*, 11(1), 137-148.
Wang, G. G. (2003). Adaptive response surface method using
    inherited Latin hypercube design points. *Journal of Mechanical
    Design*, 125(2), 210-220.
Qian, P. Z. G. (2009). Nested Latin hypercube designs.
    *Biometrika*, 96(4), 957-970.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from .sampler import _fatal

if TYPE_CHECKING:
    from .sampler import Sampler, SamplingResult


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────
def _coerce_design(
    sampler: "Sampler",
    design:  Union[np.ndarray, pd.DataFrame, list, tuple],
) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """
    Convert *design* to an ``(n, d)`` float array on the registered
    parameters.

    Returns
    -------
    arr : np.ndarray, shape (n, d)
    df  : pd.DataFrame or None
        The original DataFrame when one was passed in (so the caller
        can preserve its index and extra columns); ``None`` otherwise.

    Raises
    ------
    Exits the process via ``_fatal`` if columns are missing, sizes
    mismatch, or types are incompatible.
    """
    space = sampler.space
    if isinstance(design, pd.DataFrame):
        missing = [c for c in space.names if c not in design.columns]
        if missing:
            _fatal(
                f"DataFrame is missing parameter column(s): {missing}.\n"
                f"  Expected columns: {space.names}"
            )
        arr = design[space.names].to_numpy(dtype=float, copy=True)
        return arr, design
    if isinstance(design, (list, tuple)):
        design = np.asarray(design, dtype=float)
    arr = np.asarray(design, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != space.n_parameters:
        _fatal(
            f"design must have shape (n, {space.n_parameters}); "
            f"got {arr.shape}."
        )
    return arr, None


def _kennard_stone_array(
    sampler:  "Sampler",
    pool:     np.ndarray,
    n_select: int,
    anchor:   Union[str, int, None] = 'center',
    excluded: Optional[set]         = None,
) -> List[int]:
    """
    Array-based Kennard-Stone selection in normalised coordinates.

    Picks ``n_select`` indices from ``pool`` so that each chosen point
    is as far as possible (in normalised Euclidean distance) from the
    set of already-chosen points.

    Parameters
    ----------
    sampler : Sampler
        Provides ``sampler.space.gmins`` / ``granges`` /
        ``n_parameters`` for normalisation.
    pool : np.ndarray, shape (N, d)
        Candidate points in the original parameter space.
    n_select : int
        Number of points to pick (``1 ≤ n_select ≤ N``).
    anchor : {'center', 'maximin', int, None}, default 'center'
        Selection strategy for the first point.

        - ``'center'``: the pool point closest to the hypercube
          centre ``[0.5, …, 0.5]`` in normalised space. Recommended
          for run-ordering — keeps the first run in a well-defined
          interior location.
        - ``'maximin'`` (Kennard-Stone original): the two farthest
          pool points first, then continue. Counts as the first two
          selections.
        - ``int``: pool index ``anchor`` first.
        - ``None``: equivalent to ``'center'``.
    excluded : set of int, optional
        Pool indices that must not be picked.

    Returns
    -------
    list of int
        Selected pool indices, in selection order.

    References
    ----------
    Kennard, R. W. & Stone, L. A. (1969). *Technometrics*, 11(1),
        137-148.
    """
    space          = sampler.space
    gmins, granges = space.gmins, space.granges
    norm_pool      = (pool - gmins) / granges
    N              = len(pool)
    n_select       = int(n_select)

    if n_select <= 0:
        return []
    if n_select > N:
        _fatal(f"n_select ({n_select}) exceeds pool size ({N}).")

    excluded  = set() if excluded is None else set(excluded)
    available = set(range(N)) - excluded
    if len(available) < n_select:
        _fatal(
            f"Only {len(available)} pool points are available after "
            f"exclusions; cannot pick {n_select}."
        )

    chosen: List[int] = []

    # ── First-point selection ──────────────────────────────────────
    if anchor is None or anchor == 'center':
        centre = np.full(space.n_parameters, 0.5)
        avail  = np.fromiter(available, dtype=int)
        d_to_c = np.linalg.norm(norm_pool[avail] - centre, axis=1)
        first  = int(avail[int(np.argmin(d_to_c))])
        chosen.append(first)
        available.discard(first)
    elif anchor == 'maximin':
        avail = np.fromiter(available, dtype=int)
        sub   = norm_pool[avail]
        diffs = sub[:, None, :] - sub[None, :, :]
        d2    = np.sum(diffs * diffs, axis=2)
        i, j  = np.unravel_index(int(np.argmax(d2)), d2.shape)
        chosen.extend([int(avail[i]), int(avail[j])])
        available.difference_update(chosen)
    elif isinstance(anchor, (int, np.integer)):
        anchor = int(anchor)
        if anchor not in available:
            _fatal(
                f"anchor index {anchor} is not available "
                f"(out of range or excluded)."
            )
        chosen.append(anchor)
        available.discard(anchor)
    else:
        _fatal(
            f"anchor must be 'center', 'maximin', an int, or None; "
            f"got {anchor!r}."
        )

    # ── Sequential maximin selection ───────────────────────────────
    while len(chosen) < n_select and available:
        avail = np.fromiter(available, dtype=int)
        ref   = norm_pool[chosen]
        cand  = norm_pool[avail]
        diffs = cand[:, None, :] - ref[None, :, :]
        d_min = np.sqrt(np.sum(diffs * diffs, axis=2)).min(axis=1)
        best  = int(avail[int(np.argmax(d_min))])
        chosen.append(best)
        available.discard(best)

    return chosen


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
def extend(
    sampler:         "Sampler",
    existing_design: Union[np.ndarray, pd.DataFrame, list, tuple],
    n_new:           int,
    criteria:        str  = 'cd2',
    algorithm:       Union[str, List[str]] = 'sa',
    n_validation:    int  = 0,
    seed:            Optional[int] = 44,
    verbose:         bool = True,
) -> "SamplingResult":
    """
    Extend an existing design by ``n_new`` space-filling points.

    Every row of ``existing_design`` is treated as a fixed,
    criterion-visible anchor; ``n_new`` additional points are then
    optimised on top of them. The final design has shape
    ``(len(existing_design) + n_new, d)`` and the first
    ``len(existing_design)`` rows are exactly the input rows.

    This is the surrogate-free augmentation strategy of Wang (2003)
    and Qian (2009): preserve the historical runs, add new runs that
    maximise space-filling of the *combined* design.

    Parameters
    ----------
    sampler : Sampler
        Parameter space, constraints, and any pre-set design state
        (focus regions, exclusions, etc.). The sampler is not
        permanently modified; any state added during this call is
        rolled back on exit.
    existing_design : np.ndarray or DataFrame or list
        Points that must appear in the final design unchanged.
        DataFrame columns must include all parameter names.
    n_new : int
        Number of new points to add.
    criteria, algorithm, n_validation, seed, verbose
        Passed through to :meth:`Sampler.run`. Defaults match the
        one-shot workflow.

    Returns
    -------
    SamplingResult

    References
    ----------
    Wang, G. G. (2003). *Journal of Mechanical Design*, 125(2).
    Qian, P. Z. G. (2009). *Biometrika*, 96(4).
    """
    arr, _ = _coerce_design(sampler, existing_design)
    if int(n_new) < 1:
        _fatal(f"n_new must be ≥ 1; got {n_new}.")

    snap = sampler._snapshot_state()
    try:
        for row in arr:
            sampler.add_prescribed([row.tolist()],
                                   in_design=True, in_optim=True)
        sampler.set_design(
            n_samples    = len(arr) + int(n_new),
            n_validation = int(n_validation),
        )
        return sampler.run(criteria=criteria, algorithm=algorithm,
                           seed=seed, verbose=verbose)
    finally:
        sampler._restore_state(snap)


def fill_around(
    sampler:          "Sampler",
    reference_points: Union[np.ndarray, pd.DataFrame, list, tuple],
    n_new:            int,
    criteria:         str  = 'cd2',
    algorithm:        Union[str, List[str]] = 'sa',
    n_validation:     int  = 0,
    seed:             Optional[int] = 44,
    verbose:          bool = True,
) -> "SamplingResult":
    """
    Generate ``n_new`` points that space-fill the region *around* a
    set of reference points.

    Reference points are visible to the criterion (so new points keep
    their distance from them) but are *not* included in the final
    design. Use this when you have e.g. literature or pilot points
    that you do not want to re-run, and you want subsequent runs to
    cover the rest of the space efficiently.

    Parameters
    ----------
    sampler : Sampler
    reference_points : np.ndarray or DataFrame or list
        Coordinates of the points to fill around. Not included in the
        output.
    n_new : int
        Number of new points to generate.
    criteria, algorithm, n_validation, seed, verbose
        See :meth:`Sampler.run`.

    Returns
    -------
    SamplingResult
        Contains exactly ``n_new`` design points.
    """
    arr, _ = _coerce_design(sampler, reference_points)
    if int(n_new) < 1:
        _fatal(f"n_new must be ≥ 1; got {n_new}.")

    snap = sampler._snapshot_state()
    try:
        for row in arr:
            sampler.add_prescribed([row.tolist()],
                                   in_design=False, in_optim=True)
        sampler.set_design(
            n_samples    = int(n_new),
            n_validation = int(n_validation),
        )
        result = sampler.run(criteria=criteria, algorithm=algorithm,
                             seed=seed, verbose=verbose)
    finally:
        sampler._restore_state(snap)

    # Sampler.run() includes ``in_design=False`` prescribed rows in
    # result.samples (with point_type 'Prescribed'). Drop them so the
    # caller receives exactly ``n_new`` rows — the new points whose
    # coordinates do not match any reference row.
    sample_pts = result.samples[sampler.space.names].to_numpy()
    keep       = np.ones(len(sample_pts), dtype=bool)
    for ref in arr:
        keep &= ~np.all(np.isclose(sample_pts, ref), axis=1)
    result.samples = result.samples[keep].reset_index(drop=True)
    return result


def subsample(
    sampler:  "Sampler",
    pool:     Union[np.ndarray, pd.DataFrame],
    n_select: int,
    anchor:   Union[str, int, None] = 'center',
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Pick ``n_select`` space-filling points from a candidate pool.

    Uses Kennard-Stone selection in normalised parameter coordinates:
    each pick maximises the minimum distance to the already-picked
    points.

    Parameters
    ----------
    sampler : Sampler
        Used only for parameter-space metadata (axis ranges, names).
        The sampler is not modified.
    pool : np.ndarray or DataFrame
        Candidate pool. DataFrame columns must include all parameter
        names; the DataFrame index and extra columns are preserved on
        the output. ndarrays return a plain ndarray.
    n_select : int
        Number of points to select. ``1 ≤ n_select ≤ len(pool)``.
    anchor : {'center', 'maximin', int, None}, default 'center'
        How to seed the selection. See ``_kennard_stone_array`` for
        details. ``'center'`` is the most common choice; ``'maximin'``
        reproduces the original Kennard-Stone (1969) behaviour.

    Returns
    -------
    np.ndarray or DataFrame
        The selected rows, in selection order. Type matches the input.

    References
    ----------
    Kennard, R. W. & Stone, L. A. (1969). *Technometrics*, 11(1),
        137-148.
    """
    arr, df = _coerce_design(sampler, pool)
    idx     = _kennard_stone_array(sampler, arr, n_select, anchor=anchor)
    if df is not None:
        return df.iloc[idx].copy()
    return arr[idx].copy()


def run_order(
    sampler: "Sampler",
    design:  Union[np.ndarray, pd.DataFrame],
    anchor:  Union[str, int, None] = 'center',
    column:  str = 'run_order',
) -> pd.DataFrame:
    """
    Assign an execution order so that every cumulative prefix of the
    design remains space-filling.

    Implements a sequential Kennard-Stone reordering: row 0 is the
    chosen anchor; row k (for ``k > 0``) is the design point most
    distant (in normalised Euclidean distance) from rows ``0, …,
    k-1``. Stopping the run after any prefix yields a balanced
    sub-design rather than a clustered one.

    Use this when:

    - Runs are executed serially (wetlab, paid HPC time, single-node
      simulations)
    - You may stop early or hit a budget cut
    - Interim analyses or progress reporting may happen before
      completion

    Parameters
    ----------
    sampler : Sampler
        Provides parameter-space metadata for normalisation.
    design : np.ndarray or DataFrame
        Design to reorder. DataFrame columns must include all
        parameter names; existing DataFrame columns are preserved.
    anchor : {'center', 'maximin', int, None}, default 'center'
        How to pick the first run. See :func:`subsample`.
    column : str, default 'run_order'
        Name of the column added to the returned DataFrame holding
        the execution rank (``0`` = first to run).

    Returns
    -------
    DataFrame
        A copy of ``design`` reordered by execution rank, with the new
        ``column`` column. The original DataFrame index is preserved.

    References
    ----------
    Kennard, R. W. & Stone, L. A. (1969). *Technometrics*, 11(1),
        137-148.
    """
    arr, df = _coerce_design(sampler, design)
    order   = _kennard_stone_array(sampler, arr, len(arr), anchor=anchor)
    if df is None:
        out = pd.DataFrame(arr[order], columns=sampler.space.names)
    else:
        out = df.iloc[order].copy()
    out[column] = range(len(out))
    return out