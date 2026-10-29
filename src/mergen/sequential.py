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

k_fold_split(sampler, design, k, anchor='center')
    Split a design into k balanced folds so that each fold covers the
    space, for cross-validation of a surrogate fitted on the design.

nested(sampler, n_outer, n_inner, ...)
    Build an outer design and pick a space-filling inner subset of it,
    for two budget tiers on the same space.

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

from .sampler   import _fatal, _warn
from .distances import heom_squared, kennard_stone

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
        # Nominal and ordinal columns may hold labels, as in
        # result.samples.
        arr = space.encode_frame(design[space.names]).to_numpy(
            dtype=float, copy=True)
        return arr, design
    arr = space.encode_points(design, context="design")
    if arr.ndim != 2 or arr.shape[1] != space.n_parameters:
        _fatal(
            f"design must have shape (n, {space.n_parameters}); "
            f"got {arr.shape}."
        )
    return arr, None


def _count(name: str, value) -> int:
    """A positive integer argument; 2.7 or True is refused, not cut."""
    ok = (not isinstance(value, (bool, np.bool_))
          and isinstance(value, (int, np.integer, float, np.floating))
          and float(value).is_integer() and value >= 1)
    if not ok:
        _fatal(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


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
    is as far as possible (HEOM distance in normalised coordinates,
    see :func:`mergen.distances.kennard_stone`) from the set of
    already-chosen points.

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
    n_select       = _count('n_select', n_select)

    # When the space has any nominal factor the ordinary Euclidean
    # distance mixes ordered and unordered dimensions, which is not
    # meaningful. Route through HEOM (Wilson & Martinez 1997), which
    # applies the 0/1 overlap indicator on nominal columns and the
    # normalised Euclidean term on numerical ones. For all-numerical
    # spaces HEOM reduces to squared Euclidean, so the flag defaults
    # to ``None`` (no space handed to the distance helper) and the
    # existing behaviour is preserved bit-for-bit.
    dist_space = space if space.has_nominal else None

    if n_select <= 0:
        return []
    if n_select > N:
        _fatal(f"n_select ({n_select}) exceeds pool size ({N}).")

    excluded = set() if excluded is None else set(excluded)
    free     = np.ones(N, dtype=bool)
    free[[i for i in excluded if 0 <= i < N]] = False
    if int(free.sum()) < n_select:
        _fatal(
            f"Only {int(free.sum())} pool points are available after "
            f"exclusions; cannot pick {n_select}."
        )

    chosen: List[int] = []
    avail = np.flatnonzero(free)

    # ── First-point selection ──────────────────────────────────────
    if anchor is None or anchor == 'center':
        centre = np.full(space.n_parameters, 0.5)
        d_to_c = np.linalg.norm(norm_pool[avail] - centre, axis=1)
        chosen.append(int(avail[int(np.argmin(d_to_c))]))
    elif anchor == 'maximin':
        sub   = norm_pool[avail]
        # (N_avail, N_avail) squared HEOM distance between all pairs.
        d2    = heom_squared(sub[:, None, :], sub[None, :, :],
                             space=dist_space)
        i, j  = np.unravel_index(int(np.argmax(d2)), d2.shape)
        chosen.extend([int(avail[i]), int(avail[j])])
    elif isinstance(anchor, (int, np.integer)):
        anchor = int(anchor)
        if not (0 <= anchor < N and free[anchor]):
            _fatal(
                f"anchor index {anchor} is not available "
                f"(out of range or excluded)."
            )
        chosen.append(anchor)
    else:
        _fatal(
            f"anchor must be 'center', 'maximin', an int, or None; "
            f"got {anchor!r}."
        )
    free[chosen] = False

    # ── Sequential maximin selection ───────────────────────────────
    # The same Kennard-Stone step as the validation and extra sets of
    # Sampler.run, with the anchor rows as the reference set.
    rest = kennard_stone(norm_pool, n_select - len(chosen),
                         ref=norm_pool[chosen], available=free,
                         space=dist_space)
    chosen.extend(int(i) for i in rest)

    return chosen


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
def extend(
    sampler:         "Sampler",
    existing_design: Union[np.ndarray, pd.DataFrame, list, tuple],
    n_new:           int,
    criteria:        Optional[str] = None,
    algorithm:       Union[str, List[str]] = 'sa',
    n_validation:    int  = 0,
    seed:            Optional[int] = 44,
    verbose:         bool = True,
    n_repeats:       Union[int, str] = 'auto',
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
        rolled back on exit. Its prescribed and focus points are not
        added again: the existing design already contains them (a
        prescribed point missing from it is left out with a warning).
        Its extra sets are generated around the extended design.
    existing_design : np.ndarray or DataFrame or list
        Points that must appear in the final design unchanged.
        DataFrame columns must include all parameter names; a
        ``point_type`` column, as in ``result.samples``, keeps its
        labels.
    n_new : int
        Number of new points to add.
    criteria, algorithm, n_validation, seed, verbose, n_repeats
        Passed through to :meth:`Sampler.run`. Defaults match
        :meth:`Sampler.run`, except that no validation points are added
        (``n_validation=0``): the call adds exactly ``n_new`` runs.

    Returns
    -------
    SamplingResult

    References
    ----------
    Wang, G. G. (2003). *Journal of Mechanical Design*, 125(2).
    Qian, P. Z. G. (2009). *Biometrika*, 96(4).
    """
    arr, df = _coerce_design(sampler, existing_design)
    n_new = _count('n_new', n_new)

    tol = sampler.space.grid_tol
    missing = [pt.tolist() for pt, _, _ in sampler._prescribed
               if not np.any(np.all(np.abs(arr - pt) <= tol, axis=1))]
    if missing:
        _warn(f"{len(missing)} prescribed point(s) of the sampler are not "
              f"in existing_design and are left out: {missing}.")

    snap = sampler._snapshot_state()
    try:
        # The existing design already holds the sampler's prescribed and
        # focus points, and a loaded design is what is being extended:
        # add the rows once, as the only fixed points.
        sampler._prescribed.clear()
        sampler._focus.clear()
        sampler._loaded_design = None
        # A row that is an out-of-budget prescribed point of the sampler
        # stays out of the budget, as it was in the existing design.
        out = [any(not in_d and np.all(np.abs(row - pt) <= tol)
                   for pt, in_d, _ in snap['prescribed']) for row in arr]
        for row, is_out in zip(arr, out):
            sampler._add_prescribed_codes(row, in_design=not is_out,
                                          in_optim=True)
        sampler.set_design(
            n_samples    = len(arr) - sum(out) + int(n_new),
            n_validation = int(n_validation),
            extra_sets   = snap['extra_sets'],
        )
        result = sampler.run(criteria=criteria, algorithm=algorithm,
                             seed=seed, n_repeats=n_repeats,
                             verbose=verbose)
    finally:
        sampler._restore_state(snap)

    # The existing rows first, in their input order (an out-of-budget
    # row is otherwise placed after the new ones), then the new rows.
    P = sampler.space.encode_frame(
        result.samples[sampler.space.names]).to_numpy(dtype=float)
    first = [int(np.flatnonzero(np.all(np.abs(P - row) <= tol, axis=1))[0])
             for row in arr]
    rest = [i for i in range(len(P)) if i not in set(first)]
    result.samples = (result.samples.iloc[first + rest]
                      .reset_index(drop=True))
    result.samples.index.name = 'id'

    # Keep the labels the existing rows came with.
    if df is not None and 'point_type' in df.columns:
        col = result.samples.columns.get_loc('point_type')
        result.samples.iloc[:len(arr), col] = df['point_type'].to_numpy()
    return result


def fill_around(
    sampler:          "Sampler",
    reference_points: Union[np.ndarray, pd.DataFrame, list, tuple],
    n_new:            int,
    criteria:         Optional[str] = None,
    algorithm:        Union[str, List[str]] = 'sa',
    n_validation:     int  = 0,
    seed:             Optional[int] = 44,
    verbose:          bool = True,
    n_repeats:        Union[int, str] = 'auto',
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
        Parameter space, constraints, exclusions and user sets. Its
        prescribed points count as further reference points; its focus
        regions and any loaded design are not used. The sampler is
        restored on exit.
    reference_points : np.ndarray or DataFrame or list
        Coordinates of the points to fill around. Not included in the
        output.
    n_new : int
        Number of new points to generate.
    criteria, algorithm, n_validation, seed, verbose, n_repeats
        See :meth:`Sampler.run`. Defaults match it, except that no
        validation points are added (``n_validation=0``).

    Returns
    -------
    SamplingResult
        Contains exactly ``n_new`` design points.
    """
    arr, _ = _coerce_design(sampler, reference_points)
    n_new = _count('n_new', n_new)

    # The sampler's own prescribed points are references too; each
    # point is added once.
    tol  = sampler.space.grid_tol
    refs = []
    for row in list(arr) + [pt for pt, _, _ in sampler._prescribed]:
        if not any(np.all(np.abs(row - r) <= tol) for r in refs):
            refs.append(np.asarray(row, dtype=float))
    arr = np.array(refs)

    snap = sampler._snapshot_state()
    try:
        sampler._prescribed.clear()
        sampler._focus.clear()
        sampler._loaded_design = None
        for row in arr:
            sampler._add_prescribed_codes(row, in_design=False,
                                          in_optim=True, on_grid=False)
        sampler.set_design(
            n_samples    = int(n_new),
            n_validation = int(n_validation),
        )
        result = sampler.run(criteria=criteria, algorithm=algorithm,
                             seed=seed, n_repeats=n_repeats,
                             verbose=verbose)
    finally:
        sampler._restore_state(snap)

    # Sampler.run() includes ``in_design=False`` prescribed rows in
    # result.samples (with point_type 'Prescribed'). Drop them so the
    # caller receives exactly ``n_new`` rows — the new points whose
    # coordinates do not match any reference row.
    sample_pts = sampler.space.encode_frame(
        result.samples[sampler.space.names]).to_numpy(dtype=float)
    keep       = np.ones(len(sample_pts), dtype=bool)
    for ref in arr:
        keep &= ~np.all(np.abs(sample_pts - ref) <= sampler.space.grid_tol,
                        axis=1)
    result.samples = result.samples[keep].reset_index(drop=True)
    result._meta['n_design'] = len(result.samples)
    # Every delivered row was chosen by the optimiser; the references
    # were not available to it.
    result._meta['fixed_points'] = []
    result._meta['baseline_reserved'] = (
        list(result._meta.get('baseline_reserved') or [])
        + np.asarray(arr, dtype=float).tolist())
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
    rows = arr[idx].copy()
    space = sampler.space
    if any(space.is_categorical(n) for n in space.names):
        # Nominal and ordinal levels come back as their labels, so the
        # rows are an object array.
        rows = space.decode_frame(
            pd.DataFrame(rows, columns=space.names)).to_numpy(dtype=object)
    return rows


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
    distant (HEOM distance in normalised coordinates) from rows
    ``0, …, k-1``. Stopping the run after any prefix yields a balanced
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
        out = sampler.space.decode_frame(
            pd.DataFrame(arr[order], columns=sampler.space.names))
    else:
        out = df.iloc[order].copy()
    out[column] = range(len(out))
    return out


def k_fold_split(
    sampler: "Sampler",
    design:  Union[np.ndarray, pd.DataFrame],
    k:       int,
    anchor:  Union[str, int, None] = 'center',
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split a design into ``k`` space-filling folds for cross-validation.

    Returns ``k`` ``(train_indices, test_indices)`` pairs in the same
    format as :class:`sklearn.model_selection.KFold`. Every test fold
    covers the parameter space rather than clustering in a sub-region,
    so cross-validation estimates are not biased by an unlucky split.

    Coverage is measured by a fold's coverage radius: the largest
    distance from any design point to its nearest point of that fold,
    in normalised coordinates (HEOM when the space has nominal
    factors), as in :func:`run_order`. The folds minimise, in this
    order, the largest coverage radius over the folds and the sum of
    the radii.

    Algorithm
    ---------
    The design is first reordered with :func:`run_order` (sequential
    Kennard-Stone), and a round-robin assignment gives the starting
    folds: row 0 → fold 0, row 1 → fold 1, …, row k → fold 0, and so
    on. Pairwise swaps of rows between folds are then made while one
    lowers the objective above; for each row the best swap is taken,
    ties going to the lowest index. This is repeated from several KS
    starting points (see ``anchor``) and the best folds are kept. The
    result is a local optimum, deterministic for a given design.

    Parameters
    ----------
    sampler : Sampler
        Provides parameter-space metadata for the normalisation used
        inside :func:`run_order`.
    design : np.ndarray or DataFrame
        The full design to split. DataFrame columns must include all
        parameter names.
    k : int
        Number of folds. ``2 ≤ k ≤ len(design)``. Folds are as
        balanced as possible — the first ``len(design) mod k`` folds
        receive one extra row each.
    anchor : {'center', 'maximin', int, None}, default 'center'
        First-point strategy of the first KS ordering tried; the
        centre and the maximin pair are tried too, and for designs of
        up to 100 rows the first eight rows, and the best folds are
        kept. When starts give equally good folds, the requested
        ``anchor`` wins. See :func:`subsample`.

    Returns
    -------
    list of (train_indices, test_indices)
        Position-based indices (``0 ≤ idx < len(design)``) into the
        input ``design``. Compatible with
        ``design.iloc[idx]`` for DataFrames and ``design[idx]`` for
        ndarrays.

    References
    ----------
    Kennard, R. W. & Stone, L. A. (1969). *Technometrics*, 11(1),
        137-148.
    Qian, P. Z. G. (2012). Sliced Latin hypercube designs.
        *Journal of the American Statistical Association*, 107(497),
        393-399. [Related concept of slicing a design into
        space-filling sub-designs.]
    """
    arr, _ = _coerce_design(sampler, design)
    n      = len(arr)
    k      = _count('k', k)
    if k < 2:
        _fatal(f"k must be ≥ 2; got {k}.")
    if k > n:
        _fatal(f"k ({k}) cannot exceed design size ({n}).")

    space = sampler.space
    norm  = (arr - space.gmins) / space.granges
    from .distances import heom_pairwise
    D = heom_pairwise(norm, space=space if space.has_nominal else None)

    # Several deterministic starts (the requested anchor first), each
    # a KS ordering dealt round-robin and improved by swaps; the best
    # result is kept, ties going to the earlier start.
    starts = list(dict.fromkeys(
        [anchor, 'center', 'maximin']
        + (list(range(min(n, 8))) if n <= _FOLD_ROW_STARTS_MAX else [])))
    best = None
    for start in starts:
        order = np.asarray(_kennard_stone_array(sampler, arr, n,
                                                anchor=start), dtype=int)
        fold = np.empty(n, dtype=int)
        for rank, pos in enumerate(order):
            fold[pos] = rank % k
        fold = _swap_folds(D, fold, k)
        radii = [float(D[:, fold == f].min(axis=1).max()) for f in range(k)]
        key = (round(max(radii), 12), round(sum(radii), 12))
        if best is None or key < best[0]:
            best = (key, fold)
    fold_of_pos = best[1]

    # Build (train_idx, test_idx) pairs
    all_idx = np.arange(n)
    return [(all_idx[fold_of_pos != f], all_idx[fold_of_pos == f])
            for f in range(k)]


# Sweeps of the fold-swap search; it usually stops after a few, and
# the cap keeps very large designs bounded (a count, so deterministic).
_FOLD_SWEEPS = 50
# Up to this design size the first eight rows serve as extra starts.
_FOLD_ROW_STARTS_MAX = 100


def _max_min(cap, D, cols):
    """
    ``max_i min(cap[i], D[i, b])`` for every ``b`` in ``cols``, exactly.

    Row ``i`` contributes at most ``cap[i]``, so the maximum over the
    rows with the largest caps is exact whenever it reaches the
    largest cap left out; the other columns are computed over all rows.
    """
    n = len(cap)
    m = max(32, n // 8)
    if n <= 2 * m:
        return np.minimum(cap[:, None], D[:, cols]).max(axis=0)
    top = np.argpartition(-cap, m)[:m]
    t   = float(np.delete(cap, top).max())
    out = np.minimum(cap[top, None], D[np.ix_(top, cols)]).max(axis=0)
    redo = out < t
    if redo.any():
        out[redo] = np.minimum(cap[:, None], D[:, cols[redo]]).max(axis=0)
    return out


def _radius_without(low, high, arg, cols):
    """
    For every ``b`` in ``cols``: ``max_i`` of ``high[i]`` where
    ``arg[i] == b`` and of ``low[i]`` elsewhere (``high >= low``).

    The rows whose nearest member is ``b`` take ``high``; for the rest
    the largest ``low`` comes from the two groups with the largest
    ``low`` maxima, so the whole is O(n).
    """
    n_lab   = max(int(arg.max()), int(cols.max())) + 1
    low_g   = np.full(n_lab, -np.inf)
    high_g  = np.full(n_lab, -np.inf)
    np.maximum.at(low_g, arg, low)
    np.maximum.at(high_g, arg, high)
    g1 = int(np.argmax(low_g))
    second = np.delete(low_g, g1).max() if n_lab > 1 else -np.inf
    rest = np.where(cols == g1, second, low_g[g1])
    return np.maximum(rest, high_g[cols])


def _swap_folds(D: np.ndarray, fold: np.ndarray, k: int) -> np.ndarray:
    """
    Improve a fold assignment by pairwise swaps (see k_fold_split).

    Fold f's coverage radius is max_i min_{j in f} D[i, j]. A swap of
    row a (fold A) with row b (fold B) changes only A's and B's radii.
    Each row's nearest and second-nearest member of every fold are
    kept, so the new radii of all candidate b of a fold are found at
    once; they are refreshed for the two folds a swap changes.
    """
    fold = fold.copy()
    n    = len(D)
    cols = np.arange(n)

    def nearest_two(f):
        members = np.flatnonzero(fold == f)
        Dm = D[:, members]
        if len(members) == 1:
            return Dm[:, 0], np.full(n, np.inf), np.full(n, members[0])
        part = np.argpartition(Dm, 1, axis=1)[:, :2]
        d0 = Dm[cols, part[:, 0]]
        d1 = Dm[cols, part[:, 1]]
        swap = d1 < d0
        m1 = np.where(swap, d1, d0)
        m2 = np.where(swap, d0, d1)
        arg = members[np.where(swap, part[:, 1], part[:, 0])]
        return m1, m2, arg

    near  = [nearest_two(f) for f in range(k)]
    radii = np.array([near[f][0].max() for f in range(k)])

    def key(mx, sm):
        # Rounded, so floating-point noise cannot drive a swap.
        return np.round(mx, 12), np.round(sm, 12)

    for _ in range(_FOLD_SWEEPS):
        improved = False
        for a in range(n):
            A = fold[a]
            m1, m2, arg = near[A]
            # A without a: each row's nearest other member of A.
            near_a = np.where(arg == a, m2, m1)
            cur = key(radii.max(), radii.sum())
            best = None
            for B in range(k):
                if B == A:
                    continue
                in_b = np.flatnonzero(fold == B)
                b1, b2, barg = near[B]
                new_a = _max_min(near_a, D, in_b)
                new_b = _radius_without(np.minimum(D[:, a], b1),
                                        np.minimum(D[:, a], b2), barg, in_b)
                rest = np.delete(radii, [A, B])
                mx = np.maximum(np.maximum(new_a, new_b),
                                rest.max() if len(rest) else -np.inf)
                sm = rest.sum() + new_a + new_b
                kmx, ksm = key(mx, sm)
                t = np.lexsort((in_b, ksm, kmx))[0]           # best, low index
                cand = (kmx[t], ksm[t])
                if cand < cur and (best is None or cand < best[0]):
                    best = (cand, int(in_b[t]), B,
                            float(new_a[t]), float(new_b[t]))
            if best is not None:
                _, b, B, ra, rb = best
                fold[a], fold[b] = B, A
                radii[A], radii[B] = ra, rb
                near[A], near[B] = nearest_two(A), nearest_two(B)
                improved = True
        if not improved:
            break
    return fold


def nested(
    sampler:   "Sampler",
    n_outer:   int,
    n_inner:   int,
    criteria:  Optional[str] = None,
    algorithm: Union[str, List[str]] = 'sa',
    seed:      Optional[int] = 44,
    verbose:   bool = True,
    n_repeats: Union[int, str] = 'auto',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a nested pair of designs: a small *inner* design that is a
    subset of a larger *outer* design, both space-filling.

    Useful for multi-fidelity computer experiments, hierarchical
    sensitivity studies, and any setting where two budget tiers run
    on the same parameter space (e.g. fast/slow simulation, pilot vs.
    main wetlab study). Every row of ``inner`` appears verbatim in
    ``outer``.

    Algorithm
    ---------
    1. Build the outer design via :meth:`Sampler.run` on the current
       Sampler state (``n_samples = n_outer``).
    2. Pick the inner design as a maximin Kennard-Stone subsample of
       the ``n_outer`` budget rows of the outer design
       (:func:`subsample` with ``anchor='maximin'``).

    The resulting inner design is space-filling within outer, and
    every outer point is space-filling in the full parameter space.

    The classical He & Qian (2011) construction enforces a strict
    LHS-preserving nesting; this pragmatic implementation does not
    guarantee that, which is acceptable for the typical use case of
    multi-fidelity computer experiments (where space-fillingness
    matters more than the LHS marginals of the inner design). A
    strict LHS-nested variant is planned for a future release.

    Parameters
    ----------
    sampler : Sampler
    n_outer : int
        Size of the outer (larger) design.
    n_inner : int
        Size of the inner (smaller) design. ``1 ≤ n_inner < n_outer``.
    criteria, algorithm, seed, verbose, n_repeats
        Passed to :meth:`Sampler.run` when building the outer design.

    Returns
    -------
    outer : DataFrame
        ``result.samples`` of the outer run (the ``d`` parameter
        columns and ``'point_type'``) plus a boolean ``'in_inner'``
        column flagging the rows of the inner design. It has
        ``n_outer`` rows, more when prescribed or focus points outside
        the budget (``in_design=False``) are kept in the samples; those
        extra rows come last and are never in the inner design.
    inner : DataFrame, shape (n_inner, d + 2)
        The inner rows of ``outer``, in the order they appear in
        ``outer`` (not in selection order). Same columns as
        ``outer``; ``'in_inner'`` is ``True`` throughout.

    References
    ----------
    He, X. & Qian, P. Z. G. (2011). Nested orthogonal array-based
        Latin hypercube designs. *Biometrika*, 98(3), 721-731.
    Qian, P. Z. G. (2009). Nested Latin hypercube designs.
        *Biometrika*, 96(4), 957-970.
    """
    n_outer = _count('n_outer', n_outer)
    n_inner = _count('n_inner', n_inner)
    if n_outer <= n_inner:
        _fatal(
            f"n_outer ({n_outer}) must be strictly greater than "
            f"n_inner ({n_inner})."
        )

    # ── Step 1: outer design via Sampler.run ────────────────────────
    snap = sampler._snapshot_state()
    try:
        sampler.set_design(n_samples=n_outer, n_validation=0)
        outer_result = sampler.run(criteria=criteria, algorithm=algorithm,
                                   seed=seed, n_repeats=n_repeats,
                                   verbose=verbose)
    finally:
        sampler._restore_state(snap)

    outer = outer_result.samples.copy()
    # Out-of-budget prescribed and focus points follow the n_outer
    # budget rows; they are not run, so the inner design skips them.
    n_run = len(outer) - len(outer_result._extra_pts)
    pts   = sampler.space.encode_frame(
        outer[sampler.space.names].iloc[:n_run]).to_numpy(dtype=float)

    # ── Step 2: inner = maximin KS subsample of outer ───────────────
    inner_pos = _kennard_stone_array(sampler, pts, n_inner, anchor='maximin')
    inner_pos = sorted(inner_pos)        # stable order in outer.index

    mask = np.zeros(len(outer), dtype=bool)
    mask[inner_pos] = True
    outer['in_inner'] = mask

    inner = outer.iloc[inner_pos].copy()
    return outer, inner
