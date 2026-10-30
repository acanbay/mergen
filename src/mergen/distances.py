"""
mergen.distances
================
Distance metrics for mixed continuous / categorical parameter spaces.

Mergen supports parameter spaces that mix numerical (continuous,
discrete, integer, or ordinal) and nominal (unordered categorical)
factors. The nearest-neighbour and space-filling machinery of the
package therefore needs a distance function that treats each column
according to its type. The Heterogeneous Euclidean-Overlap Metric
(HEOM) of Wilson & Martinez (1997) is the standard choice: it
degenerates to the ordinary normalised Euclidean distance when no
nominal columns are present, so callers that never use nominal
factors are not affected.

For two rows :math:`x_a, x_b` with :math:`p` numerical columns and
:math:`q` nominal columns (:math:`d = p + q` total):

.. math::

    d_\\mathrm{HEOM}(x_a, x_b) \\;=\\; \\sqrt{\\sum_{j=1}^{d} d_j(x_{aj}, x_{bj})^2}

where the per-column term is

.. math::

    d_j = \\begin{cases}
        |x_{aj} - x_{bj}| / R_j & \\text{numerical column,} \\\\
        \\mathbb{1}(x_{aj} \\neq x_{bj}) & \\text{nominal column.}
    \\end{cases}

Here :math:`R_j = \\max_i x_{ij} - \\min_i x_{ij}` is the observed
range of column :math:`j`. The design coordinates handed to Mergen
optimisers are already normalised to :math:`[0, 1]^d`, so numerical
columns pass through as ``|x_{aj} - x_{bj}|`` with no rescaling.

References
----------
Wilson, D. R. & Martinez, T. R. (1997). Improved heterogeneous
    distance functions. *Journal of Artificial Intelligence
    Research*, 6, 1-34.
Gower, J. C. (1971). A general coefficient of similarity and some
    of its properties. *Biometrics*, 27(4), 857-871.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .space import ParameterSpace


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────
def _nominal_mask(
    space:         Optional["ParameterSpace"],
    n_columns:     int,
    nominal_mask:  Optional[np.ndarray],
) -> np.ndarray:
    """
    Resolve the boolean nominal-column mask from either an explicit
    override or the parameter space.

    Returns an ``(n_columns,)`` boolean array; ``True`` at column
    ``j`` marks that column as nominal.
    """
    if nominal_mask is not None:
        mask = np.asarray(nominal_mask, dtype=bool)
        if mask.shape != (n_columns,):
            raise ValueError(
                f"nominal_mask must have shape ({n_columns},); "
                f"got {mask.shape}."
            )
        return mask
    if space is not None:
        mask = space.is_mask
        if mask.shape != (n_columns,):
            raise ValueError(
                f"Space has {mask.shape[0]} parameters but the data has "
                f"{n_columns} columns."
            )
        return mask
    # No source of type information -> treat every column as numerical.
    return np.zeros(n_columns, dtype=bool)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
def heom_squared(
    a:            np.ndarray,
    b:            np.ndarray,
    space:        Optional["ParameterSpace"] = None,
    nominal_mask: Optional[np.ndarray]        = None,
) -> np.ndarray:
    """
    Squared HEOM distance between corresponding rows of *a* and *b*.

    Computing squared distances avoids the ``sqrt`` call, which is a
    significant saving in the inner loops of maximin/Kennard-Stone
    selection where only distance *ordering* is needed. Callers that
    require the actual distance can take ``np.sqrt`` of the return
    value.

    Both arrays must already be in the normalised design space
    :math:`[0, 1]^d`. Nominal columns are compared for exact equality:
    both arrays carry the same normalised level codes, so equal levels
    have identical floats.

    Parameters
    ----------
    a, b : np.ndarray
        Broadcast-compatible arrays of shape ``(..., d)``. The last
        axis indexes the parameter columns.
    space : ParameterSpace, optional
        Source for the nominal-column mask. Ignored if
        ``nominal_mask`` is given explicitly.
    nominal_mask : array-like of bool, shape (d,), optional
        Column-wise mask marking nominal columns. When neither
        ``space`` nor ``nominal_mask`` is provided, every column is
        treated as numerical and HEOM reduces to the squared
        normalised Euclidean distance.

    Returns
    -------
    np.ndarray
        Squared HEOM distances, with the broadcast shape of ``a`` and
        ``b`` minus their trailing column axis.

    References
    ----------
    Wilson, D. R. & Martinez, T. R. (1997). *J. Artif. Intell. Res.*,
        6, 1-34.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"Last-axis mismatch: a has {a.shape[-1]} columns, "
            f"b has {b.shape[-1]}."
        )
    d    = a.shape[-1]
    mask = _nominal_mask(space, d, nominal_mask)

    diff = a - b
    if not mask.any():
        return np.sum(diff * diff, axis=-1)

    # Nominal columns: substitute the 0/1 overlap indicator for the
    # normalised absolute difference. Boolean comparison on floats is
    # exact here because both arrays hold the same normalised level
    # codes, computed the same way.
    d_col       = np.abs(diff)
    d_col[..., mask] = (a[..., mask] != b[..., mask]).astype(float)
    return np.sum(d_col * d_col, axis=-1)


def heom(
    a:            np.ndarray,
    b:            np.ndarray,
    space:        Optional["ParameterSpace"] = None,
    nominal_mask: Optional[np.ndarray]        = None,
) -> np.ndarray:
    """
    HEOM distance between corresponding rows of *a* and *b*.

    Convenience wrapper around :func:`heom_squared` that returns the
    actual distance (i.e., with the ``sqrt`` applied). Use
    :func:`heom_squared` directly in tight loops when only distance
    *ordering* matters.

    See :func:`heom_squared` for parameter documentation.

    References
    ----------
    Wilson, D. R. & Martinez, T. R. (1997). *J. Artif. Intell. Res.*,
        6, 1-34.
    """
    return np.sqrt(heom_squared(a, b, space=space, nominal_mask=nominal_mask))


def heom_pairwise(
    X:            np.ndarray,
    space:        Optional["ParameterSpace"] = None,
    nominal_mask: Optional[np.ndarray]        = None,
    squared:      bool                        = False,
) -> np.ndarray:
    """
    Full pairwise HEOM distance matrix for a design ``X``.

    Runs in :math:`O(n^2 d)` time and uses :math:`O(n^2)` memory. For
    large designs where only the nearest-neighbour distance per row is
    needed, prefer computing row-vs-set distances with
    :func:`heom_squared` in a loop rather than materialising the full
    matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Design in normalised coordinates.
    space, nominal_mask
        See :func:`heom_squared`.
    squared : bool, default False
        Return squared distances when ``True``. Saves an ``np.sqrt``
        call when only the ordering is required (e.g., inside
        Kennard-Stone selection).

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric pairwise distance matrix. The diagonal is 0.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}.")
    n, d = X.shape
    mask = _nominal_mask(space, d, nominal_mask)

    diff = X[:, None, :] - X[None, :, :]         # (n, n, d)
    if mask.any():
        eq_nom = (X[:, None, mask] == X[None, :, mask]).astype(float)
        d_col       = np.abs(diff)
        d_col[..., mask] = 1.0 - eq_nom          # 0/1 overlap indicator
        sq = np.sum(d_col * d_col, axis=-1)
    else:
        sq = np.sum(diff * diff, axis=-1)

    # Numerical zero on the diagonal
    np.fill_diagonal(sq, 0.0)
    return sq if squared else np.sqrt(sq)


def _embed_heom(Z: np.ndarray, levels: List[Optional[np.ndarray]]) -> np.ndarray:
    """
    Embed normalised points so that Euclidean distance equals HEOM.

    Numeric columns are kept; a nominal column becomes a one-hot block
    scaled by 1/sqrt(2), so two different levels are at distance 1 and
    equal levels at distance 0 (Wilson & Martinez 1997).
    """
    cols = []
    for j, lv in enumerate(levels):
        if lv is None:
            cols.append(Z[:, [j]])
        else:
            cols.append((Z[:, [j]] == lv[None, :]) / np.sqrt(2.0))
    return np.hstack(cols)


#: Relative tolerance under which kennard_stone() treats distances as tied.
_KS_RTOL = 1e-9


def kennard_stone(
    pool:      np.ndarray,
    n:         int,
    ref:       Optional[np.ndarray]       = None,
    available: Optional[np.ndarray]       = None,
    space:     Optional["ParameterSpace"] = None,
) -> np.ndarray:
    """
    Kennard-Stone selection of pool rows, farthest from a reference set.

    Each step picks the available row with the largest HEOM distance
    to the reference set plus the rows picked so far. The scan covers
    the whole pool, so the result is deterministic. Distances that
    agree to a relative ``1e-9`` count as tied, and ties go to the
    lowest row index.

    Parameters
    ----------
    pool : np.ndarray, shape (N, d)
        Candidate points in normalised coordinates.
    n : int
        Number of rows to pick. Fewer are returned when the available
        rows run out.
    ref : np.ndarray, shape (m, d), optional
        Reference points in normalised coordinates. With no reference,
        the first pick is the lowest available row index; Kennard &
        Stone (1969) instead start from the two mutually farthest rows,
        so pass a reference when the start matters.
    available : np.ndarray of bool, shape (N,), optional
        Rows that may be picked; all rows by default.
    space : ParameterSpace, optional
        Supplies the nominal-column mask (see :func:`heom_squared`).

    Returns
    -------
    np.ndarray of int
        Selected row indices, in selection order.

    References
    ----------
    Kennard, R. W. & Stone, L. A. (1969). *Technometrics*, 11(1),
        137-148.
    """
    pool = np.asarray(pool, dtype=float)
    N    = len(pool)
    free = (np.ones(N, dtype=bool) if available is None
            else np.array(available, dtype=bool, copy=True))
    mask = _nominal_mask(space, pool.shape[1], None)
    R    = (np.empty((0, pool.shape[1])) if ref is None
            else np.asarray(ref, dtype=float).reshape(-1, pool.shape[1]))

    # Work in an embedding where Euclidean distance equals HEOM.
    levels = [np.unique(np.concatenate([pool[:, j], R[:, j]]))
              if mask[j] else None for j in range(pool.shape[1])]
    X = _embed_heom(pool, levels) if mask.any() else pool

    # Squared distance of every pool row to its nearest reference point.
    d2 = np.full(N, np.inf)
    if len(R):
        from scipy.spatial import cKDTree
        Y = _embed_heom(R, levels) if mask.any() else R
        dist, _ = cKDTree(Y).query(X, k=1)
        d2 = dist * dist

    chosen = []
    for _ in range(max(int(n), 0)):
        if not free.any():
            break
        cand = np.where(free, d2, -1.0)
        # Distances equal up to rounding count as tied, so the choice
        # does not depend on the order of floating-point operations.
        i = int(np.argmax(cand >= cand.max() * (1.0 - _KS_RTOL)))
        chosen.append(i)
        free[i] = False
        diff = X - X[i]
        np.minimum(d2, np.einsum('ij,ij->i', diff, diff), out=d2)
    return np.asarray(chosen, dtype=int)


# Search budget of maximin_subset: at most this many candidate rows
# after the Kennard-Stone bound, and this many candidate checks (about
# two seconds on a current desktop). Counts, not times, so the
# selection is the same on every machine.
_MAXIMIN_CANDIDATES = 2_000
_MAXIMIN_WORK       = 1_500_000_000
# Each partial set also costs this many checks: the fixed Python
# overhead of visiting it, which dominates on small pools.
_MAXIMIN_NODE_COST  = 5_000


def maximin_subset(
    pool:      np.ndarray,
    n:         int,
    ref:       Optional[np.ndarray]       = None,
    available: Optional[np.ndarray]       = None,
    space:     Optional["ParameterSpace"] = None,
) -> np.ndarray:
    """
    Best set of *n* pool rows by the maximin rule, chosen as a whole.

    A set S is scored, in order, by

    1. the smallest distance from a member of S to the reference set or
       to another member of S (maximin augmentation of the reference
       set; Johnson, Moore & Ylvisaker 1990),
    2. the smallest distance between two members of S, and
    3. the summed distance of the members to the reference set,

    each larger being better; distances are HEOM in normalised
    coordinates and agree when equal to nine decimals. Remaining ties
    go to the set of lowest row indices. Kennard-Stone
    (:func:`kennard_stone`) builds the set one row at a time and can miss
    this optimum; its result gives the bound that prunes the exact
    branch-and-bound search here. When more than
    ``_MAXIMIN_CANDIDATES`` rows pass that bound, the Kennard-Stone set
    is returned; when the search needs more than ``_MAXIMIN_WORK``
    candidate checks, the best set found so far is returned, which is
    never worse than the Kennard-Stone set. Both cases warn, since the
    set is then not proven optimal.

    Parameters are as for :func:`kennard_stone`. The selected row
    indices are returned in ascending order, not in selection order.

    References
    ----------
    Johnson, M. E., Moore, L. M. & Ylvisaker, D. (1990). Minimax and
        maximin distance designs. *Journal of Statistical Planning and
        Inference*, 26(2), 131-148.
    Kennard, R. W. & Stone, L. A. (1969). *Technometrics*, 11(1),
        137-148.
    """
    pool = np.asarray(pool, dtype=float)
    ks   = kennard_stone(pool, n, ref=ref, available=available, space=space)
    if len(ks) < 2:
        return ks
    n    = len(ks)
    free = (np.ones(len(pool), dtype=bool) if available is None
            else np.asarray(available, dtype=bool))
    mask = _nominal_mask(space, pool.shape[1], None)
    R    = (np.empty((0, pool.shape[1])) if ref is None
            else np.asarray(ref, dtype=float).reshape(-1, pool.shape[1]))
    levels = [np.unique(np.concatenate([pool[:, j], R[:, j]]))
              if mask[j] else None for j in range(pool.shape[1])]
    X = _embed_heom(pool, levels) if mask.any() else pool

    if len(R):
        from scipy.spatial import cKDTree
        Y = _embed_heom(R, levels) if mask.any() else R
        d_ref, _ = cKDTree(Y).query(X, k=1)
    else:
        d_ref = np.full(len(pool), np.inf)
    d_ref = np.round(d_ref, 9)

    def key(S):
        P  = X[list(S)]
        dd = np.round(np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2)
                              .sum(-1)), 9)
        pw = float(dd[np.triu_indices(len(S), 1)].min())
        cl = float(d_ref[list(S)].min())
        return (min(cl, pw), pw, round(float(d_ref[list(S)].sum()), 9))

    best_set = tuple(sorted(int(i) for i in ks))
    best_key = key(best_set)
    # Every member of a set at least as good lies this far from the
    # reference set.
    cand = np.flatnonzero(free & (d_ref >= best_key[0]))
    if len(cand) > _MAXIMIN_CANDIDATES:
        warnings.warn(
            f"maximin_subset: {len(cand)} candidate rows exceed the exact "
            f"search limit of {_MAXIMIN_CANDIDATES}; the Kennard-Stone "
            f"set is used and is not proven to be the maximin set.",
            stacklevel=2)
        return ks
    from scipy.spatial.distance import cdist
    Dc = np.round(cdist(X[cand], X[cand]), 9)
    dr = d_ref[cand]
    m  = len(cand)
    work = 0

    # S holds positions in cand; near[t] is the distance from cand[t] to
    # the nearest member of S. cl and pw only shrink as members are
    # added, so a partial set already below the best key is cut.
    def grow(S, cl, pw, start, near):
        nonlocal best_set, best_key, work
        if (min(cl, pw), pw) < best_key[:2]:
            return
        if len(S) == n:
            k = (min(cl, pw), pw, round(float(dr[S].sum()), 9))
            T = tuple(int(cand[t]) for t in S)
            if k > best_key or (k == best_key and T < best_set):
                best_set, best_key = T, k
            return
        stop = m - (n - len(S)) + 1
        if start >= stop:
            return
        work += stop - start
        new_pw = np.minimum(pw, near[start:stop])
        val    = np.minimum(np.minimum(cl, dr[start:stop]), new_pw)
        for o in np.flatnonzero(val >= best_key[0]):
            # best_key may have risen since val was screened.
            if val[o] < best_key[0]:
                continue
            t = start + int(o)
            work += m + _MAXIMIN_NODE_COST
            if work > _MAXIMIN_WORK:
                raise _SearchTooLarge
            grow(S + [t], min(cl, dr[t]), float(new_pw[o]), t + 1,
                 np.minimum(near, Dc[t]))

    try:
        grow([], np.inf, np.inf, 0, np.full(m, np.inf))
    except _SearchTooLarge:
        warnings.warn(
            "maximin_subset: the exact search reached its work limit; the "
            "best set found so far is used and is not proven to be the "
            "maximin set.", stacklevel=2)
    return np.asarray(best_set, dtype=int)


class _SearchTooLarge(Exception):
    """maximin_subset exceeded its search budget."""
