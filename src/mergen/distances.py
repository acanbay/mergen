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

from typing import TYPE_CHECKING, Optional

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
    :math:`[0, 1]^d`. Nominal columns are compared for equality via
    ``np.isclose`` because they carry float-encoded level indices
    (``0.0``, ``1.0``, ...).

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
    # exact here because nominal levels are stored as integer indices
    # (0.0, 1.0, ...) and Mergen never rescales those columns.
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