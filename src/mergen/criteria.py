"""
mergen.criteria
===============
Optimisation criteria for the Stochastic Coordinate Exchange (SCE) engine.

Each criterion exposes four methods:

    evaluate(X, space)
        Full computation on a normalised design matrix.
        Used once per SCE restart to obtain the initial score.

    incremental(X, i, new_pt, space, current_score)
        O(n·d) update when the entire point *i* is replaced by ``new_pt``.
        Used by single-point swap operators (legacy SA path, perturbation
        kicks).  Returns ``(log_delta, new_raw_score)``.

    begin_1d(X, i, current_score)
        Initialise a per-point cache for fast 1D coordinate-exchange trials.

    try_1d(cache, axis, new_value, space)
        O(n) update when only one coordinate of point *i* changes.
        This is the inner-loop operation of the SCE algorithm
        (Meyer & Nachtsheim 1995; Kang 2019).
        Returns ``(log_delta, new_score, new_cache)``.

All criteria operate on **normalised** coordinates in [0, 1]^d.

Available criteria
------------------
    'umaxpro'     UMaxPro    (default) — Vorechovsky & Elias (2026)
    'maxpro'      MaxPro               — Joseph, Gul & Ba (2015)
    'phi_p'       PhiP (p=15)          — Morris & Mitchell (1995)
    'cd2'         CD2 discrepancy      — Hickernell (1998)
    'stratified'  Stratified L2        — Tian & Xu (2025)

Usage
-----
    from mergen.criteria import get_criterion

    crit  = get_criterion('umaxpro')
    score = crit.evaluate(X_norm, space)

    # Coordinate-exchange inner loop:
    cache = crit.begin_1d(X_norm, i, score)
    for axis in range(d):
        log_delta, new_score, new_cache = crit.try_1d(cache, axis, value, space)
        if log_delta < 0:                       # improvement → accept
            X_norm[i, axis] = value
            cache, score    = new_cache, new_score

References
----------
Vorechovsky & Elias (2026), Computers & Structures.        [uMaxPro]
Joseph, Gul & Ba (2015), Biometrika 102(2).                [MaxPro]
Morris & Mitchell (1995), J. Statist. Plan. Infer. 43.     [phi_p]
Hickernell (1998), Math. Comp. 67.                         [CD2]
Tian & Xu (2025), J. Royal Statist. Soc. B 88(2).          [Stratified L2]
Meyer & Nachtsheim (1995), Technometrics 37(1).            [Coordinate exchange]
Kang (2019), Quality Engineering 31(3).                    [SCE]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

# ── small floor to avoid log(0) and division by zero ─────────────────────
_EPS = 1e-10


# ======================================================================
# Base class
# ======================================================================

class BaseCriterion(ABC):
    """
    Abstract base for space-filling design criteria.

    Subclasses implement :meth:`evaluate` and :meth:`incremental`.
    Fast 1D coordinate-exchange updates (:meth:`begin_1d`,
    :meth:`try_1d`) are provided in the base class as a fallback and
    may be overridden by criteria with separable structure for an
    O(n) inner loop.
    """

    @abstractmethod
    def evaluate(self, X: np.ndarray, space) -> float:
        """
        Compute the criterion value for design *X*.

        Parameters
        ----------
        X     : np.ndarray, shape (n, d) — normalised coordinates in [0, 1]^d
        space : ParameterSpace — used for dimension metadata if needed

        Returns
        -------
        float — raw criterion score (not log-transformed)
        """

    @abstractmethod
    def incremental(
        self,
        X:             np.ndarray,
        i:             int,
        new_pt:        np.ndarray,
        space,
        current_score: float,
    ) -> Tuple[float, float]:
        """
        O(n·d) incremental update when point *i* is swapped for *new_pt*.

        Parameters
        ----------
        X             : np.ndarray, shape (n, d) — current normalised design
        i             : int — index of the point being swapped
        new_pt        : np.ndarray, shape (d,) — candidate replacement (normalised)
        space         : ParameterSpace
        current_score : float — raw criterion score before the swap

        Returns
        -------
        log_delta : float
            log(new_score) - log(current_score).
            Negative → improvement; positive → degradation.
        new_score : float
            Raw criterion score after the swap.
        """

    def begin_1d(
        self,
        X:             np.ndarray,
        i:             int,
        current_score: float,
    ):
        """
        Initialise a per-point cache for fast 1D coordinate-exchange updates.

        Called once before trying multiple coordinate changes on the same
        point *i*.  The returned cache object is then passed to
        :meth:`try_1d` for each candidate value.

        Subclasses with separable structures (UMaxPro, MaxPro, PhiP) override
        this to precompute per-pair contributions, enabling O(n) per-try
        updates instead of O(n·d).

        Default implementation stores ``(X, i, current_score)`` for the
        base :meth:`try_1d` fallback, which builds the full new point
        and delegates to :meth:`incremental`.

        Parameters
        ----------
        X             : np.ndarray, shape (n, d) — current normalised design
        i             : int — index of the point to be modified
        current_score : float — raw criterion score before any change

        Returns
        -------
        cache : object
            Opaque structure used by :meth:`try_1d`.  Subclass-specific.
        """
        return {'X': X, 'i': i, 'score': current_score}

    def try_1d(
        self,
        cache,
        axis:      int,
        new_value: float,
        space,
    ) -> Tuple[float, float, object]:
        """
        Try changing one coordinate of the cached point and return the delta.

        For separable criteria, this runs in O(n).  For non-separable
        criteria (CD2, StratifiedL2), the base implementation falls back
        to the full O(n·d) :meth:`incremental` evaluation.

        The returned cache reflects the *trial* state and may be reused for
        further trials on the same point.  It does not modify the original
        design array.

        Parameters
        ----------
        cache     : object — returned by :meth:`begin_1d`
        axis      : int — coordinate index being changed (0 ≤ axis < d)
        new_value : float — new normalised value at coordinate ``axis``
        space     : ParameterSpace

        Returns
        -------
        log_delta : float — log(new_score) − log(current_score)
        new_score : float — raw criterion score after the change
        new_cache : object — cache updated to reflect the trial change
        """
        X  = cache['X']
        i  = cache['i']
        s  = cache['score']
        new_pt       = X[i].copy()
        new_pt[axis] = new_value
        log_delta, new_score = self.incremental(X, i, new_pt, space, s)
        new_cache = {'X': X, 'i': i, 'score': new_score}
        return log_delta, new_score, new_cache

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ======================================================================
# uMaxPro  (default criterion)
# ======================================================================

class UMaxPro(BaseCriterion):
    """
    Uniform MaxPro criterion with periodic (toroidal) distance.

    Defines the per-axis squared periodic distance::

        δ²_v(i, j) = min(|x_iv − x_jv|,  1 − |x_iv − x_jv|)²

    The criterion to minimise is then::

        uMaxPro(X) = Σ_{i<j}  1 / prod_v  δ²_v(i, j)

    The toroidal wrap removes the boundary bias of plain MaxPro: corner
    regions are no longer artificially far from each other, so the
    resulting design is statistically uniform across the entire
    hypercube including its boundaries.

    References
    ----------
    Vorechovsky, M. & Elias, J. (2026).
        *Computers & Structures*.  [uMaxPro]
    Joseph, V. R., Gul, E. & Ba, S. (2015).
        *Biometrika*, 102(2), 371–380.  [original MaxPro]
    """

    @staticmethod
    def _periodic_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Periodic squared distance element-wise: min(|a-b|, 1-|a-b|)²."""
        d = np.abs(a - b)
        return np.minimum(d, 1.0 - d) ** 2

    def evaluate(self, X: np.ndarray, space) -> float:
        n = len(X)
        score = 0.0
        for i in range(n - 1):
            sq    = np.maximum(self._periodic_sq(X[i], X[i + 1:]), _EPS)
            score += np.sum(1.0 / np.prod(sq, axis=1))
        return max(score, _EPS)

    def incremental(self, X, i, new_pt, space, current_score):
        mask   = np.ones(len(X), dtype=bool);  mask[i] = False
        others = X[mask]

        sq_old = np.maximum(self._periodic_sq(X[i],   others), _EPS)
        sq_new = np.maximum(self._periodic_sq(new_pt, others), _EPS)

        old_contrib = np.sum(1.0 / np.prod(sq_old, axis=1))
        new_contrib = np.sum(1.0 / np.prod(sq_new, axis=1))

        new_score = max(current_score - old_contrib + new_contrib, _EPS)
        log_delta = np.log(new_score) - np.log(max(current_score, _EPS))
        return float(log_delta), float(new_score)

    def begin_1d(self, X, i, current_score):
        """
        Precompute per-pair contributions and per-axis squared periodic
        distances for fast 1D coordinate-exchange trials.

        Complexity
        ----------
        O(n · d) once per point.  Each subsequent :meth:`try_1d` call
        then costs O(n), so the total cost for trying all d axes on the
        same point is O(n · d) — the same as a single full
        :meth:`incremental`, but split across d independent trials.
        """
        mask   = np.ones(len(X), dtype=bool);  mask[i] = False
        others = X[mask]                                     # (n-1, d)
        sq     = np.maximum(self._periodic_sq(X[i], others), _EPS)  # (n-1, d)
        # Per-pair full contribution: 1 / prod_v δ_v²
        contribs = 1.0 / np.prod(sq, axis=1)                 # (n-1,)
        return {
            'X'      : X,
            'i'      : i,
            'others' : others,
            'sq'     : sq,
            'contribs': contribs,
            'score'  : current_score,
            'point'  : X[i].copy(),    # current (possibly modified) point i
        }

    def try_1d(self, cache, axis, new_value, space):
        """
        O(n) update: change one coordinate and rescale the cached
        contributions by the ratio of squared periodic distances.

        Mathematics
        -----------
        For each j ≠ i, the pairwise contribution to uMaxPro is

            c_j = 1 / prod_v  δ_v(i, j)²

        Changing only coordinate ``axis`` of point i replaces a single
        factor in the product::

            c_j_new = c_j_old · ( δ_axis_old² / δ_axis_new² )

        and the total score updates as

            S_new = S_old − Σ_j c_j_old + Σ_j c_j_new.
        """
        others_axis = cache['others'][:, axis]
        sq_old_axis = cache['sq'][:, axis]                              # (n-1,)
        # New squared periodic distance on this axis
        d_new       = np.abs(new_value - others_axis)
        sq_new_axis = np.maximum(np.minimum(d_new, 1.0 - d_new) ** 2, _EPS)
        # Rescale per-pair contributions: divide out old δ², multiply new
        new_contribs = cache['contribs'] * (sq_old_axis / sq_new_axis)
        old_sum      = float(np.sum(cache['contribs']))
        new_sum      = float(np.sum(new_contribs))
        new_score    = max(cache['score'] - old_sum + new_sum, _EPS)
        log_delta    = np.log(new_score) - np.log(max(cache['score'], _EPS))

        # Build a trial cache reflecting the proposed change
        new_sq           = cache['sq'].copy()
        new_sq[:, axis]  = sq_new_axis
        new_point        = cache['point'].copy()
        new_point[axis]  = new_value
        new_cache = {
            'X'      : cache['X'],
            'i'      : cache['i'],
            'others' : cache['others'],
            'sq'     : new_sq,
            'contribs': new_contribs,
            'score'  : new_score,
            'point'  : new_point,
        }
        return float(log_delta), float(new_score), new_cache


# ======================================================================
# MaxPro
# ======================================================================

class MaxPro(BaseCriterion):
    """
    MaxPro criterion with Euclidean squared distance.

    Criterion (minimise)::

        MaxPro = Σ_{i<j}  1 / prod_v  (x_iv - x_jv)²

    Notes
    -----
    Suffers from boundary bias: corner points are treated as maximally
    separated even when they are geometrically close under a periodic
    metric.  Prefer :class:`UMaxPro` for general use.

    References
    ----------
    Joseph, V. R., Gul, E. & Ba, S. (2015).
        *Biometrika*, 102(2), 371–380.
    """

    def evaluate(self, X: np.ndarray, space) -> float:
        n = len(X)
        score = 0.0
        for i in range(n - 1):
            sq    = np.maximum((X[i] - X[i + 1:]) ** 2, _EPS)
            score += np.sum(1.0 / np.prod(sq, axis=1))
        return max(score, _EPS)

    def incremental(self, X, i, new_pt, space, current_score):
        mask   = np.ones(len(X), dtype=bool);  mask[i] = False
        others = X[mask]

        sq_old = np.maximum((X[i]   - others) ** 2, _EPS)
        sq_new = np.maximum((new_pt - others) ** 2, _EPS)

        old_contrib = np.sum(1.0 / np.prod(sq_old, axis=1))
        new_contrib = np.sum(1.0 / np.prod(sq_new, axis=1))

        new_score = max(current_score - old_contrib + new_contrib, _EPS)
        log_delta = np.log(new_score) - np.log(max(current_score, _EPS))
        return float(log_delta), float(new_score)

    def begin_1d(self, X, i, current_score):
        """Precompute per-pair contributions and per-axis squared distances."""
        mask   = np.ones(len(X), dtype=bool);  mask[i] = False
        others = X[mask]
        sq     = np.maximum((X[i] - others) ** 2, _EPS)
        contribs = 1.0 / np.prod(sq, axis=1)
        return {
            'X'      : X,
            'i'      : i,
            'others' : others,
            'sq'     : sq,
            'contribs': contribs,
            'score'  : current_score,
            'point'  : X[i].copy(),
        }

    def try_1d(self, cache, axis, new_value, space):
        """
        O(n) rescaling update for MaxPro (Euclidean variant).

        See :meth:`UMaxPro.try_1d` for the algebra; the only difference
        is that the per-axis squared distance is the plain Euclidean
        ``(u - v)²`` rather than the periodic minimum form.
        """
        others_axis = cache['others'][:, axis]
        sq_old_axis = cache['sq'][:, axis]
        sq_new_axis = np.maximum((new_value - others_axis) ** 2, _EPS)
        new_contribs = cache['contribs'] * (sq_old_axis / sq_new_axis)
        old_sum   = float(np.sum(cache['contribs']))
        new_sum   = float(np.sum(new_contribs))
        new_score = max(cache['score'] - old_sum + new_sum, _EPS)
        log_delta = np.log(new_score) - np.log(max(cache['score'], _EPS))

        new_sq          = cache['sq'].copy()
        new_sq[:, axis] = sq_new_axis
        new_point       = cache['point'].copy()
        new_point[axis] = new_value
        new_cache = {
            'X'      : cache['X'],
            'i'      : cache['i'],
            'others' : cache['others'],
            'sq'     : new_sq,
            'contribs': new_contribs,
            'score'  : new_score,
            'point'  : new_point,
        }
        return float(log_delta), float(new_score), new_cache


# ======================================================================
# phi_p
# ======================================================================

class PhiP(BaseCriterion):
    """
    Φ_p (phi-p) space-filling criterion.

    Criterion (minimise)::

        Φ_p(X) = ( Σ_{i<j}  ||x_i − x_j||^{−p} )^{1/p}

    As ``p → ∞`` this converges to the maximin criterion (maximise the
    minimum pairwise distance).  ``p = 15`` is the standard default
    introduced by Morris & Mitchell (1995) as a smooth, differentiable
    maximin proxy that is well behaved under local search.

    Parameters
    ----------
    p : int or float, default 15

    References
    ----------
    Morris, M. D. & Mitchell, T. J. (1995).
        *Journal of Statistical Planning and Inference*, 43, 381–402.
    """

    def __init__(self, p: float = 15) -> None:
        self.p = float(p)

    def evaluate(self, X: np.ndarray, space) -> float:
        n     = len(X)
        total = 0.0
        for i in range(n - 1):
            dists = np.sqrt(np.sum((X[i] - X[i + 1:]) ** 2, axis=1))
            total += np.sum(np.maximum(dists, _EPS) ** (-self.p))
        return max(total, _EPS) ** (1.0 / self.p)

    def incremental(self, X, i, new_pt, space, current_score):
        # Track the inner sum Σ d^(-p) directly: the 1/p exponent makes
        # the score non-additive, but the inner sum is.  We recover the
        # current inner sum from the current Φ_p score, subtract the
        # contributions involving point i (old), add the new ones, then
        # re-apply the 1/p root.
        mask   = np.ones(len(X), dtype=bool);  mask[i] = False
        others = X[mask]

        d_old = np.sqrt(np.sum((X[i]   - others) ** 2, axis=1))
        d_new = np.sqrt(np.sum((new_pt - others) ** 2, axis=1))

        # Inner sum contribution from point i
        old_inner = np.sum(np.maximum(d_old, _EPS) ** (-self.p))
        new_inner = np.sum(np.maximum(d_new, _EPS) ** (-self.p))

        # Recover total inner sum from current phi_p score
        total_inner_old = max(current_score ** self.p, _EPS)
        total_inner_new = max(total_inner_old - old_inner + new_inner, _EPS)

        new_score = total_inner_new ** (1.0 / self.p)
        log_delta = np.log(max(new_score, _EPS)) - np.log(max(current_score, _EPS))
        return float(log_delta), float(new_score)

    def begin_1d(self, X, i, current_score):
        """
        Precompute per-pair squared distances (per axis and total) plus the
        per-pair contribution d^(-p), supporting O(n) updates when only one
        coordinate of point *i* changes.
        """
        mask   = np.ones(len(X), dtype=bool);  mask[i] = False
        others = X[mask]
        sq     = (X[i] - others) ** 2                # (n-1, d) per-axis
        d2_tot = np.sum(sq, axis=1)                  # (n-1,) squared distance
        d_safe = np.maximum(np.sqrt(d2_tot), _EPS)
        contribs = d_safe ** (-self.p)               # (n-1,) per-pair d^(-p)
        # Inner sum for the *full* design (over all pairs, not just those
        # involving i).  Recover from the current phi_p score.
        total_inner = max(current_score ** self.p, _EPS)
        return {
            'X'         : X,
            'i'         : i,
            'others'    : others,
            'sq'        : sq,
            'd2_tot'    : d2_tot,
            'contribs'  : contribs,    # per-pair power contributions
            'i_inner'   : float(np.sum(contribs)),  # contribution of point i
            'total_inner': total_inner,
            'score'     : current_score,
            'point'     : X[i].copy(),
        }

    def try_1d(self, cache, axis, new_value, space):
        """
        O(n) update for Φ_p when only one coordinate of point *i* changes.

        Mathematics
        -----------
        The squared distance to each neighbour updates as

            d²_new(i, j) = d²_old(i, j) − δ_axis_old² + δ_axis_new²,

        and the per-pair contribution becomes

            c_j_new = max(√d²_new, ε)^(−p).

        The total inner sum is then

            S_new = S_old − Σ_j c_j_old + Σ_j c_j_new,

        from which the new Φ_p score is ``S_new^(1/p)``.
        """
        others_axis = cache['others'][:, axis]
        sq_old_axis = cache['sq'][:, axis]
        sq_new_axis = (new_value - others_axis) ** 2
        d2_tot_new  = cache['d2_tot'] - sq_old_axis + sq_new_axis
        d_safe      = np.maximum(np.sqrt(np.maximum(d2_tot_new, 0.0)), _EPS)
        new_contribs = d_safe ** (-self.p)

        new_i_inner = float(np.sum(new_contribs))
        new_total_inner = max(
            cache['total_inner'] - cache['i_inner'] + new_i_inner, _EPS
        )
        new_score = new_total_inner ** (1.0 / self.p)
        log_delta = np.log(max(new_score, _EPS)) - np.log(max(cache['score'], _EPS))

        new_sq           = cache['sq'].copy()
        new_sq[:, axis]  = sq_new_axis
        new_point        = cache['point'].copy()
        new_point[axis]  = new_value
        new_cache = {
            'X'         : cache['X'],
            'i'         : cache['i'],
            'others'    : cache['others'],
            'sq'        : new_sq,
            'd2_tot'    : d2_tot_new,
            'contribs'  : new_contribs,
            'i_inner'   : new_i_inner,
            'total_inner': new_total_inner,
            'score'     : new_score,
            'point'     : new_point,
        }
        return float(log_delta), float(new_score), new_cache

    def __repr__(self) -> str:
        return f"PhiP(p={self.p})"


# ======================================================================
# Centred L2 Discrepancy (CD2)
# ======================================================================

class CD2(BaseCriterion):
    """
    Centred L2 discrepancy.

    Measures deviation of the empirical distribution from the uniform
    distribution on [0, 1]^d.  Lower values indicate better uniformity.

    Criterion (minimise)::

        CD2 = sqrt(
            (13/12)^d
            - (2/n) Σ_i  prod_v (1 + 0.5|x_iv - 0.5| - 0.5(x_iv - 0.5)²)
            + (1/n²) Σ_{i,k} prod_v (1 + 0.5|x_iv - 0.5|
                                        + 0.5|x_kv - 0.5|
                                        - 0.5|x_iv - x_kv|)
        )

    References
    ----------
    Hickernell, F. J. (1998).
        *Mathematics of Computation*, 67(221), 299–322.
    """

    def evaluate(self, X: np.ndarray, space) -> float:
        X    = np.asarray(X, dtype=float)
        n, d = X.shape

        t1 = (13.0 / 12.0) ** d

        ci = 1.0 + 0.5 * np.abs(X - 0.5) - 0.5 * (X - 0.5) ** 2
        t2 = (2.0 / n) * np.prod(ci, axis=1).sum()

        xi  = X[:, np.newaxis, :]        # (n, 1, d)
        xk  = X[np.newaxis, :, :]        # (1, n, d)
        cik = (1.0
               + 0.5 * np.abs(xi - 0.5)
               + 0.5 * np.abs(xk - 0.5)
               - 0.5 * np.abs(xi - xk))
        t3 = np.prod(cik, axis=2).sum() / n ** 2

        return float(np.sqrt(max(t1 - t2 + t3, 0.0)))

    def incremental(self, X, i, new_pt, space, current_score):
        # Build updated design and recompute — CD2 cross terms make a
        # closed-form O(n·d) update possible but complex; full recompute
        # is O(n²·d) and acceptable for typical design sizes (n ≤ 200).
        X_new    = X.copy();  X_new[i] = new_pt
        new_score = self.evaluate(X_new, space)
        log_delta = np.log(max(new_score, _EPS)) - np.log(max(current_score, _EPS))
        return float(log_delta), float(new_score)

    def __repr__(self) -> str:
        return "CD2()"


# ======================================================================
# Stratified L2
# ======================================================================

class StratifiedL2(BaseCriterion):
    """
    Stratified L2-discrepancy (Tian & Xu 2025).

    Measures uniformity by aggregating the L2-norm of the local
    discrepancy over a hierarchical family of stratified regions. For a
    given base ``s`` and maximum depth ``p``, the unit hypercube
    :math:`[0,1)^m` is stratified into ``s^u`` equal intervals on each
    axis for :math:`u = 0, 1, \\ldots, p`, producing
    :math:`(p+1)^m` possible stratifications. The criterion sums the
    squared L2 deviations from uniformity across all of these
    stratifications, weighted by ``w(u)``.

    Closed-form computational expression (Theorem 1, Eq. 13):

    .. math::

        SD(P)^2 = -\\left[\\sum_{i=0}^p w(i)\\, s^{-2i}\\right]^m
                + \\frac{1}{n^2} \\sum_{a,b=1}^n \\prod_{j=1}^m
                  \\left[\\sum_{i=0}^p w(i)\\, s^{-i}\\,
                         \\delta_i(x_{aj}, x_{bj})\\right]

    where
    :math:`\\delta_i(t, z) = \\mathbf{1}\\{\\lfloor s^i t \\rfloor =
    \\lfloor s^i z \\rfloor\\}`.
    Note that :math:`\\delta_0(t, z) \\equiv 1`.

    Parameters
    ----------
    s : int, default 2
        Base number of strata per dimension. Common choices are
        ``s = 2`` (binary partitions) or ``s = 3`` (ternary).
    p : int or None, default None
        Maximum stratification depth. ``None`` selects
        ``floor(log_s(n))`` at evaluation time so that ``s^p ≤ n``
        (Tian & Xu's recommendation to avoid over-stratification).
    weights : {'constant', 'exponential'} or array-like, default 'auto'
        - ``'constant'``: ``w(i) = 1`` for all ``i``, suitable for low
          to moderate dimension.
        - ``'exponential'``: ``w(i) = y^i`` with ``y = 2/(m+1)``, which
          bounds ``SD(P)^2 < e`` regardless of dimension (Corollary 1).
        - ``'auto'`` (default): exponential when ``m ≥ 8`` (curse-of-
          dimensionality territory), constant otherwise.
        - array-like: explicit weights ``[w(0), w(1), …, w(p)]``.

    Notes
    -----
    Computational complexity is :math:`O(n^2 m p)` per evaluation. The
    base :meth:`incremental` falls back to full re-evaluation, which
    keeps SA's cost at :math:`O(n^2 m p)` per move; this matches CD2
    and is acceptable for ``n \\le 200``.

    References
    ----------
    Tian, Y. & Xu, H. (2025). A stratified L2-discrepancy with
        application to space-filling designs.
        *Journal of the Royal Statistical Society, Series B*, 88(2).
    """

    def __init__(
        self,
        s:       int                   = 2,
        p:       Optional[int]         = None,
        weights: Union[str, np.ndarray] = 'auto',
    ) -> None:
        if int(s) < 2:
            raise ValueError(f"s must be ≥ 2; got {s}")
        self.s       = int(s)
        self.p       = (None if p is None else int(p))
        self.weights = weights

    # ── internal helpers ─────────────────────────────────────────────
    def _resolve_p(self, n: int) -> int:
        """Tian & Xu's default: p = floor(log_s n) to cap at s^p ≤ n."""
        if self.p is not None:
            return self.p
        if n <= 1:
            return 1
        return max(1, int(np.floor(np.log(n) / np.log(self.s))))

    def _resolve_weights(self, m: int, p: int) -> np.ndarray:
        """Return the weight vector ``[w(0), w(1), …, w(p)]``."""
        w = self.weights
        if isinstance(w, str):
            kind = w.lower()
            if kind == 'auto':
                kind = 'exponential' if m >= 8 else 'constant'
            if kind == 'constant':
                return np.ones(p + 1, dtype=float)
            if kind == 'exponential':
                y = 2.0 / (m + 1.0)
                return np.array([y ** i for i in range(p + 1)], dtype=float)
            raise ValueError(
                f"weights must be 'constant', 'exponential', 'auto' "
                f"or an array; got {w!r}"
            )
        w = np.asarray(w, dtype=float)
        if w.shape != (p + 1,):
            raise ValueError(
                f"weights array must have shape ({p + 1},); got {w.shape}"
            )
        return w

    @staticmethod
    def _stratum_bins(X: np.ndarray, s: int, i: int) -> np.ndarray:
        """
        Compute the stratum index ``floor(s^i * x)`` clamped to
        ``[0, s^i - 1]`` so that x = 1 is handled correctly.

        Returns an int array of the same shape as X.
        """
        if i == 0:
            return np.zeros_like(X, dtype=np.int64)
        N    = s ** i
        bins = np.floor(N * X).astype(np.int64)
        np.clip(bins, 0, N - 1, out=bins)
        return bins

    # ── evaluate ─────────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, space) -> float:
        X    = np.asarray(X, dtype=float)
        n, m = X.shape
        s    = self.s
        p    = self._resolve_p(n)
        w    = self._resolve_weights(m, p)

        # Term 1: -[Σ_i w(i) · s^(-2i)]^m
        coef_t1 = float(np.sum(w * np.array(
            [s ** (-2 * i) for i in range(p + 1)], dtype=float)))
        term1 = -(coef_t1 ** m)

        # Term 2: (1/n²) Σ_{a,b} ∏_j [Σ_i w(i) · s^(-i) · δ_i(x_aj, x_bj)]
        # Build, for each axis j, the n×n inner-sum matrix K_axis[a,b,j].
        # i = 0 contributes w(0) (δ_0 ≡ 1).
        K_axis = np.full((n, n, m), w[0], dtype=float)
        for i in range(1, p + 1):
            bins   = self._stratum_bins(X, s, i)             # (n, m)
            equal  = (bins[:, None, :] == bins[None, :, :])   # (n, n, m)
            coef   = float(w[i]) * (s ** (-i))
            K_axis = K_axis + coef * equal

        # Product over j and double sum over (a, b)
        K_full = np.prod(K_axis, axis=2)
        term2  = float(K_full.sum()) / (n * n)

        sd2 = term1 + term2
        return float(np.sqrt(max(sd2, 0.0)))

    # ── incremental (full recompute, same as CD2) ────────────────────
    def incremental(self, X, i, new_pt, space, current_score):
        X_new      = X.copy(); X_new[i] = new_pt
        new_score  = self.evaluate(X_new, space)
        log_delta  = np.log(max(new_score, _EPS)) - np.log(max(current_score, _EPS))
        return float(log_delta), float(new_score)

    def __repr__(self) -> str:
        if isinstance(self.weights, str):
            w_str = repr(self.weights)
        else:
            w_str = "array"
        return f"StratifiedL2(s={self.s}, p={self.p}, weights={w_str})"


# ======================================================================
# Factory
# ======================================================================

#: Registry of built-in criterion names → class
_REGISTRY: dict = {
    'umaxpro'   : UMaxPro,
    'maxpro'    : MaxPro,
    'phi_p'     : PhiP,
    'phip'      : PhiP,       # alias
    'cd2'       : CD2,
    'stratified': StratifiedL2,
    'stratified_l2': StratifiedL2,  # alias
}


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
    >>> type(crit)
    <class 'mergen.criteria.UMaxPro'>
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = ', '.join(f"'{k}'" for k in sorted(set(_REGISTRY.keys())
                              - {'phip', 'stratified_l2'}))
        raise ValueError(
            f"\n\033[0;31m[MERGEN ERROR]\033[0m  "
            f"Unknown criterion '{name}'.\n"
            f"  Available: {available}"
        )
    return _REGISTRY[key]()


def list_criteria() -> list:
    """Return a list of available criterion names (canonical, no aliases)."""
    return ['umaxpro', 'maxpro', 'phi_p', 'cd2', 'stratified']