"""
mergen.criteria.phi_p
=====================
Morris-Mitchell phi_p criterion (p = 15 by default) — a smooth
proxy for the maximin-distance criterion.

Reference
---------
Morris, M. D. & Mitchell, T. J. (1995). Exploratory designs for
    computational experiments. *Journal of Statistical Planning and
    Inference*, 43(3), 381-402.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import BaseCriterion, _EPS


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