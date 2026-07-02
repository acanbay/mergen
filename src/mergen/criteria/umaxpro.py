"""
mergen.criteria.umaxpro
=======================
Uniform Maximum Projection (UMaxPro) criterion.

Reference
---------
Vorechovsky, M. & Elias, J. (2026). Uniform Maximum Projection
    Designs for Computer Experiments. *Computers & Structures*.
"""

from __future__ import annotations


import numpy as np

from .base import BaseCriterion, _EPS


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
        mask   = np.ones(len(X), dtype=bool)
        mask[i] = False
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
        mask   = np.ones(len(X), dtype=bool)
        mask[i] = False
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
