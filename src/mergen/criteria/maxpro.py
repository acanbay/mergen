"""
mergen.criteria.maxpro
======================
Maximum Projection (MaxPro) criterion.

Reference
---------
Joseph, V. R., Gul, E. & Ba, S. (2015). Maximum projection designs
    for computer experiments. *Biometrika*, 102(2), 371-380.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import BaseCriterion, _EPS


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