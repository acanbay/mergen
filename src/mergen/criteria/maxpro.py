"""
mergen.criteria.maxpro
======================
Maximum Projection (MaxPro) criterion.

Reference
---------
Joseph, V. R., Gul, E. & Ba, S. (2015). Maximum projection designs
    for computer experiments. *Biometrika*, 102(2), 371-380.
Joseph, V. R., Gul, E. & Ba, S. (2020). Designing computer
    experiments with multiple types of factors: the MaxPro approach.
    *Journal of Quality Technology*, 52(4), 343-354.
"""

from __future__ import annotations


import numpy as np

from .base import BaseCriterion, _EPS, _PeakTracking, _level_offsets


class MaxPro(_PeakTracking, BaseCriterion):
    """
    MaxPro criterion with Euclidean squared distance.

    Criterion (minimise)::

        MaxPro = Σ_{i<j}  1 / prod_v  (|x_iv - x_jv| + c_v)²

    with c_v = 0 on an axis that has at least n feasible levels, which
    is the published criterion (Joseph, Gul & Ba 2015). On an axis with
    m_v < n levels, repeated levels are forced and the published
    product is infinite; there c_v = 1/m_v, the form Joseph, Gul & Ba
    (2020, Eq. 9) give for discrete factors.

    Notes
    -----
    Suffers from boundary bias: corner points are treated as maximally
    separated even when they are geometrically close under a periodic
    metric.  Prefer :class:`UMaxPro` for general use.

    References
    ----------
    Joseph, V. R., Gul, E. & Ba, S. (2015).
        *Biometrika*, 102(2), 371–380.
    Joseph, V. R., Gul, E. & Ba, S. (2020).
        *Journal of Quality Technology*, 52(4), 343–354.
    """

    # ── Per-axis term; UMaxPro overrides the distance and the scale ──

    def _scale(self, space, d: int) -> np.ndarray:
        """Per-axis factor applied to coordinate differences."""
        return np.ones(d)

    @staticmethod
    def _dist(a, b, s):
        """Per-axis distance of coordinates a and b, scaled by s."""
        return np.abs(a - b) * s

    def _terms(self, space, n: int, d: int):
        """
        The per-axis scale and level offsets for an n-row design,
        kept for the last (space pool, n, d) so the hot update path
        does not recompute them.
        """
        pool = None if space is None else space.candidate_pool
        cached = getattr(self, '_terms_cache', None)
        if (cached is None or cached[0] is not pool
                or cached[1] != (n, d)):
            cached = (pool, (n, d), self._scale(space, d),
                      _level_offsets(space, n, d))
            self._terms_cache = cached
        return cached[2], cached[3]

    def _sq(self, a, b, s, c):
        """Per-axis squared term (distance + c)², floored at _EPS."""
        return np.maximum((self._dist(a, b, s) + c) ** 2, _EPS)

    # ── Criterion ────────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, space) -> float:
        X    = np.asarray(X, dtype=float)
        n, d = X.shape
        s, c = self._terms(space, n, d)
        iu, ju = np.triu_indices(n, k=1)
        sq    = self._sq(X[iu], X[ju], s, c)          # all pairs at once
        score = max(float(np.sum(1.0 / np.prod(sq, axis=1))), _EPS)
        self._exact(score)
        return score

    def incremental(self, X, i, new_pt, space, current_score):
        X      = np.asarray(X, dtype=float)
        n, d   = X.shape
        s, c   = self._terms(space, n, d)
        mask   = np.ones(n, dtype=bool)
        mask[i] = False
        others = X[mask]

        old_contrib = np.sum(1.0 / np.prod(self._sq(X[i], others, s, c),
                                           axis=1))
        new_contrib = np.sum(1.0 / np.prod(
            self._sq(np.asarray(new_pt, dtype=float), others, s, c), axis=1))

        new_score = current_score - old_contrib + new_contrib
        exact = None
        if self._too_far_below(current_score, new_score, current_score):
            X_new = X.copy()
            X_new[i] = new_pt
            new_score = exact = self._trial_total(
                lambda: self.evaluate(X_new, space))
        new_score = max(new_score, _EPS)
        self._settle(new_score, exact)
        log_delta = np.log(new_score) - np.log(max(current_score, _EPS))
        return float(log_delta), float(new_score)

    def begin_1d(self, X, i, current_score, space=None):
        """
        Precompute per-pair contributions and per-axis squared terms.
        The scale and offsets are fixed here, so :meth:`try_1d` stays
        consistent with this cache whatever space it is given.
        """
        X      = np.asarray(X, dtype=float)
        n, d   = X.shape
        s, c   = self._terms(space, n, d)
        mask   = np.ones(n, dtype=bool)
        mask[i] = False
        others = X[mask]
        sq     = self._sq(X[i], others, s, c)
        return {
            'X'       : X,
            'i'       : i,
            'point'   : X[i].copy(),
            'space'   : space,
            'others'  : others,
            's'       : s,
            'c'       : c,
            'sq'      : sq,
            'contribs': 1.0 / np.prod(sq, axis=1),
            'score'   : current_score,
        }

    def try_1d(self, cache, axis, new_value, space):
        """
        O(n) update when one coordinate of point *i* changes: each pair's
        contribution is rescaled by the ratio of the old to the new
        squared term on that axis.

        For each j ≠ i the pairwise contribution is
        c_j = 1 / prod_v term_v(i, j); changing coordinate ``axis`` of
        point i replaces one factor, so
        c_j_new = c_j_old · term_axis_old / term_axis_new and
        S_new = S_old − Σ_j c_j_old + Σ_j c_j_new.
        """
        sq_old_axis = cache['sq'][:, axis]
        sq_new_axis = self._sq(float(new_value), cache['others'][:, axis],
                               cache['s'][axis], cache['c'][axis])
        new_contribs = cache['contribs'] * (sq_old_axis / sq_new_axis)
        old_sum   = float(np.sum(cache['contribs']))
        new_sum   = float(np.sum(new_contribs))
        new_point = cache['point'].copy()
        new_point[axis] = new_value
        new_score = cache['score'] - old_sum + new_sum
        exact = None
        if self._too_far_below(cache['score'], new_score, cache['score']):
            X_new = cache['X'].copy()
            X_new[cache['i']] = new_point
            new_score = exact = self._trial_total(
                lambda: self.evaluate(X_new, cache['space']))
        new_score = max(new_score, _EPS)
        self._settle(new_score, exact)
        log_delta = np.log(new_score) - np.log(max(cache['score'], _EPS))

        new_sq          = cache['sq'].copy()
        new_sq[:, axis] = sq_new_axis
        new_cache = dict(cache, sq=new_sq, contribs=new_contribs,
                         score=new_score, point=new_point)
        return float(log_delta), float(new_score), new_cache
