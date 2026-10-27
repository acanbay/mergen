"""
mergen.criteria.cd2
===================
Centered L2-Discrepancy (CD2) criterion.

Reference
---------
Hickernell, F. J. (1998). A generalized discrepancy and quadrature
    error bound. *Mathematics of Computation*, 67(221), 299-322.
"""

from __future__ import annotations


import numpy as np

from .base import BaseCriterion, _EPS


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
        # Moving point i changes its own term in the single sum and row
        # and column i of the double sum, so the squared discrepancy
        # changes by an O(n·d) amount.
        X      = np.asarray(X, dtype=float)
        n      = len(X)
        old_pt = X[i]
        new_pt = np.asarray(new_pt, dtype=float)
        a      = np.abs(X - 0.5)
        a_old  = np.abs(old_pt - 0.5)
        a_new  = np.abs(new_pt - 0.5)

        def single(ap):
            return np.prod(1.0 + 0.5 * ap - 0.5 * ap ** 2)

        def pairs(p, ap):
            # Products against every row, and against the point itself.
            row  = np.prod(1.0 + 0.5 * ap + 0.5 * a - 0.5 * np.abs(p - X),
                           axis=1)
            self_term = np.prod(1.0 + ap)
            return row, self_term

        row_old, self_old = pairs(old_pt, a_old)
        row_new, self_new = pairs(new_pt, a_new)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        delta = (-(2.0 / n) * (single(a_new) - single(a_old))
                 + (2.0 * np.sum(row_new[mask] - row_old[mask])
                    + self_new - self_old) / n ** 2)
        new_sq    = max(current_score ** 2 + delta, 0.0)
        new_score = float(np.sqrt(new_sq))
        log_delta = np.log(max(new_score, _EPS)) - np.log(max(current_score, _EPS))
        return float(log_delta), float(new_score)

    def __repr__(self) -> str:
        return "CD2()"
