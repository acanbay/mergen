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

from typing import Tuple

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
        # Build updated design and recompute — CD2 cross terms make a
        # closed-form O(n·d) update possible but complex; full recompute
        # is O(n²·d) and acceptable for typical design sizes (n ≤ 200).
        X_new    = X.copy();  X_new[i] = new_pt
        new_score = self.evaluate(X_new, space)
        log_delta = np.log(max(new_score, _EPS)) - np.log(max(current_score, _EPS))
        return float(log_delta), float(new_score)

    def __repr__(self) -> str:
        return "CD2()"