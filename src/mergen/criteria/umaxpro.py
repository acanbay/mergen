"""
mergen.criteria.umaxpro
=======================
Uniform Maximum Projection (UMaxPro) criterion.

Reference
---------
Vorechovsky, M. & Masek, J. (2026). Uniform Maximum Projection
    Designs for Computer Experiments. *Computers & Structures*.
"""

from __future__ import annotations


import numpy as np

from .maxpro import MaxPro


class UMaxPro(MaxPro):
    """
    Uniform MaxPro criterion with periodic (toroidal) distance.

    Each axis is a circle on which the m feasible levels sit at their
    stratum midpoints, level k at (k + 1/2)/m for equally spaced levels
    (the midpoint placement of Vorechovsky & Masek), so the lowest and
    highest level are 1/m apart rather than one point. Unequally spaced
    levels (a log grid, a list of values) keep their relative spacing,
    with a wrap gap equal to the mean spacing. With the per-axis periodic distance::

        δ_v(i, j) = min(|t_iv − t_jv|,  1 − |t_iv − t_jv|)

    the criterion to minimise is::

        uMaxPro(X) = Σ_{i<j}  1 / prod_v  (δ_v(i, j) + c_v)²

    where c_v = 0 on an axis with at least n feasible levels and
    c_v = 1/m_v on an axis with fewer, the form Joseph, Gul & Ba (2020,
    Eq. 9) give for MaxPro on discrete factors; combining it with the
    periodic distance is Mergen's choice.

    The toroidal wrap removes the boundary bias of plain MaxPro: corner
    regions are no longer artificially far from each other, so the
    resulting design is statistically uniform across the entire
    hypercube including its boundaries. On equally spaced levels the
    criterion is invariant under cyclic shifts of a column.

    References
    ----------
    Vorechovsky, M. & Masek, J. (2026).
        *Computers & Structures*.  [uMaxPro]
    Joseph, V. R., Gul, E. & Ba, S. (2015).
        *Biometrika*, 102(2), 371–380.  [original MaxPro]
    Joseph, V. R., Gul, E. & Ba, S. (2020).
        *Journal of Quality Technology*, 52(4), 343–354.
    """

    def _scale(self, space, d: int) -> np.ndarray:
        """Differences on the midpoint scale: t = (x (m-1) + 1/2) / m is
        affine in the normalised x, so t_a - t_b = (x_a - x_b)(m-1)/m."""
        if space is None:
            return np.ones(d)
        m = space.feasible_levels.astype(float)
        return (m - 1.0) / m

    @staticmethod
    def _dist(a, b, s):
        dd = np.abs(a - b) * s
        return np.minimum(dd, 1.0 - dd)
