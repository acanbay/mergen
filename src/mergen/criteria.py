"""
mergen.criteria
===============
Optimisation criteria for the Simulated Annealing engine.

Each criterion exposes two methods:

    evaluate(X, space)
        Full computation on a normalised design matrix.
        Used once per SA restart to obtain the initial score.

    incremental(X, i, new_pt, space, current_score)
        O(n·d) update when point i is swapped for new_pt.
        Returns (log_delta, new_raw_score).
        SA uses log_delta to decide acceptance: negative = improvement.

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

    crit = get_criterion('umaxpro')
    score = crit.evaluate(X_norm, space)
    log_delta, new_score = crit.incremental(X_norm, i, new_pt_norm, space, score)

References
----------
Vorechovsky & Elias (2026), Computers & Structures.         [uMaxPro]
Joseph, Gul & Ba (2015), Biometrika 102(2).                 [MaxPro]
Morris & Mitchell (1995), J. Statist. Plan. Infer. 43.      [phi_p, SA]
Hickernell (1998), Math. Comp. 67.                          [CD2]
Tian & Xu (2025), J. Royal Statist. Soc. B 88(2).           [Stratified L2]
Kirkpatrick, Gelatt & Vecchi (1983), Science 220.           [SA cooling]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

# ── small floor to avoid log(0) and division by zero ─────────────────────
_EPS = 1e-10


# ======================================================================
# Base class
# ======================================================================

class BaseCriterion(ABC):
    """
    Abstract base for SA optimisation criteria.

    Subclasses must implement :meth:`evaluate` and :meth:`incremental`.
    Duck typing is supported — ``isinstance`` checks are not required.
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ======================================================================
# uMaxPro  (default criterion)
# ======================================================================

class UMaxPro(BaseCriterion):
    """
    Uniform MaxPro criterion with periodic distance.

    Replaces Euclidean squared differences with the periodic version::

        Δ_v(i, j) = min(|x_iv - x_jv|,  1 - |x_iv - x_jv|)²

    This eliminates the boundary bias of MaxPro: corner regions are no
    longer artificially far from each other under the toroidal metric,
    so the resulting design is statistically uniform across the entire
    hypercube including its boundaries.

    Criterion (minimise)::

        uMaxPro = Σ_{i<j}  1 / prod_v  Δ_v(i, j)

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


# ======================================================================
# phi_p
# ======================================================================

class PhiP(BaseCriterion):
    """
    Φ_p (phi-p) space-filling criterion.

    Criterion (minimise)::

        Φ_p = ( Σ_{i<j}  ||x_i - x_j||^{-p} )^{1/p}

    As p → ∞ this converges to the maximin criterion (maximise the
    minimum pairwise distance).  p = 15 is the standard default used
    by Morris & Mitchell (1995) as a smooth maximin proxy that remains
    differentiable and SA-friendly.

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
        # Recompute full score after swap — phi_p power sum is non-additive
        # in the same way as MaxPro, but the 1/p exponent means we need the
        # inner sum to update the outer root.  We track the inner sum directly.
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
    Stratified L2 discrepancy.

    Extends CD2 by penalising non-uniformity in all 2^d - 1 projections
    onto lower-dimensional margins, weighted by projection dimension.
    Encourages uniformity not only in the full d-dimensional space but
    also in every lower-dimensional subspace.

    Criterion (minimise): weighted sum of CD2 values across all
    non-empty subsets of parameter dimensions.

    References
    ----------
    Tian, Y. & Xu, H. (2025).
        *Journal of the Royal Statistical Society, Series B*, 88(2).
    """

    def evaluate(self, X: np.ndarray, space) -> float:
        X    = np.asarray(X, dtype=float)
        n, d = X.shape

        total = 0.0
        # Iterate over all 2^d - 1 non-empty subsets of dimensions
        for mask in range(1, 2 ** d):
            dims   = [j for j in range(d) if (mask >> j) & 1]
            k      = len(dims)
            X_sub  = X[:, dims]
            cd2_k  = self._cd2_sub(X_sub, n, k)
            # Weight by subset size: higher-dimensional projections count more
            total += k * cd2_k

        return float(total)

    @staticmethod
    def _cd2_sub(X_sub: np.ndarray, n: int, d: int) -> float:
        """CD2 for a d-dimensional submatrix."""
        t1  = (13.0 / 12.0) ** d
        ci  = 1.0 + 0.5 * np.abs(X_sub - 0.5) - 0.5 * (X_sub - 0.5) ** 2
        t2  = (2.0 / n) * np.prod(ci, axis=1).sum()
        xi  = X_sub[:, np.newaxis, :]
        xk  = X_sub[np.newaxis, :, :]
        cik = (1.0
               + 0.5 * np.abs(xi - 0.5)
               + 0.5 * np.abs(xk - 0.5)
               - 0.5 * np.abs(xi - xk))
        t3  = np.prod(cik, axis=2).sum() / n ** 2
        return float(np.sqrt(max(t1 - t2 + t3, 0.0)))

    def incremental(self, X, i, new_pt, space, current_score):
        # Full recompute — 2^d subset enumeration makes a closed-form
        # incremental update impractical for general d.
        X_new     = X.copy();  X_new[i] = new_pt
        new_score = self.evaluate(X_new, space)
        log_delta = np.log(max(new_score, _EPS)) - np.log(max(current_score, _EPS))
        return float(log_delta), float(new_score)

    def __repr__(self) -> str:
        return "StratifiedL2()"


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

#: Human-readable names for repr / reporting
_NAMES: dict = {
    'umaxpro'      : 'uMaxPro',
    'maxpro'       : 'MaxPro',
    'phi_p'        : 'Φ_p (p=15)',
    'phip'         : 'Φ_p (p=15)',
    'cd2'          : 'CD2',
    'stratified'   : 'Stratified L2',
    'stratified_l2': 'Stratified L2',
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