"""
mergen.criteria.stratified
==========================
Stratified L2-Discrepancy criterion (SD).

Reference
---------
Tian, Y. & Xu, H. (2025). A stratified L2-discrepancy with
    application to space-filling designs. *Journal of the Royal
    Statistical Society: Series B*, 88(2).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .base import BaseCriterion, _EPS


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