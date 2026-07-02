"""
mergen.criteria.qqd
===================
Qualitative-Quantitative Discrepancy (QQD).

Zhang, Yang & Zhou (2021) extend the wrap-around and discrete
discrepancies of Hickernell (1998, 1999) to designs that mix
qualitative and quantitative factors. The two kernels of the
product-form uniformity criterion are chosen so that both factor
types dominate the QQD equally:

- quantitative columns use the wrap-around L2 kernel
  :math:`K_k(t,z) = \\tfrac{3}{2} - |t-z| + |t-z|^{2}`,
  which lives in :math:`[5/4, 3/2]`;
- qualitative (nominal) columns use the discrete-discrepancy kernel
  with :math:`(a, b) = (3/2, 5/4)`, so
  :math:`K_k(t, z) = (3/2)^{\\delta_{tz}} (5/4)^{1-\\delta_{tz}}` and
  the kernel takes only the two boundary values of the quantitative
  range.

For a design :math:`D = (D_1, D_2)` with :math:`p` qualitative and
:math:`q` quantitative columns Theorem 1 of Zhang, Yang & Zhou
(2021) gives the closed form

.. math::

    \\text{QQD}^{2}(D) = C + \\frac{1}{n^{2}} \\sum_{i,j=1}^{n}
        \\left(\\tfrac{5}{4}\\right)^{p}
        \\left(\\tfrac{6}{5}\\right)^{\\delta_{ij}(D_1)}
        \\prod_{k=p+1}^{p+q}
        \\left(\\tfrac{3}{2} - |x_{ik}-x_{jk}| + |x_{ik}-x_{jk}|^{2}\\right),

with :math:`C = -\\prod_{k=1}^{p}\\tfrac{5s_k+1}{4s_k}\\,(4/3)^{q}`
and :math:`\\delta_{ij}(D_1)` the number of qualitative columns on
which rows :math:`i` and :math:`j` share a level (Zhang et al.'s
"coincidence number"). QQD is a strict generalisation of the
wrap-around L2 discrepancy: when :math:`p=0` the qualitative factor
collapses out and the criterion reduces to the WD of the
quantitative sub-design.

Following the convention shared by every other Mergen criterion,
:meth:`evaluate` returns the :math:`\\text{QQD}^{2}(D)` value (the
value that appears throughout the paper); optimisation drivers rank
designs by ``log(score)``, and the discrepancy is strictly positive
on any non-degenerate design.

Ordinal columns are treated as quantitative and enter the WD-kernel
product via their normalised level indices, matching the scoring
approach used by Joseph, Gul & Ba (2019) §3 for MaxProQQ.

References
----------
Zhang, M., Yang, F. & Zhou, Y.-D. (2021). Uniformity criterion for
    designs with both qualitative and quantitative factors.
    *Statistics*, arXiv:2101.02416.
Hickernell, F. J. (1998). A generalized discrepancy and quadrature
    error bound. *Mathematics of Computation*, 67(221), 299-322.
Hickernell, F. J. (1999). Goodness-of-fit statistics, discrepancies
    and robust designs. *Statistics & Probability Letters*, 44(1),
    73-78.
Joseph, V. R., Gul, E. & Ba, S. (2019). Designing computer
    experiments with multiple types of factors: The MaxPro approach.
    *Journal of Quality Technology*, 52(4), 343-354.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from .base import BaseCriterion, _EPS

if TYPE_CHECKING:
    from ..space import ParameterSpace


# ─────────────────────────────────────────────────────────────────────
# Constants (Zhang, Yang & Zhou 2021, Theorem 1)
# ─────────────────────────────────────────────────────────────────────
_A_QUAL = 3.0 / 2.0     # discrete-discrepancy kernel upper bound
_B_QUAL = 5.0 / 4.0     # discrete-discrepancy kernel lower bound
# Match/mismatch factors used in the closed form of Theorem 1
_FACTOR_5_4 = 5.0 / 4.0
_FACTOR_6_5 = 6.0 / 5.0


# ─────────────────────────────────────────────────────────────────────
# Column-type helper
# ─────────────────────────────────────────────────────────────────────
_QUANTITATIVE_TYPES = ('continuous', 'discrete', 'integer', 'ordinal')


def _split_columns(space: "ParameterSpace"):
    """
    Partition the columns of *space* into qualitative (nominal) and
    quantitative (everything else) index lists.

    Ordinal columns are classified as quantitative, in line with the
    scoring approach of Joseph, Gul & Ba (2019) §3 that Mergen also
    applies in :class:`~mergen.criteria.MaxProQQ`.
    """
    types    = [space.param_types[n] for n in space.names]
    n_levels = space.n_levels
    qual_idx  = [k for k, t in enumerate(types) if t == 'nominal']
    quant_idx = [k for k, t in enumerate(types) if t in _QUANTITATIVE_TYPES]
    qual_sizes = [n_levels[k] for k in qual_idx]
    return qual_idx, quant_idx, qual_sizes


def _constant_C(p_qual_sizes, q: int) -> float:
    """
    Closed-form constant :math:`C` from Theorem 1:

    .. math::

        C = -\\prod_{k=1}^{p}\\frac{5s_k+1}{4s_k}\\;\\left(\\frac{4}{3}\\right)^{q}.
    """
    c = 1.0
    for s_k in p_qual_sizes:
        c *= (5.0 * s_k + 1.0) / (4.0 * s_k)
    c *= (4.0 / 3.0) ** q
    return -c


def _wd_kernel_columns(diff_quant: np.ndarray) -> np.ndarray:
    """
    Wrap-around L2 kernel :math:`\\tfrac{3}{2} - |dx| + dx^{2}`,
    applied element-wise to a difference tensor of shape ``(..., q)``.
    """
    abs_d = np.abs(diff_quant)
    return _A_QUAL - abs_d + abs_d * abs_d


# ─────────────────────────────────────────────────────────────────────
# QQD
# ─────────────────────────────────────────────────────────────────────
class QQD(BaseCriterion):
    """
    Qualitative-Quantitative Discrepancy (Zhang, Yang & Zhou 2021).

    Discrepancy-style companion to :class:`~mergen.criteria.MaxProQQ`
    for designs that mix quantitative (continuous, discrete, integer,
    ordinal) and nominal factors. QQD generalises Hickernell's
    wrap-around L2 discrepancy: on all-quantitative spaces it
    reduces to the WD of the quantitative sub-design.

    Attributes
    ----------
    supports_nominal : bool
        ``True`` — the criterion is well-defined on spaces
        containing nominal factors and is one of the two entry
        points (with MaxProQQ) that Mergen exposes for such spaces.

    Notes
    -----
    :meth:`evaluate` returns :math:`\\text{QQD}^{2}(D)` (the value
    reported throughout Zhang, Yang & Zhou 2021). The discrepancy is
    strictly positive on any non-degenerate design; a small floor at
    ``_EPS`` guards ``log(score)`` calls inside the SA / SCE
    acceptance rules against harmless numerical drift below zero.

    Compared with the pair-sum objective of MaxProQQ, QQD is
    naturally computed as an ``n x n`` matrix sum because the paper
    writes it over both symmetric and diagonal terms. The diagonal
    contribution is :math:`n \\cdot (3/2)^{p+q}` — independent of the
    design — so a design swap changes only the off-diagonal
    contributions of the affected row and column, giving an
    :math:`O(n\\,d)` incremental update.

    References
    ----------
    Zhang, M., Yang, F. & Zhou, Y.-D. (2021). *Statistics*,
        arXiv:2101.02416, Theorem 1.
    """

    supports_nominal: bool = True

    # ── evaluate ───────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, space) -> float:
        """
        Compute :math:`\\text{QQD}^{2}(D)` on the normalised design.

        Parameters
        ----------
        X     : np.ndarray, shape (n, d) — normalised coordinates in
                :math:`[0, 1]^d`. Nominal columns carry their level
                indices divided by ``L_h - 1``.
        space : ParameterSpace — supplies the per-column factor type
                and level count.

        Returns
        -------
        float — :math:`\\text{QQD}^{2}(D)`, smaller is better.
        """
        n = X.shape[0]
        if n < 2:
            return _EPS

        qual_idx, quant_idx, qual_sizes = _split_columns(space)
        p, q = len(qual_idx), len(quant_idx)

        # Sub-designs by kind
        X_qual  = X[:, qual_idx]  if qual_idx  else np.empty((n, 0))
        X_quant = X[:, quant_idx] if quant_idx else np.empty((n, 0))

        # Coincidence count delta_{ij}: how many qualitative columns
        # rows i, j agree on. Nominal columns hold float-encoded
        # integer level indices (0/(L-1), 1/(L-1), ...); equality is
        # exact for those values so a straight == suffices.
        if p > 0:
            match = (X_qual[:, None, :] == X_qual[None, :, :])   # (n, n, p)
            delta = np.sum(match, axis=-1)                        # (n, n) int
        else:
            delta = np.zeros((n, n), dtype=int)

        # Qualitative factor: (5/4)^p * (6/5)^delta
        qual_term = (_FACTOR_5_4 ** p) * (_FACTOR_6_5 ** delta)   # (n, n)

        # Quantitative factor: prod_k (3/2 - |dx_k| + dx_k^2)
        if q > 0:
            diff_q      = X_quant[:, None, :] - X_quant[None, :, :]   # (n, n, q)
            kernel_cols = _wd_kernel_columns(diff_q)                  # (n, n, q)
            quant_term  = np.prod(kernel_cols, axis=-1)               # (n, n)
        else:
            quant_term = np.ones((n, n))

        pair_matrix = qual_term * quant_term                          # (n, n)
        total_sum   = float(pair_matrix.sum())

        C       = _constant_C(qual_sizes, q)
        qqd_sq  = C + total_sum / (n * n)

        # Numerical safety: the analytic minimum of QQD^2 is strictly
        # positive (Corollary 1), but rounding on badly conditioned
        # designs can push the finite-precision sum slightly below
        # zero. Clamp to _EPS so the optimiser's log(score) stays
        # well-defined.
        return max(qqd_sq, _EPS)

    # ── incremental (single-point swap) ────────────────────────────
    def incremental(
        self,
        X:             np.ndarray,
        i:             int,
        new_pt:        np.ndarray,
        space,
        current_score: float,
    ) -> Tuple[float, float]:
        """
        O(n*d) update when point *i* is replaced by *new_pt*.

        The QQD closed form sums a symmetric ``n x n`` kernel matrix.
        Replacing row *i* changes only row *i* and column *i*; by
        symmetry those two contributions are equal and can be
        subtracted / added as ``2 * row_i`` after excluding the
        (i, i) diagonal entry, which is design-independent.

        Parameters
        ----------
        X             : np.ndarray, shape (n, d)
        i             : int — row being replaced
        new_pt        : np.ndarray, shape (d,)
        space         : ParameterSpace
        current_score : float — :math:`\\text{QQD}^{2}(D)` before swap

        Returns
        -------
        log_delta : float — ``log(new_score) - log(current_score)``.
        new_score : float — :math:`\\text{QQD}^{2}(D)` after the swap.
        """
        n = X.shape[0]
        if n < 2:
            return 0.0, current_score

        qual_idx, quant_idx, qual_sizes = _split_columns(space)
        p, q = len(qual_idx), len(quant_idx)

        # Reconstruct the raw pair-matrix sum implied by current_score
        # so we can update it locally: current_score = C + sum / n^2.
        C          = _constant_C(qual_sizes, q)
        cur_sum    = (current_score - C) * (n * n)

        # Other rows and their coordinates in each factor group
        others_mask = np.arange(n) != i
        others_all  = X[others_mask]                                 # (n-1, d)

        # Row i contribution to the sum (both row and column of the
        # symmetric matrix): 2 * K(i, j) for j != i.
        def _row_off_diag_sum(pt: np.ndarray) -> float:
            if p > 0:
                match = others_all[:, qual_idx] == pt[qual_idx]      # (n-1, p)
                delta = np.sum(match, axis=-1)                        # (n-1,)
                qual_term = (_FACTOR_5_4 ** p) * (_FACTOR_6_5 ** delta)
            else:
                qual_term = np.full(n - 1, _FACTOR_5_4 ** 0)          # == 1
            if q > 0:
                diff_q     = others_all[:, quant_idx] - pt[quant_idx] # (n-1, q)
                kernel_col = _wd_kernel_columns(diff_q)               # (n-1, q)
                quant_term = np.prod(kernel_col, axis=-1)             # (n-1,)
            else:
                quant_term = np.ones(n - 1)
            return float(np.sum(qual_term * quant_term))

        old_off_diag = _row_off_diag_sum(X[i])
        new_off_diag = _row_off_diag_sum(new_pt)

        # Off-diagonal terms count twice in the full sum (symmetry);
        # the diagonal entry K(i, i) is (3/2)^(p+q) regardless of
        # where row i sits, so it cancels out of the update.
        new_sum   = cur_sum + 2.0 * (new_off_diag - old_off_diag)
        new_score = max(C + new_sum / (n * n), _EPS)

        log_delta = float(np.log(new_score) - np.log(max(current_score, _EPS)))
        return log_delta, new_score
