"""
mergen.criteria.maxproqq
========================
Maximum Projection criterion for mixed quantitative and qualitative
factors (MaxProQQ).

Joseph, Gul & Ba (2020) extend the MaxPro criterion of Joseph, Gul &
Ba (2015) so that a single space-filling objective covers four
factor types:

- **continuous**: normalised to :math:`[0, 1]` with :math:`n` distinct
  levels; contribution :math:`(x_{il}-x_{jl})^{2}`.
- **discrete numeric** and **integer**: :math:`m_k` levels;
  contribution :math:`(|u_{ik}-u_{jk}| + 1/m_k)^{2}` where the
  :math:`1/m_k` term prevents the denominator from vanishing when two
  points fall on the same level.
- **ordinal**: scored to a numeric grid (Joseph 2019 §3) and treated
  as discrete numeric with :math:`m_k = L_k` levels.
- **nominal**: contribution
  :math:`(\\mathbb{1}(v_{ih}\\neq v_{jh}) + 1/L_h)^{2}` where
  :math:`\\mathbb{1}(\\cdot)` is the level-mismatch indicator and
  :math:`L_h` is the number of levels of the :math:`h`-th nominal
  factor.

For a design :math:`D=\\{x_1,\\dots,x_n\\}` with :math:`p_1` continuous,
:math:`p_2` discrete-numeric/ordinal, and :math:`p_3` nominal
factors, MaxProQQ minimises

.. math::

    \\psi(D) = \\left\\{
      \\frac{1}{\\binom{n}{2}}\\sum_{i<j}
        \\frac{1}
             {\\prod_{l=1}^{p_1}(x_{il}-x_{jl})^{2}\\;
              \\prod_{k=1}^{p_2}\\left(|u_{ik}-u_{jk}|+\\tfrac{1}{m_k}\\right)^{2}\\;
              \\prod_{h=1}^{p_3}\\left(\\mathbb{1}(v_{ih}\\neq v_{jh})+\\tfrac{1}{L_h}\\right)^{2}}
    \\right\\}^{1/(p_1+p_2+p_3)}.

When all factors are continuous the criterion reduces exactly to
the original MaxPro objective (Joseph, Gul & Ba 2015), so MaxProQQ
is a strict generalisation. The exponent :math:`1/(p_1+p_2+p_3)`
matches the dimensional homogeneity of MaxPro.

References
----------
Joseph, V. R., Gul, E. & Ba, S. (2019). Designing computer
    experiments with multiple types of factors: The MaxPro approach.
    *Journal of Quality Technology*, 52(4), 343-354.
Joseph, V. R., Gul, E. & Ba, S. (2015). Maximum projection designs
    for computer experiments. *Biometrika*, 102(2), 371-380.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from .base import BaseCriterion, _EPS

if TYPE_CHECKING:
    from ..space import ParameterSpace


# ─────────────────────────────────────────────────────────────────────
# Per-column contribution table
# ─────────────────────────────────────────────────────────────────────
#
# For each column of the normalised design we need to know two things
# to compute (and later update) the criterion:
#
#   kind  : 'continuous' | 'quantitative_discrete' | 'nominal'
#   size  : number of levels of that column (only used for
#           'quantitative_discrete' and 'nominal', to build the
#           1/m and 1/L regularisers).
#
# The Sampler hands us a ParameterSpace whose parameter types are one
# of  {'continuous', 'discrete', 'integer', 'ordinal', 'nominal'}.
# Following Joseph, Gul & Ba (2019) §3 we score ordinal factors as
# discrete numeric ones and merge all non-continuous non-nominal
# columns into the single 'quantitative_discrete' bucket.

_QUANTITATIVE_DISCRETE = ('discrete', 'integer', 'ordinal')


def _column_kind(param_type: str) -> str:
    """Map a ParameterSpace type string to the MaxProQQ column kind."""
    if param_type == 'continuous':
        return 'continuous'
    if param_type == 'nominal':
        return 'nominal'
    if param_type in _QUANTITATIVE_DISCRETE:
        return 'quantitative_discrete'
    raise ValueError(
        f"MaxProQQ: unknown parameter type {param_type!r}."
    )


def _column_contribution(
    diff: np.ndarray,
    kind: str,
    size: int,
) -> np.ndarray:
    """
    Compute the per-column pairwise contribution to the MaxProQQ
    denominator.

    Parameters
    ----------
    diff : np.ndarray
        Pairwise differences on the normalised scale, shape
        broadcastable to any ``(...,)``.
    kind : str
        One of ``'continuous'``, ``'quantitative_discrete'``,
        ``'nominal'``.
    size : int
        Number of levels of the column (used only when *kind* is
        ``'quantitative_discrete'`` or ``'nominal'``).

    Returns
    -------
    np.ndarray
        The squared contribution used in the product term.

    Notes
    -----
    Continuous columns are floored at ``_EPS`` following the same
    convention as :class:`~mergen.criteria.MaxPro`: when two rows
    collide on a continuous column the theoretical contribution is
    zero, which would make the pairwise denominator vanish. The
    floor keeps the criterion finite, which lets SA / SCE keep
    exploring past the collision instead of stalling on ``+inf``
    scores.
    """
    if kind == 'continuous':
        return np.maximum(diff * diff, _EPS)
    if kind == 'quantitative_discrete':
        return (np.abs(diff) + 1.0 / size) ** 2
    # kind == 'nominal'
    indicator = (~np.isclose(diff, 0.0)).astype(float)
    return (indicator + 1.0 / size) ** 2


# ─────────────────────────────────────────────────────────────────────
# MaxProQQ
# ─────────────────────────────────────────────────────────────────────
class MaxProQQ(BaseCriterion):
    """
    Maximum Projection criterion for quantitative and qualitative
    factors (Joseph, Gul & Ba 2019).

    Extends :class:`~mergen.criteria.MaxPro` to spaces that mix
    continuous, discrete-numeric / integer / ordinal, and nominal
    factors. When the space contains only continuous columns, the
    criterion reduces exactly to plain MaxPro.

    Parameters
    ----------
    delta : float, default 0.0
        Additive floor for the pairwise denominator. Guards against
        division by zero when two rows share the same value on a
        continuous column (which does not happen for a valid LHD but
        can arise in constrained or coarse-grid searches). Joseph et
        al. (2019) do not use a floor; the default of ``0.0``
        reproduces the paper exactly. Set to a small positive value
        (e.g. ``1e-12``) for extra numerical safety.

    Attributes
    ----------
    supports_nominal : bool
        ``True`` — the criterion is defined on spaces containing
        nominal factors.

    Notes
    -----
    The design matrix passed to :meth:`evaluate` and
    :meth:`incremental` is the same normalised ``X`` that every other
    Mergen criterion sees; nominal columns arrive as their integer
    level indices divided by ``L_h - 1`` (so distinct levels stay
    distinct after normalisation). The indicator :math:`\\mathbb{1}(v_i \\neq v_j)`
    is therefore evaluated by an ``np.isclose`` comparison rather than
    by looking at the raw labels.

    References
    ----------
    Joseph, V. R., Gul, E. & Ba, S. (2020).
        *Journal of Quality Technology*, 52(4), 343-354.
    """

    supports_nominal: bool = True

    def __init__(self, delta: float = 0.0) -> None:
        if delta < 0.0:
            raise ValueError(f"delta must be >= 0; got {delta}.")
        self.delta = float(delta)

    # ── internal helper ────────────────────────────────────────────
    def _column_info(
        self,
        space: "ParameterSpace",
    ) -> Tuple[List[str], List[int]]:
        """Return per-column ``(kind, level_count)`` lists."""
        kinds  = [_column_kind(space.param_types[n]) for n in space.names]
        sizes  = list(space.n_levels)
        return kinds, sizes

    # ── evaluate ───────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, space) -> float:
        """
        Compute the MaxProQQ score on the normalised design *X*.

        To stay consistent with Mergen's :class:`MaxPro` — which
        reports the inner pairwise sum rather than the outer
        :math:`(\\cdot)^{1/d}` root — this method also returns the raw
        sum

        .. math::

            \\text{score}(D) = \\sum_{i<j}
                \\frac{1}{\\prod_{k} c_k(x_{ik}, x_{jk}) + \\delta},

        where the per-column term :math:`c_k` follows Joseph, Gul &
        Ba (2019) Eq. 9. The Mergen optimisers rank designs by
        ``log(score)``, so the monotone :math:`1/d` power is
        redundant and would only shrink dynamic range for the SA/SCE
        acceptance rules. Users who want the paper's :math:`\\psi(D)`
        can recover it as
        ``(score / (n*(n-1)/2)) ** (1/d)``.

        Parameters
        ----------
        X     : np.ndarray, shape (n, d) — normalised coordinates in
                :math:`[0, 1]^d`.
        space : ParameterSpace — supplies the per-column factor type
                and level count.

        Returns
        -------
        float — the raw pairwise score (smaller is better).
        """
        n, d = X.shape
        if n < 2:
            return _EPS

        kinds, sizes = self._column_info(space)

        # Pairwise differences: (n, n, d)
        diff = X[:, None, :] - X[None, :, :]

        # Per-column contribution: same shape
        contrib = np.empty_like(diff)
        for k in range(d):
            contrib[..., k] = _column_contribution(
                diff[..., k], kinds[k], sizes[k],
            )

        # Product across columns, then upper triangle
        prod = np.prod(contrib, axis=-1)                     # (n, n)
        iu, ju = np.triu_indices(n, k=1)
        pair_prod = prod[iu, ju]

        # Continuous columns are floored at _EPS in
        # _column_contribution, so pair_prod is strictly positive
        # even when two rows collide on every column. The optional
        # delta adds an extra safety cushion.
        denom = pair_prod + self.delta
        return float(np.sum(1.0 / denom))

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

        The MaxProQQ objective is a sum over pairs, so replacing a
        single row changes only the ``n-1`` pairs that involve that
        row. We subtract those pairs' old contributions and add the
        new ones without recomputing the full :math:`O(n^2)` sum.

        Parameters
        ----------
        X             : np.ndarray, shape (n, d)
        i             : int — row being replaced
        new_pt        : np.ndarray, shape (d,)
        space         : ParameterSpace
        current_score : float — raw pairwise sum before the swap

        Returns
        -------
        log_delta : float
            ``log(new_score) - log(current_score)``; negative values
            mark an improvement.
        new_score : float
            Raw pairwise sum after the swap.
        """
        n, d = X.shape
        if n < 2:
            return 0.0, current_score
        kinds, sizes = self._column_info(space)

        # Pair contributions involving row i (old and prospective).
        others_mask       = np.arange(n) != i
        others            = X[others_mask]                    # (n-1, d)
        old_diff          = others - X[i]                     # (n-1, d)
        new_diff          = others - new_pt
        old_contrib_cols  = np.empty_like(old_diff)
        new_contrib_cols  = np.empty_like(new_diff)
        for k in range(d):
            old_contrib_cols[:, k] = _column_contribution(
                old_diff[:, k], kinds[k], sizes[k],
            )
            new_contrib_cols[:, k] = _column_contribution(
                new_diff[:, k], kinds[k], sizes[k],
            )
        old_prod = np.prod(old_contrib_cols, axis=-1)
        new_prod = np.prod(new_contrib_cols, axis=-1)

        # Continuous contributions are floored at _EPS in
        # _column_contribution, so both denom vectors are strictly
        # positive. Optional delta adds an extra safety cushion.
        old_denom = old_prod + self.delta
        new_denom = new_prod + self.delta

        old_contrib = float(np.sum(1.0 / old_denom))
        new_contrib = float(np.sum(1.0 / new_denom))

        new_score = current_score - old_contrib + new_contrib
        if new_score <= 0.0:
            new_score = _EPS

        log_delta = float(np.log(new_score) - np.log(max(current_score, _EPS)))
        return log_delta, new_score
