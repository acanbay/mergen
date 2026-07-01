"""
mergen.criteria.base
====================
Abstract base class and shared constants for space-filling design
criteria. Every criterion in ``mergen.criteria`` subclasses
:class:`BaseCriterion` and lives in its own module.

References
----------
Meyer, R. K. & Nachtsheim, C. J. (1995). The coordinate exchange
    algorithm for constructing exact optimal experimental designs.
    *Technometrics*, 37(1), 60-69.
Kang, L. (2019). Stochastic coordinate-exchange optimal designs
    with complex constraints. *Quality Engineering*, 31(3), 401-416.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

# ── small floor to avoid log(0) and division by zero ─────────────────
_EPS = 1e-10


class BaseCriterion(ABC):
    """
    Abstract base for space-filling design criteria.

    Subclasses implement :meth:`evaluate` and :meth:`incremental`.
    Fast 1D coordinate-exchange updates (:meth:`begin_1d`,
    :meth:`try_1d`) are provided in the base class as a fallback and
    may be overridden by criteria with separable structure for an
    O(n) inner loop.
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

    def begin_1d(
        self,
        X:             np.ndarray,
        i:             int,
        current_score: float,
    ):
        """
        Initialise a per-point cache for fast 1D coordinate-exchange updates.

        Called once before trying multiple coordinate changes on the same
        point *i*.  The returned cache object is then passed to
        :meth:`try_1d` for each candidate value.

        Subclasses with separable structures (UMaxPro, MaxPro, PhiP) override
        this to precompute per-pair contributions, enabling O(n) per-try
        updates instead of O(n·d).

        Default implementation stores ``(X, i, current_score)`` for the
        base :meth:`try_1d` fallback, which builds the full new point
        and delegates to :meth:`incremental`.

        Parameters
        ----------
        X             : np.ndarray, shape (n, d) — current normalised design
        i             : int — index of the point to be modified
        current_score : float — raw criterion score before any change

        Returns
        -------
        cache : object
            Opaque structure used by :meth:`try_1d`.  Subclass-specific.
        """
        return {'X': X, 'i': i, 'score': current_score}

    def try_1d(
        self,
        cache,
        axis:      int,
        new_value: float,
        space,
    ) -> Tuple[float, float, object]:
        """
        Try changing one coordinate of the cached point and return the delta.

        For separable criteria, this runs in O(n).  For non-separable
        criteria (CD2, StratifiedL2), the base implementation falls back
        to the full O(n·d) :meth:`incremental` evaluation.

        The returned cache reflects the *trial* state and may be reused for
        further trials on the same point.  It does not modify the original
        design array.

        Parameters
        ----------
        cache     : object — returned by :meth:`begin_1d`
        axis      : int — coordinate index being changed (0 ≤ axis < d)
        new_value : float — new normalised value at coordinate ``axis``
        space     : ParameterSpace

        Returns
        -------
        log_delta : float — log(new_score) − log(current_score)
        new_score : float — raw criterion score after the change
        new_cache : object — cache updated to reflect the trial change
        """
        X  = cache['X']
        i  = cache['i']
        s  = cache['score']
        new_pt       = X[i].copy()
        new_pt[axis] = new_value
        log_delta, new_score = self.incremental(X, i, new_pt, space, s)
        new_cache = {'X': X, 'i': i, 'score': new_score}
        return log_delta, new_score, new_cache

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"