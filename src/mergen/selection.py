"""
mergen.selection
================
Choosing one design among several candidates by quality metrics.

:meth:`Sampler.compare` and :meth:`Sampler.run` (with ``n_repeats > 1``
and a ``priority``) both end with the same question: given a few
candidate designs and their raw metric values, which one is delivered?
This module answers it with one of two documented rules.

``'lexicographic'`` (default of :func:`select`)
    The priority metrics are ranked by importance. Candidates are
    filtered metric by metric: at each level only those within the
    tolerance of the best surviving value are kept, and the next metric
    decides among them (hierarchical optimisation with tolerances;
    Waltz 1967; Ehrgott 2005, ch. 5). Each level compares against the
    best survivor, never pairwise, so tolerances cannot chain.

``'pareto'`` (default of :meth:`Sampler.run` and :meth:`Sampler.compare`)
    The priority metrics are treated as unordered, competing objectives
    (Lu, Anderson-Cook & Robinson 2011). The non-dominated candidates
    are kept; each metric is min-max scaled over these front members,
    metrics whose spread on the front lies within the tolerance are
    dropped as uninformative, and the front member closest to the
    Utopia point wins. Scaling over the front means a dominated
    candidate can never change the winner. Ties on Utopia distance (for
    example, two front members that each win one of two metrics) are
    resolved in priority order; :attr:`Selection.decided_by` reports
    when that happens.

Two values ``a`` and ``b`` of a metric count as equal when
``|a - b| <= tol + rtol * max(|a|, |b|)`` (the semantics of
:func:`math.isclose`). The default ``tol=1e-9`` with ``rtol=0`` only
absorbs floating-point noise; metrics are computed on normalised
coordinates, so a larger ``tol`` reads as a fraction of the parameter
range (an indifference threshold chosen by the user). A relative
tolerance is ill-posed for metrics whose ideal value is zero, such as
``max_abs_correlation``.

Candidates whose priority metrics are NaN or infinite are never
selected.

References
----------
Waltz, F. M. (1967). An engineering approach: hierarchical optimization
    criteria. *IEEE Transactions on Automatic Control*, 12(2), 179–180.

Ehrgott, M. (2005). *Multicriteria Optimization*, 2nd ed. Springer.

Lu, L., Anderson-Cook, C. M. & Robinson, T. J. (2011). Optimization of
    designed experiments based on multiple criteria utilizing a Pareto
    frontier. *Technometrics*, 53(4), 353–365.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SELECTION_RULES = ('lexicographic', 'pareto')

Tolerance = Union[float, Mapping[str, float]]


@dataclass(frozen=True)
class Selection:
    """
    Outcome of :func:`select`.

    Attributes
    ----------
    winner : int
        Index of the selected candidate.
    ties : tuple of int
        Candidates the rule could not tell apart from the winner on the
        metrics (the winner included): for ``'lexicographic'`` those
        tied on every priority metric, for ``'pareto'`` the front
        members tied on Utopia distance.
    decided_by : str
        What settled the choice: ``'metrics'`` (no tie),
        ``'priority order'`` (a Pareto tie broken by the order of the
        priority metrics) or ``'tiebreak'`` (the ``tiebreak`` key or,
        failing that, the candidate order).
    front : tuple of int
        Candidates not dominated by any other on the priority metrics,
        within the tolerance. The ``'pareto'`` rule picks its winner
        from this set. Under ``'lexicographic'`` it is informational:
        tolerant equality is not transitive, so with a large ``tol`` the
        lexicographic winner can lie outside it.
    excluded : tuple of int
        Candidates left out because a priority metric is NaN or
        infinite.
    rule : str
        The rule that was applied.
    """

    winner:     int
    ties:       Tuple[int, ...]
    front:      Tuple[int, ...]
    excluded:   Tuple[int, ...]
    rule:       str
    decided_by: str = 'metrics'


def resolve_tolerance(value: Tolerance, names: Sequence[str],
                      what: str) -> np.ndarray:
    """
    Expand ``tol`` or ``rtol`` into one non-negative value per metric.

    A float applies to every metric. A mapping sets values per metric;
    metrics it does not name keep the default (``1e-9`` for ``tol``,
    ``0`` for ``rtol``).
    """
    default = 1e-9 if what == 'tol' else 0.0
    bad = ValueError(f"{what} must be a non-negative number or a mapping "
                     f"{{metric: number}}; got {value!r}")
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise ValueError(
                f"{what} names metrics that are not in priority: {unknown}. "
                f"Priority: {list(names)}")
        items = [value.get(n, default) for n in names]
    else:
        items = [value] * len(names)
    if any(isinstance(v, (bool, np.bool_, str, bytes)) or np.ndim(v) != 0
           for v in items):
        raise bad
    try:
        out = np.array([float(v) for v in items], dtype=float)
    except (TypeError, ValueError):
        raise bad from None
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise ValueError(f"{what} must be finite and non-negative; "
                         f"got {value!r}")
    return out


def _close(a, b, tol, rtol):
    """``|a - b| <= tol + rtol * max(|a|, |b|)``, element-wise."""
    return np.abs(a - b) <= tol + rtol * np.maximum(np.abs(a), np.abs(b))


def _lexicographic_survivors(V, candidates, tol, rtol):
    """
    Filter ``candidates`` metric by metric (columns of ``V`` in priority
    order, oriented so lower is better), keeping at each level only
    those within tolerance of the best surviving value.
    """
    surv = list(candidates)
    for j in range(V.shape[1]):
        if len(surv) == 1:
            break
        col  = V[surv, j]
        best = col.min()
        keep = (col <= best) | _close(col, best, tol[j], rtol[j])
        surv = [c for c, k in zip(surv, keep) if k]
    return surv


def _pareto_front(V, candidates, tol, rtol):
    """
    Candidates not dominated by another within the tolerance. Candidate
    ``a`` dominates ``b`` when it is no worse on every metric beyond the
    tolerance and better than the tolerance on at least one. Tolerant
    dominance is not transitive and can, in rare cases, dominate every
    candidate; the exact front (zero tolerance) is used then.
    """
    def front(t, r):
        out = []
        for b in candidates:
            dominated = False
            for a in candidates:
                if a == b:
                    continue
                eq     = _close(V[a], V[b], t, r)
                better = (V[a] < V[b]) & ~eq
                worse  = (V[a] > V[b]) & ~eq
                if better.any() and not worse.any():
                    dominated = True
                    break
            if not dominated:
                out.append(b)
        return out

    f = front(tol, rtol)
    if not f:
        f = front(np.zeros_like(tol), np.zeros_like(rtol))
    return f


def select(
    values:        np.ndarray,
    higher_better: Sequence[bool],
    rule:          str = 'lexicographic',
    tol:           Optional[np.ndarray] = None,
    rtol:          Optional[np.ndarray] = None,
    tiebreak:      Optional[Sequence[float]] = None,
) -> Selection:
    """
    Select one candidate from raw metric values.

    Parameters
    ----------
    values : array, shape (n_candidates, n_metrics)
        Raw metric values, columns in priority order.
    higher_better : sequence of bool, length n_metrics
        Direction of each metric.
    rule : ``'lexicographic'`` or ``'pareto'``
    tol, rtol : array, shape (n_metrics,), optional
        Absolute and relative tolerances per metric (see
        :func:`resolve_tolerance`). Default ``1e-9`` and ``0``.
    tiebreak : sequence of float, optional
        Secondary key used when the rule leaves several candidates
        (lower wins), e.g. the criterion score. Remaining ties go to
        the lowest candidate index.

    Returns
    -------
    Selection
    """
    if rule not in SELECTION_RULES:
        raise ValueError(f"Unknown selection rule {rule!r}. "
                         f"Available: {list(SELECTION_RULES)}")
    V = np.asarray(values, dtype=float)
    if V.ndim != 2 or V.shape[0] == 0:
        raise ValueError("values must be a non-empty 2-D array")
    n, m = V.shape
    if len(higher_better) != m:
        raise ValueError("higher_better must have one entry per metric")
    tol  = np.full(m, 1e-9) if tol  is None else np.asarray(tol, float)
    rtol = np.zeros(m)      if rtol is None else np.asarray(rtol, float)

    # Orient every metric so that lower is better.
    sign = np.where(np.asarray(higher_better, dtype=bool), -1.0, 1.0)
    V = V * sign

    excluded = tuple(i for i in range(n) if not np.isfinite(V[i]).all())
    valid    = [i for i in range(n) if i not in excluded]
    if not valid:
        raise ValueError("every candidate has a NaN or infinite priority "
                         "metric; nothing can be selected")

    front = _pareto_front(V, valid, tol, rtol)

    if rule == 'lexicographic':
        ties = _lexicographic_survivors(V, valid, tol, rtol)
        rest = ties
        decided_by = 'metrics' if len(ties) == 1 else 'tiebreak'
    else:
        # Scale over the front, so dominated candidates cannot move the
        # Utopia point or the ranges.
        lo, hi = V[list(front)].min(axis=0), V[list(front)].max(axis=0)
        span   = hi - lo
        inform = ~_close(hi, lo, tol, rtol)
        if inform.any():
            Z    = (V[:, inform] - lo[inform]) / span[inform]
            dist = np.linalg.norm(Z, axis=1)
            dmin = min(dist[i] for i in front)
            near = [i for i in front if dist[i] - dmin <= 1e-12]
        else:
            near = list(front)
        ties = near
        rest = _lexicographic_survivors(V, near, tol, rtol)
        if len(near) == 1:
            decided_by = 'metrics'
        elif len(rest) == 1:
            decided_by = 'priority order'
        else:
            decided_by = 'tiebreak'

    if tiebreak is not None:
        tb = np.asarray(tiebreak, dtype=float)
        winner = min(rest, key=lambda i: (tb[i], i))
    else:
        winner = min(rest)

    return Selection(winner=int(winner), ties=tuple(int(i) for i in ties),
                     front=tuple(int(i) for i in front),
                     excluded=excluded, rule=rule, decided_by=decided_by)
