"""
mergen.sampler
==============
Space-filling design construction: candidate pool, user points and
the optimisers ``'sa'`` (default), ``'sce'`` and ``'ese'``.

This module exposes four objects:

  - :class:`Sampler`        — configure and run the design construction.
  - :class:`SamplingResult` — container for design + validation + extras.
  - :class:`FocusPoint`     — denser sampling around a critical region.
  - :class:`ExclusionPoint` — a grid point removed from the candidates.

Quick start
-----------
::

    from mergen.space   import ParameterSpace
    from mergen.sampler import Sampler

    space   = ParameterSpace({'temperature': range(100, 400, 10),
                              'pressure'   : ('continuous', 0.5, 5.0)})
    sampler = Sampler(space)
    sampler.add_prescribed([[200, 2.5]], in_design=True,  in_optim=False)
    sampler.add_focus    ([350, 4.5], spread=1.5,         in_design=True, in_optim=True)
    sampler.add_exclusion([100, 0.5], spread=1.0)
    sampler.set_design(n_samples=30)
    sampler.set_optimizer('sa', n_restarts=5)
    result  = sampler.run(criteria='umaxpro', seed=44)
    result.summary()

Algorithm
---------
The default optimiser is simulated annealing (``algorithm='sa'``, see
:mod:`mergen.algorithms`). The Stochastic Coordinate Exchange optimiser
(``algorithm='sce'``) implemented in this module is an outer-inner loop:

  * **Inner loop** — for each restart, repeatedly pick one row of the
    design and propose new values along **one coordinate at a time**,
    accepting changes that improve the criterion.  The 1D nature makes
    the proposal cost O(n) (see :mod:`mergen.criteria`) and reduces
    multi-dimensional feasibility constraints to one-dimensional box
    constraints, which is the key advantage demonstrated by
    Kang (2019).

  * **Outer loop** — Iterated Local Search (Lourenço, Martin & Stützle
    2003): when the inner loop stalls, perturb the current best design
    and run the inner loop again.  The best design across all restarts
    is returned.

The first restart is seeded by a greedy maximin scheme
(Morris & Mitchell 1995); subsequent restarts are seeded by ILS kicks
from the current best (rather than random restart) for variance
reduction across seeds.

References
----------
Meyer, R. K. & Nachtsheim, C. J. (1995).
    The coordinate-exchange algorithm for constructing exact
    optimal experimental designs.
    *Technometrics*, 37(1), 60–69.
Kang, L. (2019).
    Stochastic coordinate-exchange optimal designs with complex
    constraints.
    *Quality Engineering*, 31(3), 401–416.
You, Y., Jin, G., Pan, Z. & Guo, R. (2021).
    MP-CE method for space-filling design in constrained space with
    multiple types of factors.
    *Mathematics*, 9(24), 3314.
Lourenço, H. R., Martin, O. C. & Stützle, T. (2003).
    Iterated Local Search.  In *Handbook of Metaheuristics*,
    Springer, 320–353.
Morris, M. D. & Mitchell, T. J. (1995).
    Exploratory designs for computational experiments.
    *J. Statist. Plan. Infer.*, 43, 381–402.
Loeppky, J. L., Sacks, J. & Welch, W. J. (2009).
    Choosing the sample size of a computer experiment: A practical
    guide.  *Technometrics*, 51(4), 366–376.
Kennard, R. W. & Stone, L. A. (1969).
    Computer aided design of experiments.
    *Technometrics*, 11(1), 137–148.
"""

from __future__ import annotations

import itertools
import numbers
from collections.abc import Mapping
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .space    import ParameterSpace, GridSampler
from .criteria import (
    BaseCriterion,
    get_criterion,
    list_criteria,
    nominal_supporting_criteria,
)

# ── Terminal colours ──────────────────────────────────────────────────────
from ._term import RED as _RED, GREEN as _GREEN, YELLOW as _YELLOW, \
    CYAN as _CYAN, RESET as _RESET

# Small numerical floor used to avoid log(0) in score reporting only.
_EPS = 1e-300


def _info(msg: str)  -> None: print(f"  {_CYAN}[INFO]{_RESET}     {msg}")
def _warn(msg: str)  -> None: print(f"  {_YELLOW}[WARNING]{_RESET}  {msg}")
def _ok(msg: str)    -> None: print(f"  {_GREEN}[MERGEN]{_RESET}   {msg}")
def _fatal(msg: str) -> None:
    raise ValueError(f"\n{_RED}[MERGEN ERROR]{_RESET}  {msg}")


# ======================================================================
# Parallelism helpers
# ======================================================================

def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """
    Convert a user-supplied ``n_jobs`` into a concrete positive integer
    in the range ``[1, cpu_count]`` following the joblib convention.

    Mapping (with ``cpu = os.cpu_count()``):

    - ``None``         -> 1
    - ``1, 2, ..., cpu`` -> as-is
    - ``-1``           -> cpu
    - ``-2``           -> cpu - 1
    - ``-N``           -> cpu - N + 1

    Invalid inputs (``0``, ``n > cpu``, ``n <= -cpu``, non-int) raise
    a fatal error so the user can correct the call before any heavy
    work starts.
    """
    cpu = os.cpu_count() or 1

    if n_jobs is None:
        return 1
    if not isinstance(n_jobs, (int, np.integer)) or isinstance(n_jobs, bool):
        _fatal(
            f"n_jobs must be an integer or None; got "
            f"{type(n_jobs).__name__}."
        )
    n_jobs = int(n_jobs)

    if n_jobs == 0:
        _fatal(
            "n_jobs=0 is not allowed. Use 1 for sequential, a positive "
            "integer for a fixed number of workers, or -1 to use all "
            f"{cpu} CPU(s)."
        )
    if n_jobs > cpu:
        _fatal(
            f"n_jobs={n_jobs} exceeds the number of available CPUs "
            f"({cpu}). Specify 1 <= n_jobs <= {cpu}, or use n_jobs=-1 "
            f"for all CPUs."
        )
    if n_jobs > 0:
        return n_jobs
    # n_jobs < 0
    if n_jobs <= -cpu:
        _fatal(
            f"n_jobs={n_jobs} is too negative. With {cpu} CPU(s), the "
            f"valid negative range is -{cpu - 1} <= n_jobs <= -1 "
            f"(equivalent positive values 1..{cpu})."
        )
    return cpu + 1 + n_jobs


def _normalised(design, space) -> np.ndarray:
    """Design in the [0, 1] coordinates the quality metrics expect."""
    granges = space.granges
    return ((np.asarray(design, dtype=float) - space.gmins)
            / granges)


# How compare() breaks a tie left by the selection rule.
_TIE_RULE = ("the other metrics that may serve as a priority, then the "
             "criterion and algorithm names")


# Metrics reported by quality_report() and compare() that do not measure
# space filling on their own, so they may not decide between designs.
_NOT_PRIORITY = {
    'mean_distance': "a design can raise it by crowding points into the "
                     "corners.",
    'cv_distances':  "a regular lattice scores well on it whatever its "
                     "spacing.",
}
_PRIORITY_METRICS = ('min_distance', 'minimax', 'max_abs_correlation',
                     'projection_cd2', 'cd2')


def default_criterion(space) -> str:
    """
    The criterion ``run()`` uses when none is given: ``'maxproqq'`` for
    a space with a nominal or ordinal parameter (MaxPro for
    quantitative and qualitative factors; Joseph, Gul & Ba 2020), and
    ``'umaxpro'`` otherwise.
    """
    if any(space.is_categorical(p) for p in space.names):
        return 'maxproqq'
    return 'umaxpro'


def _check_priority(priority, allow_none: bool):
    """
    Validate a ``priority`` argument and return it as a tuple of metric
    names (or None when allowed). A single name may be given as a string.
    """
    from . import metrics as _m
    if priority is None:
        if allow_none:
            return None
        _fatal("priority must name at least one metric.")
    if isinstance(priority, str):
        priority = (priority,)
    priority = tuple(priority)
    if not priority:
        _fatal("priority must name at least one metric; got an empty "
               "tuple.")
    for m in priority:
        if m in _NOT_PRIORITY:
            _fatal(f"{m!r} cannot be a priority metric: {_NOT_PRIORITY[m]} "
                   f"Use one of {sorted(_PRIORITY_METRICS)}.")
        if m not in _m._METRIC_FN:
            _fatal(f"Unknown priority metric {m!r}. "
                   f"Available: {sorted(_PRIORITY_METRICS)}")
    dupes = sorted({m for m in priority if priority.count(m) > 1})
    if dupes:
        _fatal(f"priority lists {dupes} more than once; each metric may "
               f"appear only once.")
    return priority


def _check_selection(selection, priority, tol, rtol):
    """Validate the selection rule and expand the tolerances."""
    from .selection import SELECTION_RULES, resolve_tolerance
    if selection not in SELECTION_RULES:
        _fatal(f"Unknown selection rule {selection!r}. "
               f"Available: {list(SELECTION_RULES)}")
    if not priority:
        def is_num(v):
            return isinstance(v, numbers.Real) and not isinstance(
                v, (bool, np.bool_))
        default = (is_num(tol) and tol == 1e-9
                   and is_num(rtol) and rtol == 0)
        if selection != 'pareto' or not default:
            _warn("selection, tol and rtol apply only when a priority is "
                  "given; they are ignored here.")
        return np.full(0, 1e-9), np.zeros(0)
    names = priority
    try:
        return (resolve_tolerance(tol, names, 'tol'),
                resolve_tolerance(rtol, names, 'rtol'))
    except ValueError as exc:
        _fatal(str(exc))


def _priority_values(design, priority, space, extra=None) -> np.ndarray:
    """Raw values of the priority metrics for one design, with the
    delivered out-of-budget rows ``extra`` added."""
    from . import metrics as _m
    if extra is not None and len(extra):
        design = np.vstack([np.asarray(design, dtype=float), extra])
    Xn = _normalised(design, space)
    return np.array([_m._compute_metric(m, Xn, space=space)
                     for m in priority], dtype=float)


def _select_repeat(reps, priority, space, selection='lexicographic',
                   tol=None, rtol=None, extra=None):
    """
    Pick the representative repeat from a list of OptimisationResults.

    With ``priority=None`` the repeat with the strictly lowest criterion
    score wins (ties: the design whose sorted rows come first) — the
    classic best-of-k restarts rule.

    With a ``priority`` tuple of metric names, the repeat is chosen on
    the raw values of those metrics by the ``selection`` rule (see
    :mod:`mergen.selection`). The optimiser minimises a smooth proxy
    criterion, but near-optimal proxy scores can correspond to designs
    with very different coverage properties; selecting the final design
    by the *true* objectives (optimise-by-proxy, select-by-objective)
    makes the delivered design stable under incidental setting changes
    such as n_restarts. Ties break on the lower criterion score, then on
    the design whose sorted rows come first. Deterministic throughout,
    and independent of the order in which the repeats were found.
    """
    if len(reps) == 1:
        return reps[0]
    # Remaining ties go to the design that sorts first, not to spawn
    # order, so the choice does not depend on the seed.
    order = sorted(range(len(reps)), key=lambda k: _design_key(reps[k].design))
    reps  = [reps[k] for k in order]
    if not priority:
        best = reps[0]
        for r in reps[1:]:
            if r.score < best.score:
                best = r
        return best
    from . import metrics as _m
    from .selection import select
    vals = np.array([_priority_values(r.design, priority, space, extra)
                     for r in reps])
    sel = select(vals, [m in _m._HIGHER_BETTER for m in priority],
                 rule=selection, tol=tol, rtol=rtol,
                 tiebreak=[r.score for r in reps])
    return reps[sel.winner]


def _gap_text(info) -> str:
    """'a median 2.1 %, at most 8.4 %' from repeat_info gaps."""
    return (f"a median {100 * info.get('gap_median', 0.0):.1f} %, at most "
            f"{100 * info.get('gap_max', 0.0):.1f} %")


# n_repeats='auto' runs rounds of _AUTO_ROUND repeats up to a work cap
# and stops early only when the selected result has been found by
# _AUTO_HITS of them (Mergen's own rule; it seldom triggers except on
# small problems, so the cap is usually reached). While no other repeat
# has reproduced the best result, each round doubles the optimiser
# budget of the next one. The
# total work is capped at _AUTO_MAX_EFFORT repeats at the default
# budget: a count, not a time, so a seed gives the same design on every
# machine.
_AUTO_ROUND      = 5
_AUTO_HITS       = 5
_AUTO_MAX_EFFORT = 32
# ... and at this many repeats, which matters when a short search set
# with set_optimizer() makes each repeat cheap.
_AUTO_MAX_REPEATS = 64
# Two repeats found the same result when the criterion score and every
# priority metric agree to this relative tolerance.
_SAME_RTOL       = 1e-9
# Mirror images are searched over at most this many axes (2**k images).
_MAX_MIRROR_AXES = 12


def _design_key(design, n_frozen=0):
    """Order-free identity of a design: its rows, the movable ones sorted."""
    D = np.asarray(design, dtype=float)
    M = D[n_frozen:]
    return (tuple(D[:n_frozen].ravel().tolist()),
            tuple(M[np.lexsort(M.T[::-1])].ravel().tolist()))


def _reflect_indices(idx, gs, j):
    """Grid indices with the level on axis *j* reversed."""
    idx = np.asarray(idx, dtype=np.int64)
    stride, L = gs._strides[j], gs.n_levels[j]
    lev = (idx // stride) % L
    return idx + (L - 1 - 2 * lev) * stride


def _mirror_axes(space, gs, banned, frozen_groups):
    """
    Axes along which reversing the grid maps the whole problem onto
    itself: the levels are symmetric in normalised coordinates, and the
    feasible pool, the banned nodes and each group of frozen rows are
    unchanged by the reflection. A design and its reflection along such
    an axis are then equally good on every distance-based measure.
    """
    pool   = np.sort(gs.pool_indices())
    banned = np.array(sorted(banned), dtype=np.int64)
    groups = [np.array(sorted(gs.point_to_index(r) for r in g),
                       dtype=np.int64) for g in frozen_groups]
    axes = []
    for j, v in enumerate(space.values):
        if space.is_mask[j] or len(v) < 2:
            continue
        u = (np.asarray(v, float) - space.gmins[j]) / space.granges[j]
        if not np.allclose(u, 1.0 - u[::-1], rtol=0, atol=1e-9):
            continue
        if all(np.array_equal(np.sort(_reflect_indices(a, gs, j)), a)
               for a in [pool, banned] + groups):
            axes.append(j)
    return axes


def _canonical_design(design, axes, gs, n_frozen):
    """
    The member of a design's mirror family (reflections along *axes*)
    whose sorted movable rows come first lexicographically, with those
    rows sorted. Frozen rows keep their place; the reflections in *axes*
    map them onto themselves.
    """
    D = np.asarray(design, dtype=float)

    def sort_movable(X):
        M = X[n_frozen:]
        return np.vstack([X[:n_frozen], M[np.lexsort(M.T[::-1])]])

    best = sort_movable(D)
    axes = axes[:_MAX_MIRROR_AXES]
    if not axes:
        return best
    flipped = {}
    for j in axes:
        vals = np.asarray(gs.values[j], dtype=float)
        lev  = np.array([gs.level_of(j, x) for x in D[n_frozen:, j]])
        flipped[j] = vals[len(vals) - 1 - lev]
    best_key = _design_key(best, n_frozen)
    for mask in range(1, 2 ** len(axes)):
        X = D.copy()
        for b, j in enumerate(axes):
            if mask >> b & 1:
                X[n_frozen:, j] = flipped[j]
        X = sort_movable(X)
        key = _design_key(X, n_frozen)
        if key < best_key:
            best, best_key = X, key
    return best


def _same_values(a, b):
    """True when two lists of scores agree to _SAME_RTOL."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False
    return bool(np.all(np.abs(a - b)
                       <= _SAME_RTOL * np.maximum(np.abs(a), np.abs(b))))


def _public_repeat_info(info, auto):
    """The documented view of one repeat summary."""
    return dict(n_repeats=int(info['n_repeats']), hits=int(info['hits']),
                stopped_early=bool(info['converged']) if auto else None,
                budget_factor=info['budget_factor'],
                gap_median=float(info['gap_median']),
                gap_max=float(info['gap_max']))


def _measured_repeat_info(info, reps, win, priority, space, extra):
    """``info`` with ``hits`` and the gaps measured against ``win``,
    the repeat delivered under ``priority``; how the repeats stopped
    (``converged``, ``budget_factor``) is kept."""
    def values(r):
        v = [r.score]
        if priority:
            v += list(_priority_values(r.design, priority, space, extra))
        return v
    ref = values(win)
    gaps = [abs(r.score / win.score - 1) for r in reps
            if r is not win and win.score > 0]
    return dict(info, n_repeats=len(reps),
                hits=sum(_same_values(values(r), ref) for r in reps),
                gap_median=float(np.median(gaps)) if gaps else 0.0,
                gap_max=max(gaps, default=0.0))


def _repeats_text(n_repeats):
    """'5 repeats', '1 repeat' or 'auto repeats, capped'."""
    if n_repeats == 'auto':
        return "auto repeats, capped"
    return f"{n_repeats} repeat{'s' if n_repeats > 1 else ''}"


# The parameter that sets the search length of each built-in optimiser,
# scaled by n_repeats='auto' when it doubles the budget.
_BUDGET_PARAM = {'sa': 'max_iter', 'sce': 'n_restarts', 'ese': 'Q'}


# Largest neighbourhood box enumerated around a focus centre.
_FOCUS_BOX_MAX = 200_000


# Metrics tabulated by compare(), in table order.
_COMPARE_METRICS = ('min_distance', 'minimax', 'max_abs_correlation',
                    'projection_cd2', 'cv_distances', 'mean_distance')


def _compare_one_job(sampler, cr, alg, seed, n_repeats, priority=None,
                     selection='lexicographic', tol=1e-9, rtol=0.0,
                     n_jobs=1):
    """
    Run one (criterion, algorithm) combination for ``compare`` through
    the public :meth:`Sampler.run`, so the two entry points share a
    single optimisation path and a single seed derivation. Returns the
    combination identity together with the list of repeat results (in
    spawn order), from which compare() picks the representative repeat,
    and the repeat summary (``_repeat_info``) of the combination, and
    the delivered out-of-budget rows (the same for every combination).
    ``n_jobs`` runs the combination's repeats in parallel; the result
    does not depend on it.
    The priority and selection settings are passed on because
    ``n_repeats='auto'`` stops on the result they select.

    Top-level (module-scope) function so joblib can serialise the task;
    the ``Sampler`` is passed explicitly.
    """
    import contextlib
    import io
    buf = io.StringIO()
    # Holdout sets are chosen after the optimisation and do not affect
    # it; compare() builds them once, for best_result.
    snap = sampler._snapshot_state()
    sampler._n_validation, sampler._extra_sets = 0, None
    try:
        with contextlib.redirect_stdout(buf):
            res = sampler.run(criteria=cr, algorithm=alg, seed=seed,
                              n_repeats=n_repeats, priority=priority,
                              selection=selection, tol=tol, rtol=rtol,
                              n_jobs=n_jobs, verbose=False)
    finally:
        sampler._restore_state(snap)
    return ((cr, alg), res._repeats_by_alg[alg], res._repeat_info[alg],
            res._extra_pts, res._meta['fixed_points'])


def _run_one_algorithm_task(
    alg_name:        str,
    params:          dict,
    anchors:         np.ndarray,
    budget:          int,
    initial_reserved: set,
    space:           "ParameterSpace",
    criterion:       "BaseCriterion",
    prescribed_in_pts:           List[np.ndarray],
    focus_in_pts:                List[np.ndarray],
    in_design_sa_prescribed:     List[np.ndarray],
    focus_sa_in_design_sampled:  List[np.ndarray],
    n_dims:          int,
    n_frozen:        int,
    crit_start:      int,
    seed:            int,
    verbose:         bool,
    budget_factor:   int = 1,
    visible_out:     int = 0,
):
    """
    Run a single optimiser end-to-end (initial design + optimise).

    ``budget_factor`` multiplies the default search length of a built-in
    optimiser (see ``_BUDGET_PARAM``) unless ``params`` sets it.

    The last ``visible_out`` anchor rows are criterion-visible points
    outside the budget: the optimiser sees them as frozen rows, and
    they are removed from the returned design.

    Top-level (module-scope) function so that joblib can serialise the
    task. Called sequentially
    from ``Sampler.run`` when ``n_jobs == 1`` and from a joblib
    ``Parallel`` pool when ``n_jobs > 1``.
    """
    from .algorithms import get_optimizer

    opt_cls   = get_optimizer(alg_name)
    optimiser = opt_cls(**params)

    # Per-algorithm initial design — each algorithm chooses its own
    # starting layout via prepare_initial_design().
    alg_reserved = set(initial_reserved)
    seed_design, alg_reserved = optimiser.prepare_initial_design(
        anchors  = anchors,
        budget   = budget,
        space    = space,
        reserved = alg_reserved,
        seed     = seed,
    )

    # Re-assemble the design with criterion-blind frozen rows on top.
    # This mirrors Sampler._assemble_initial_design without needing the
    # Sampler instance itself (which would not pickle cleanly).
    sa_set            = {tuple(p) for p in in_design_sa_prescribed}
    blind_prescribed  = [p for p in prescribed_in_pts
                         if tuple(p) not in sa_set]
    sa_focus_set      = {tuple(p) for p in focus_sa_in_design_sampled}
    blind_focus       = [p for p in focus_in_pts
                         if tuple(p) not in sa_focus_set]

    parts: List[np.ndarray] = []
    if blind_prescribed:
        parts.append(np.array(blind_prescribed, dtype=float))
    if blind_focus:
        parts.append(np.array(blind_focus, dtype=float))
    parts.append(seed_design)
    full_design = (np.vstack(parts) if parts
                   else np.empty((0, n_dims)))

    key = _BUDGET_PARAM.get(alg_name)
    if budget_factor > 1 and key is not None and key not in params:
        if alg_name == 'sa':
            # SAOptimizer's own default: max(2000, 100 x design rows).
            base = max(2000, 100 * len(full_design))
        else:
            base = opt_cls.get_default_params()[key]
        if base is None:
            # The optimiser sizes this from the design (ESE's Q).
            optimiser.budget_factor = int(budget_factor)
        else:
            setattr(optimiser, key, int(budget_factor * base))

    res = optimiser.optimize(
        initial_design = full_design,
        space          = space,
        criterion      = criterion,
        banned         = initial_reserved,
        n_frozen       = n_frozen,
        crit_start     = crit_start,
        seed           = seed,
        verbose        = verbose,
    )
    if visible_out:
        from dataclasses import replace
        res = replace(res, design=np.delete(
            res.design, np.arange(n_frozen - visible_out, n_frozen), axis=0))
    # Frozen rows at the top of the returned design (compare's polishing
    # keeps them).
    res.metadata['n_kept'] = n_frozen - visible_out
    return res


# ======================================================================
# FocusPoint
# ======================================================================

class FocusPoint:
    """
    A critical region where denser sampling is desired.

    Parameters
    ----------
    point : array-like, shape (n_parameters,)
        Grid coordinates of the focus centre.
    spread : float
        Width (standard deviation) of the Gaussian kernel that weights
        the neighbours, in grid steps of each ordered axis; a nominal
        axis keeps the centre's category. Must be positive.
    n_samples : int or None
        Number of focus samples to draw.
        ``None`` → auto: ``max(1, int((2 * d + 1) * spread))``.
    include_center : bool or None
        ``True``  → the centre is guaranteed to appear in the design.
        ``False`` → the centre is excluded from the candidate pool.
        ``None``  → stochastic (the centre competes with its neighbours
        through the Gaussian-weighted draw).
    in_design : bool
        ``True``  → focus samples count toward the ``n_samples`` budget.
        ``False`` → focus samples are added on top of ``n_samples`` and
        the total design size grows.
    in_optim : bool
        ``True``  → the optimiser sees these points and pushes
        optimised points away.
        ``False`` → the optimiser ignores these points (they are only
        reserved against re-selection).
    """

    def __init__(
        self,
        point:          np.ndarray,
        spread:         float            = 1.0,
        n_samples:      Optional[int]    = None,
        include_center: Optional[bool]   = None,
        in_design:      bool             = True,
        in_optim:          bool             = True,
    ) -> None:
        self.point          = np.asarray(point, dtype=float).ravel()
        self.spread         = float(spread)
        self.n_samples      = n_samples
        self.include_center = include_center
        self.in_design      = bool(in_design)
        self.in_optim          = bool(in_optim)

        if self.spread <= 0:
            _fatal(f"FocusPoint spread must be > 0, got {self.spread}.")
        if self.n_samples is not None and self.n_samples < 1:
            _fatal(f"FocusPoint n_samples must be >= 1, got {self.n_samples}.")

    def resolve_n_samples(self, n_dims: int) -> None:
        """Set ``n_samples`` from the auto formula if not user-supplied."""
        if self.n_samples is None:
            self.n_samples = max(1, int((2 * n_dims + 1) * self.spread))

    def __repr__(self) -> str:
        ic = {None: 'stochastic', True: 'guaranteed', False: 'excluded'}
        return (
            f"FocusPoint(point={self.point.tolist()}, spread={self.spread}, "
            f"n_samples={self.n_samples}, center={ic[self.include_center]}, "
            f"in_design={self.in_design}, in_optim={self.in_optim})"
        )


# ======================================================================
# ExclusionPoint
# ======================================================================

class ExclusionPoint:
    """
    A grid point removed from the candidates. No design, validation or
    extra-set point is placed there; its neighbours stay available.

    Parameters
    ----------
    point : array-like, shape (n_parameters,)
        Grid coordinates of the excluded point.
    spread : float
        Accepted but unused in this version.
    """

    def __init__(self, point: np.ndarray, spread: float = 1.0) -> None:
        self.point  = np.asarray(point, dtype=float).ravel()
        self.spread = float(spread)
        if self.spread <= 0:
            _fatal(f"ExclusionPoint spread must be > 0, got {self.spread}.")

    def __repr__(self) -> str:
        return (
            f"ExclusionPoint(point={self.point.tolist()}, spread={self.spread})"
        )


# ======================================================================
# SamplingResult
# ======================================================================

class SamplingResult:
    """
    Container returned by :meth:`Sampler.run`.

    Attributes
    ----------
    samples    : pd.DataFrame
        Main design (prescribed + focus + optimised points).
    validation : pd.DataFrame
        Maximin validation set.
    sets       : dict[str, pd.DataFrame]
        Extra named sets (e.g. ``{'test': df, 'holdout': df}``).
    designs    : dict[str, pd.DataFrame]
        The design under the name of its criterion (one entry, since
        ``run()`` optimises one criterion).
    space      : ParameterSpace
    """

    PRESCRIBED = "Prescribed"
    FOCUS      = "Focus"
    OPTIMISED  = "Optimised"
    VALIDATION = "Validation"

    def __init__(
        self,
        samples:    pd.DataFrame,
        validation: pd.DataFrame,
        space:      ParameterSpace,
    ) -> None:
        # Nominal and ordinal columns hold their labels; library code
        # that needs numbers uses space.encode_frame().
        self.samples        = space.decode_frame(samples)
        self.validation     = space.decode_frame(validation)
        self.space          = space
        self.sets:    Dict[str, pd.DataFrame] = {}
        self.designs: Dict[str, pd.DataFrame] = {}
        # Multi-algorithm results — populated by Sampler.run() when
        # ``algorithm=[...]`` is requested. Keys: algorithm name,
        # Values: OptimisationResult instance (algorithms/base.py).
        self.algorithm_results: Dict[str, Any] = {}
        # The algorithm whose design is exposed as ``self.samples`` (the
        # lowest-scoring one when multiple were run, or the one chosen
        # on the priority metrics when a priority was given).
        self.best_algorithm: Optional[str] = None
        self.output_dir     = "outputs"
        self._meta: dict    = {}

    # ------------------------------------------------------------------ #
    # Multi-algorithm convenience                                          #
    # ------------------------------------------------------------------ #

    @property
    def best_design(self) -> pd.DataFrame:
        """Alias for :attr:`samples` — the design from the best algorithm."""
        return self.samples

    @property
    def repeat_info(self) -> Dict[str, dict]:
        """Per algorithm, how its repeats ended: ``n_repeats`` run,
        ``hits`` (repeats that found the delivered result),
        ``stopped_early`` (``n_repeats='auto'`` found it 5 times before
        the work cap; ``None`` for an integer ``n_repeats``),
        ``budget_factor`` (search-length multiplier reached), and
        ``gap_median`` / ``gap_max``, the relative distance of the other
        repeats' criterion scores from the delivered one. Empty for a
        loaded design."""
        auto = self._meta.get('n_repeats') == 'auto'
        return {a: _public_repeat_info(i, auto)
                for a, i in getattr(self, '_repeat_info', {}).items()}

    @property
    def best_score(self) -> Optional[float]:
        """Criterion score of the best algorithm's design, or None."""
        if self.best_algorithm and self.best_algorithm in self.algorithm_results:
            return float(self.algorithm_results[self.best_algorithm].score)
        return None

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def summary(self) -> None:
        """Print a concise design summary to stdout."""
        vc  = self.samples["point_type"].value_counts()
        W   = 52
        sep = "─" * W
        print(f"\n{sep}")
        print("  MERGEN Design Summary")
        print(sep)
        for label in (self.PRESCRIBED, self.FOCUS, self.OPTIMISED):
            n = vc.get(label, 0)
            if n:
                print(f"  {label:<16}: {n}")
        print(f"  {'Total design':<16}: {len(self.samples)}")
        if len(self.validation):
            print(f"  {'Validation':<16}: {len(self.validation)}")
        for name, df in self.sets.items():
            print(f"  {name:<16}: {len(df)}")
        print(sep)
        print(f"  {'Parameters':<16}: {self.space.n_parameters}")
        print(f"  {'Candidates':<16}: {self.space.n_candidates}")
        if self._meta:
            crit = self._meta.get('criteria', '?')
            seed = self._meta.get('seed', '?')
            print(f"  {'Criterion':<16}: {crit}")
            print(f"  {'Seed':<16}: {seed}")

            # Algorithm reporting: single or multi
            alg = self._meta.get('algorithm', None)
            if isinstance(alg, list):
                print(f"  {'Algorithms':<16}: {', '.join(alg)}")
                print(f"  {'Best algorithm':<16}: {self.best_algorithm}")
            elif alg is not None:
                print(f"  {'Algorithm':<16}: {alg}")
            src = self._meta.get('polished_from')
            if src:
                print(f"  {'Polished':<16}: from {src[0]} + {src[1]} "
                      f"(compare)")

        # Per-algorithm scores when multiple were run
        if len(self.algorithm_results) > 1:
            print(sep)
            print("  Per-algorithm scores")
            print(sep)
            # Sort by score (best first)
            ranked = sorted(
                self.algorithm_results.items(),
                key=lambda kv: kv[1].score,
            )
            for name, res in ranked:
                mark = " *" if name == self.best_algorithm else "  "
                print(f"  {mark} {name:<6} : score={res.score:.6g}  "
                      f"elapsed={res.elapsed:.2f}s  n_iter={res.n_iter}")
        print(f"{sep}\n")

    def quality_report(
        self,
        metrics:          Union[str, list]   = 'default',
        criteria_metrics: Optional[list]     = None,
        mc_samples:       int                = 300,
        verbose:          bool               = True,
    ) -> dict:
        """
        Compute and print design quality metrics.

        Delegates to :func:`mergen.metrics.quality_report`.

        Each metric is reported both as a raw value and as a percentile
        rank against a Monte Carlo baseline of random designs of the
        same size, so the table shows not just the value but whether it
        is good. Pass ``mc_samples=0`` to skip the baseline (values
        only).

        Parameters
        ----------
        metrics          : ``'default'`` or list of metric names
        criteria_metrics : list of criterion names to evaluate post-hoc
        mc_samples       : number of random designs for the Monte Carlo
                           baseline (``0`` disables it; default ``300``)
        verbose          : print the metrics table (default ``True``)

        Returns
        -------
        dict of metric values and (optionally) baseline statistics
        """
        from . import metrics as _metrics
        return _metrics.quality_report(
            self,
            metrics=metrics,
            criteria_metrics=criteria_metrics,
            mc_samples=mc_samples,
            verbose=verbose,
        )

    def comparison(self) -> pd.DataFrame:
        """
        The standard quality metrics of the design, as a one-row
        DataFrame indexed by its criterion (``run()`` optimises one
        criterion). To compare criteria, use :meth:`Sampler.compare`.
        """
        from . import metrics as _metrics
        return _metrics.comparison(self)

    def plot(self, kind: str = 'pairplot', **kwargs) -> None:
        """
        Visualise the design.  Delegates to :mod:`mergen.output`.

        Parameters
        ----------
        kind : ``'pairplot'`` | ``'1d'`` | ``'2d'`` | ``'distances'``
               | ``'correlation'`` | ``'quality'`` | ``'comparison'``
               | ``'all'`` (see :func:`mergen.output.plot`)
        """
        from . import output as _output
        _output.plot(self, kind=kind, **kwargs)

    def to_csv(self, filename: str = 'design.csv') -> None:
        from . import output as _output
        _output.export_csv(self, filename)

    def to_json(self, filename: str = 'design.json') -> None:
        from . import output as _output
        _output.export_json(self, filename)

    def to_markdown(self, filename: str = 'design.md') -> None:
        from . import output as _output
        _output.export_markdown(self, filename)

    def to_latex(self, filename: str = 'design.tex') -> None:
        from . import output as _output
        _output.export_latex(self, filename)

    def to_html(self, filename: str = 'design.html') -> None:
        from . import output as _output
        _output.export_html(self, filename)

    def to_excel(self, filename: str = 'design.xlsx') -> None:
        from . import output as _output
        _output.export_excel(self, filename)

    def _full_df(self) -> pd.DataFrame:
        """Combine all sets into a single DataFrame."""
        frames = [self.samples]
        if len(self.validation):
            df = self.validation.copy()
            df['point_type'] = self.VALIDATION
            frames.append(df)
        for name, df in self.sets.items():
            frames.append(df.copy())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def __repr__(self) -> str:
        return (
            f"SamplingResult("
            f"samples={len(self.samples)}, "
            f"validation={len(self.validation)}, "
            f"sets={list(self.sets.keys())})"
        )


# ======================================================================
# Sampler
# ======================================================================

class ComparisonResult:
    """
    Outcome of :meth:`Sampler.compare`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per (criterion, algorithm) combination, in the order the
        criteria and algorithms were listed. The metric columns hold
        percentile ranks (0-100, higher is better) against the shared
        Monte Carlo baseline; they describe how each design compares
        with random designs and are not used for selection. The winner
        is flagged in the ``best`` column.
    metrics : pandas.DataFrame
        The raw metric values of each combination's representative
        design (normalised coordinates), on which the selection is made.
    results : dict
        Maps ``(criterion, algorithm)``, or ``(criterion,
        '<algorithm>+polish')`` for a polished design, to the representative
        :class:`~mergen.algorithms.base.OptimisationResult` (design,
        criterion score and optimiser metadata) of that combination. The
        score of a polished design is NaN; ``best_result`` carries it
        when that design wins.
    best : tuple of (str, str)
        The winning (criterion, algorithm) pair (read-only).
    priority : tuple of str
        The metrics that decided the selection, in order.
    selection : str
        The rule that was applied, ``'lexicographic'`` or ``'pareto'``.
    tol, rtol : dict
        Absolute and relative tolerance per priority metric.
    ties : tuple of (str, str)
        Combinations the rule could not tell apart from the winner on the
        metrics (the winner included).
    decided_by : str
        What settled the choice: ``'metrics'`` (no tie), ``'priority
        order'`` (a Pareto tie broken by the order of the priority
        metrics) or ``'tiebreak'`` (the other priority-capable metrics of
        the table, then the criterion and algorithm names).
    front : tuple of (str, str)
        Combinations not dominated on the priority metrics, within the
        tolerance. The ``'pareto'`` rule picks its winner from this
        set. Under ``'lexicographic'`` it is informational: tolerant
        equality is not transitive, so with a large ``tol`` the
        lexicographic winner can lie outside it.
    repeat_info : dict
        Maps each row of the table to how the repeats of its combination
        ended, as in :attr:`SamplingResult.repeat_info` (``n_repeats``,
        ``hits``, ``stopped_early``, ``budget_factor``, ``gap_median``,
        ``gap_max``). ``hits`` and the gaps are measured against the
        repeat the current priority selects; a ``'+polish'`` row
        reports the repeats of the combination it started from.
    """

    def __init__(self, table, metrics, results, best, priority,
                 selection='lexicographic', tol=None, rtol=None,
                 ties=(), front=(), decided_by='metrics'):
        self.table     = table
        self.metrics   = metrics
        self.results   = results
        self._best     = best
        self.priority  = tuple(priority)
        self.selection = selection
        self.tol       = dict(tol or {})
        self.rtol      = dict(rtol or {})
        self.ties      = tuple(ties) if ties else (best,)
        self.front     = tuple(front)
        self.decided_by = decided_by
        self._best_result = None   # set by Sampler.compare after selection
        self._context     = None   # what reselect() needs; set by compare
        self.repeat_info: Dict[Tuple[str, str], dict] = {}

    @property
    def best(self) -> Tuple[str, str]:
        """The winning (criterion, algorithm) pair. To choose again with
        a different rule, use :meth:`reselect`."""
        return self._best

    @property
    def best_result(self) -> "SamplingResult":
        """The full SamplingResult of the winning combination, identical
        to calling run() for that combination with the same seed,
        n_repeats, priority, selection and tolerances, unless the winner
        is a polished design, which run() does not produce. After
        :meth:`reselect` it is built from all the stored repeats, as
        ``n_repeats`` equal to their number."""
        return self._best_result

    def selection_text(self) -> str:
        """One line describing how the winner was chosen."""
        from .metrics import _HIGHER_BETTER

        def tol_text(m):
            t, r = self.tol.get(m, 1e-9), self.rtol.get(m, 0.0)
            return f"tol={t:g}" + (f", rtol={r:g}" if r else "")

        parts = [f"{m} ({'higher' if m in _HIGHER_BETTER else 'lower'} "
                 f"is better; {tol_text(m)})" for m in self.priority]
        if self.selection == 'lexicographic':
            return ("Selection: lexicographic on raw metric values, "
                    "in order: " + " > ".join(parts))
        return ("Selection: Pareto front + Utopia distance on raw metric "
                "values, objectives unordered: " + ", ".join(parts))

    def reselect(
        self,
        priority:  Optional[Tuple[str, ...]] = None,
        selection: Optional[str] = None,
        tol:       Optional[Union[float, Dict[str, float]]] = None,
        rtol:      Optional[Union[float, Dict[str, float]]] = None,
        verbose:   bool = True,
    ) -> "ComparisonResult":
        """
        Choose the winner again with a different priority, rule or
        tolerance, without re-running the sweep.

        The stored repeats of every combination are reused; only the
        representative repeats, the metrics and the winner are
        recomputed. Arguments left as None keep their current value.
        ``best_result`` is rebuilt from the winning combination's stored
        repeats through the sampler that produced this comparison, so
        it holds the design the table describes; the sampler must not
        have been reconfigured in between.

        Returns
        -------
        ComparisonResult
            A new result; this one is left unchanged.
        """
        if self._context is None:
            _fatal("This ComparisonResult was not produced by "
                   "Sampler.compare(); it cannot be reselected.")
        ctx = self._context
        priority  = self.priority if priority is None else priority
        priority  = (priority,) if isinstance(priority, str) else priority
        selection = self.selection if selection is None else selection

        if tol is not None or rtol is not None:
            _check_selection(selection, _check_priority(priority, False),
                             1e-9 if tol is None else tol,
                             0.0 if rtol is None else rtol)
        # Tolerances not given again keep the values passed to compare();
        # per-metric entries for metrics outside the new priority are
        # ignored rather than rejected.
        tol  = ctx['tol']  if tol  is None else tol
        rtol = ctx['rtol'] if rtol is None else rtol
        return ctx['sampler']._finish_compare(
            repeats=ctx['repeats'], crit_list=ctx['crit_list'],
            alg_list=ctx['alg_list'], priority=priority,
            selection=selection, tol=tol, rtol=rtol,
            baseline=ctx['baseline'], seed=ctx['seed'],
            n_repeats=ctx['n_repeats'], verbose=verbose,
            extra_pts=ctx['extra_pts'], polish=ctx.get('polish', False),
            polish_kicks=ctx.get('polish_kicks', 0),
            settle=ctx.get('settle'))

    def summary(self) -> None:
        """Print the comparison table and how the winner was chosen."""
        print()
        print("═" * 72)
        print("  MERGEN — Criterion / Algorithm Comparison")
        print("═" * 72)
        print(f"  {self.selection_text()}")
        print("  Table: percentile vs shared MC baseline (higher is "
              "better; descriptive only)")
        print("─" * 72)
        with pd.option_context('display.width', 100,
                               'display.float_format',
                               lambda v: f"{v:6.1f}"):
            print(self.table.to_string(index=False))
        print("─" * 72)
        print(f"  Best: criteria='{self.best[0]}', "
              f"algorithm='{self.best[1]}'")
        tie = self._tie_text()
        if tie:
            print(f"  Tie : {tie}")
        print("═" * 72)

    def _tie_text(self) -> str:
        """How a tie on the metrics was broken, or '' if there was none."""
        if self.decided_by == 'metrics':
            return ''
        others = ", ".join(f"{c}/{a}" for c, a in self.ties
                           if (c, a) != self.best)
        if self.decided_by == 'priority order':
            return (f"tied under the Pareto rule with {others}; broken by "
                    f"the order of the priority metrics.")
        if self.selection == 'pareto':
            return (f"tied under the Pareto rule with {others}; the "
                    f"priority order left more than one, decided by "
                    f"{_TIE_RULE}.")
        return (f"indistinguishable from {others} within the tolerance; "
                f"decided by {_TIE_RULE}.")

    def plot(self, save: bool = False, filename: Optional[str] = None,
             show: bool = True, **kwargs) -> None:
        """
        Heat map of the percentile-rank table.

        Rows are (criterion, algorithm) combinations, columns are the
        quality metrics, cell colour encodes the percentile rank; the
        winning row is starred. Saved under ``outputs/`` when
        ``save=True``.
        """
        from . import output as _output
        _output.plot_comparison_matrix(
            self, save=save, filename=filename, show=show, **kwargs)

    def to_markdown(self, filename: str) -> None:
        """
        Save the comparison table as a Markdown file under the output
        directory.

        Unlike calling ``comparison.table.to_markdown(...)`` directly,
        this writes into ``outputs/`` (creating it if needed), matching
        where the plots and design exports are saved.
        """
        import os
        outdir = getattr(self, 'output_dir', 'outputs')
        os.makedirs(outdir, exist_ok=True)
        path = filename if os.path.isabs(filename) or os.path.dirname(filename) \
            else os.path.join(outdir, filename)
        md = self.table.to_markdown(index=False)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(md + "\n")
        print(f"  Saved: {path}")

    def __repr__(self) -> str:
        return (f"ComparisonResult(best={self.best}, "
                f"n_combinations={len(self.results)}, "
                f"priority={self.priority}, selection={self.selection!r})")


class Sampler:
    """
    Space-filling design sampler (default optimiser: simulated
    annealing, ``'sa'``; also ``'sce'`` and ``'ese'``).

    The sampler is configured fluently — ``add_*`` methods register
    prescribed/focus/exclusion points, ``set_*`` methods set sizes and
    optimiser hyperparameters, and :meth:`run` produces the design.

    Parameters
    ----------
    space : ParameterSpace

    Examples
    --------
    >>> space   = ParameterSpace({'x': range(1, 21), 'y': range(1, 21)})
    >>> sampler = Sampler(space)
    >>> sampler.set_design(n_samples=20)
    >>> result  = sampler.run(seed=44)
    >>> result.summary()
    """

    def __init__(self, space: ParameterSpace) -> None:
        if not isinstance(space, ParameterSpace):
            _fatal(f"space must be a ParameterSpace, got {type(space).__name__}.")
        if not space.is_valid():
            _fatal("ParameterSpace has no parameters or no feasible candidates.")

        self.space = space

        # Registered points
        # Each prescribed entry: (point, in_design, in_optim)
        self._prescribed:  List[Tuple[np.ndarray, bool, bool]] = []
        self._focus:       List[FocusPoint]      = []
        self._exclusions:  List[ExclusionPoint]  = []

        # Design settings
        self._n_samples:    Optional[int]         = None
        self._n_validation: Optional[int]         = None
        self._extra_sets:   Optional[Dict]        = None
        # User-supplied named sets: {name: (points, colour_or_None)}
        self._user_sets:    Dict[str, Tuple[np.ndarray, Optional[str]]] = {}
        # Externally supplied design: (points, label, colour_or_None)
        self._loaded_design: Optional[Tuple[np.ndarray, str, Optional[str]]] = None
        self._dim_weights:  Optional[np.ndarray]  = None

        # Per-optimiser hyperparameter store: {'sa': {...}, 'sce': {...}, ...}
        # Populated by set_optimizer(name, **kwargs). Empty dict means
        # use the optimiser's defaults.
        self._optimizer_params: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Public: configuration                                                #
    # ------------------------------------------------------------------ #

    def add_prescribed(
        self,
        points:    Union[Sequence, np.ndarray],
        in_design: bool = True,
        in_optim:  bool = True,
    ) -> "Sampler":
        """
        Add one or more prescribed (fixed) points.

        Prescribed points are always present in the final design and are
        never moved by the optimiser.

        Parameters
        ----------
        points : array-like
            Single point ``[x1, x2, ...]`` or list of points.
            All points must lie on the parameter grid.
        in_design : bool
            ``True``  → counted within the ``n_samples`` budget.
            ``False`` → added on top of ``n_samples`` (the total
            design size grows accordingly).
        in_optim : bool
            ``True`` (default) → the optimiser sees these points and
            places the optimised points to complement them.
            ``False`` → the optimiser ignores them (the points are
            only reserved against being re-selected).

        Returns
        -------
        self
        """
        pts = self.space.encode_points(points, context="Prescribed point")
        return self._add_prescribed_codes(pts, in_design, in_optim)

    def _add_prescribed_codes(self, pts, in_design: bool = True,
                              in_optim: bool = True,
                              on_grid: bool = True) -> "Sampler":
        """
        add_prescribed() for rows already on the grid encoding. With
        ``on_grid=False`` a row need only lie within the parameter
        ranges (and on a level of each nominal or ordinal parameter):
        sequential.fill_around takes references such as literature
        points this way, out of the budget, so they are never
        delivered.
        """
        for row in np.asarray(pts, dtype=float).reshape(
                -1, self.space.n_parameters):
            if on_grid:
                validated = self.space.validate_point(
                    row, label="Prescribed point", encoded=True)
            else:
                validated = self.space.validate_reference(
                    row, label="Reference point")
            self._prescribed.append((validated, bool(in_design), bool(in_optim)))
        return self

    def _encode_points(self, points, context: str) -> np.ndarray:
        """Points as grid-encoded floats; see ParameterSpace.encode_points."""
        return self.space.encode_points(points, context=context)

    def add_set(
        self,
        name:   str,
        points: Union[Sequence, np.ndarray],
        color:  Optional[str] = None,
    ) -> "Sampler":
        """
        Add a user-supplied named point set (e.g. an external test set).

        The points are validated against the parameter grid, reserved so
        the optimiser cannot re-select them, and carried into the result
        as ``result.sets[name]``. They appear in plots under ``name`` and
        in exports as a separate table.

        Parameters
        ----------
        name : str
            Label of the set (e.g. ``'test'``). Must not collide with
            the built-in labels ``'Optimised'``, ``'Validation'``,
            ``'Prescribed'`` or ``'Focus'``, nor with a generated extra
            set or a previously added set.
        points : array-like
            Single point ``[x1, x2, ...]`` or list of points. All points
            must lie on the parameter grid.
        color : str, optional
            Matplotlib colour used for this set in plots
            (e.g. ``'#3a86ff'``). If omitted, a fallback colour
            is assigned by the plotting layer.

        Returns
        -------
        self
        """
        builtin = {"Optimised", "Validation", "Prescribed", "Focus"}
        taken   = builtin | set(self._user_sets) | set(self._extra_sets or {})
        if self._loaded_design is not None:
            taken.add(self._loaded_design[1])
        if name in taken:
            _fatal(f"Set name '{name}' is already in use.")
        pts = self._encode_points(points, context=f"Set '{name}'")
        validated = np.vstack([
            self.space.validate_point(row, label=f"Set '{name}' point",
                                      encoded=True)
            for row in pts
        ])
        self._user_sets[name] = (validated, color)
        return self

    def load_design(
        self,
        points: Union[Sequence, np.ndarray, pd.DataFrame],
        name:   str = "Existing",
        color:  Optional[str] = None,
    ) -> "Sampler":
        """
        Load an existing design instead of optimising a new one.

        The supplied points become the design itself: :meth:`run` skips
        optimisation entirely and only generates the requested
        validation set and any extra sets (maximin sets) around them.
        Use :func:`mergen.sequential.extend` instead if you want to
        grow the design with new optimised points.

        Parameters
        ----------
        points : array-like or pandas.DataFrame
            The existing design. A DataFrame is matched to the
            parameter space by column names; an array must have one
            column per parameter in space order. All points must lie
            on the parameter grid.
        name : str
            Label under which the points appear in plots, summaries
            and exports (default ``'Existing'``).
        color : str, optional
            Matplotlib colour for this label in plots. Defaults to the
            Optimised blue (``'#3a86ff'``).

        Returns
        -------
        self
        """
        if isinstance(points, pd.DataFrame):
            missing = [p for p in self.space.names if p not in points.columns]
            if missing:
                _fatal(
                    f"load_design: DataFrame is missing parameter "
                    f"column(s) {missing}."
                )
            pts = self._encode_points(
                points[self.space.names].to_numpy(dtype=object).tolist(),
                context="load_design")
        else:
            pts = self._encode_points(points, context="load_design")
        if pts.ndim != 2 or pts.shape[1] != self.space.n_parameters:
            _fatal(
                f"load_design: each point must have "
                f"{self.space.n_parameters} coordinates, got shape {pts.shape}."
            )
        if str(name) in {"Validation", "Prescribed", "Focus",
                         "Optimised"} | set(self._user_sets) | set(
                self._extra_sets or {}):
            _fatal(f"load_design: the label '{name}' is already used by "
                   f"another set of points.")
        validated = np.vstack([
            self.space.validate_point(row, label="Loaded design point",
                                      encoded=True)
            for row in pts
        ])
        self._loaded_design = (validated, str(name), color)
        return self

    def add_focus(
        self,
        point:          Sequence[float],
        spread:         float          = 1.0,
        n_samples:      Optional[int]  = None,
        include_center: Optional[bool] = None,
        in_design:      bool           = True,
        in_optim:          bool           = True,
    ) -> "Sampler":
        """
        Add a focus point — denser sampling near a critical region.

        Parameters
        ----------
        point          : grid coordinates of the focus centre
        spread         : width of the Gaussian kernel in grid steps of
                         each ordered axis (a nominal axis keeps the
                         centre's category); must be positive
        n_samples      : number of focus samples (``None`` → auto)
        include_center : ``True`` / ``False`` / ``None``
                         (guaranteed / excluded / stochastic)
        in_design      : ``True`` → within budget; ``False`` → extra
        in_optim          : ``True`` → the optimiser sees these points

        Returns
        -------
        self
        """
        pt = self.space.validate_point(point, label="Focus point")
        if not (isinstance(spread, numbers.Real) and spread > 0):
            _fatal(f"spread must be a positive number; got {spread!r}")
        self._focus.append(FocusPoint(
            pt, spread=spread, n_samples=n_samples,
            include_center=include_center,
            in_design=in_design, in_optim=in_optim,
        ))
        return self

    def add_exclusion(
        self,
        point:  Sequence[float],
        spread: float = 1.0,
    ) -> "Sampler":
        """
        Add an exclusion point: a grid point that is never proposed.

        Only the point itself is removed from the candidates; its
        neighbours stay available. To keep a whole region free, use
        :meth:`ParameterSpace.add_constraint`.

        Parameters
        ----------
        point  : grid coordinates of the excluded point
        spread : accepted but unused in this version

        Returns
        -------
        self
        """
        pt = self.space.validate_point(point, label="Exclusion point")
        self._exclusions.append(ExclusionPoint(pt, spread=spread))
        return self

    def set_design(
        self,
        n_samples:    Optional[int]  = None,
        n_validation: Optional[int]  = None,
        extra_sets:   Optional[Dict] = None,
    ) -> "Sampler":
        """
        Configure design size and holdout sets.

        Parameters
        ----------
        n_samples    : total design size (``None`` → 10 × n_parameters,
                       Loeppky, Sacks & Welch 2009).
        n_validation : validation set size (``None`` → 20% of
                       ``n_samples``).
        extra_sets   : dict of ``{name: size}`` for additional
                       maximin holdout sets, e.g.
                       ``{'test': 10, 'holdout': 5}``.

        Returns
        -------
        self
        """
        def count(value, label, minimum):
            if value is None:
                return None
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or value < minimum):
                _fatal(f"{label} must be an integer >= {minimum}; "
                       f"got {value!r}.")
            return int(value)

        n_samples    = count(n_samples, "n_samples", 1)
        n_validation = count(n_validation, "n_validation", 0)
        if extra_sets is not None:
            if not isinstance(extra_sets, Mapping):
                _fatal(f"extra_sets must be a dict of {{name: size}}; "
                       f"got {extra_sets!r}.")
            builtin = {"Optimised", "Validation", "Prescribed", "Focus"}
            for name, size in extra_sets.items():
                if not isinstance(name, str):
                    _fatal(f"Extra set names must be strings; got "
                           f"{name!r}.")
                if name in builtin or name in self._user_sets:
                    _fatal(f"Set name '{name}' is already in use.")
                count(size, f"Size of extra set '{name}'", 0)
            extra_sets = {name: int(size)
                          for name, size in extra_sets.items()}
        self._n_samples    = n_samples
        self._n_validation = n_validation
        self._extra_sets   = extra_sets
        return self

    def set_optimizer(
        self,
        name:    str,
        **kwargs: Any,
    ) -> "Sampler":
        """
        Configure hyperparameters for an optimisation algorithm.

        The named algorithm must be registered in
        :mod:`mergen.algorithms` (e.g. ``'sa'``, ``'sce'``, ``'ese'``).
        Hyperparameters are validated against the algorithm's
        :meth:`~mergen.algorithms.BaseOptimizer.get_default_params`
        schema; unknown keys raise :class:`ValueError`.

        Calling this method does *not* run the optimiser — it only stores
        the parameters for later use by :meth:`run`. You can call it
        multiple times to configure different algorithms.

        Parameters
        ----------
        name : str
            The registered name of the optimiser (e.g. ``'sa'``).
        **kwargs
            Hyperparameters specific to the chosen algorithm. See each
            algorithm's documentation for the full list.

        Returns
        -------
        self
            Enables fluent chaining.

        Raises
        ------
        KeyError
            If ``name`` is not a registered optimiser.
        ValueError
            If any ``kwargs`` key is not a valid hyperparameter for the
            algorithm.

        Examples
        --------
        >>> sampler.set_optimizer('sa', n_restarts=5, max_iter=10000)
        >>> sampler.set_optimizer('sce', n_restarts=3, K_axis=20)
        >>> sampler.set_optimizer('ese', M=50, J=20)
        """
        # Import here to avoid circular imports at module load time
        from .algorithms import get_optimizer

        # Normalise the name (registry is case-insensitive on user input)
        name = name.strip().lower()

        # Look up the optimiser class (raises KeyError if unknown)
        opt_cls = get_optimizer(name)

        # Validate kwargs against the optimiser's parameter schema by
        # instantiating it (ValueError on unknown keys), and their
        # values (ValueError when out of range).
        opt_cls(**kwargs).validate()

        # Store the hyperparameters
        self._optimizer_params[name] = dict(kwargs)
        return self

    def set_dimension_weights(
        self,
        weights: Union[Dict[str, float], List[float]],
    ) -> "Sampler":
        """
        Store per-dimension importance weights.

        In this version no criterion or metric reads the weights, so
        they do not change the design; a warning says so.

        Parameters
        ----------
        weights : dict ``{name: weight}`` or list of floats
                  (one weight per parameter, in parameter order).

        Returns
        -------
        self
        """
        names = self.space.names
        if isinstance(weights, dict):
            w = np.array([weights.get(n, 1.0) for n in names], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
        if isinstance(weights, dict):
            unknown = [k for k in weights if k not in names]
            if unknown:
                _fatal(f"set_dimension_weights: unknown parameter(s) "
                       f"{unknown}. Parameters: {names}.")
        if len(w) != self.space.n_parameters:
            _fatal(
                f"dimension_weights length ({len(w)}) must match "
                f"n_parameters ({self.space.n_parameters})."
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
            _fatal(f"set_dimension_weights: weights must be finite, "
                   f"non-negative and not all zero; got {w.tolist()}.")
        self._dim_weights = w / w.sum()
        warnings.warn(
            "set_dimension_weights() has no effect in this version: no "
            "criterion or metric reads the weights.", UserWarning,
            stacklevel=2)
        return self

    def __repr__(self) -> str:
        return (
            f"Sampler("
            f"space={self.space.n_parameters}D, "
            f"prescribed={len(self._prescribed)}, "
            f"focus={len(self._focus)}, "
            f"exclusions={len(self._exclusions)})"
        )

    # ------------------------------------------------------------------ #
    # Public: run                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        criteria:  Union[str, List[str], BaseCriterion, None] = None,
        algorithm: Union[str, List[str]]                = 'sa',
        seed:      Optional[int]                        = 44,
        n_repeats: Union[int, str]                      = 'auto',
        priority:  Optional[Tuple[str, ...]]            = None,
        n_jobs:    Optional[int]                        = -1,
        verbose:   bool                                  = True,
        selection: str                                  = 'pareto',
        tol:       Union[float, Dict[str, float]]       = 1e-9,
        rtol:      Union[float, Dict[str, float]]       = 0.0,
    ) -> SamplingResult:
        """
        Generate the space-filling design.

        Parameters
        ----------
        criteria : str, BaseCriterion instance or None
            Optimisation criterion (a one-element list is accepted; use
            :meth:`compare` to try several).
            Available built-in names: ``'umaxpro'``, ``'maxpro'``,
            ``'maxproqq'``, ``'phi_p'``, ``'cd2'``, ``'qqd'``,
            ``'stratified'``. ``None`` (default) chooses
            ``'maxproqq'`` when the space has a nominal or ordinal
            parameter and ``'umaxpro'`` otherwise (see
            :func:`default_criterion`).
        algorithm : str or list of str, optional
            Name of the optimiser (or list of names) to run. Each name
            must be a registered optimiser
            (see :func:`mergen.list_optimizers`). When a list is given,
            every algorithm is run independently and the results are
            collected in :attr:`SamplingResult.algorithm_results`.
            Default ``'sa'``.
        seed : int or None
            Random seed for reproducibility (default 44). ``None`` is
            the same as 44: a run is always reproducible.
        n_repeats : int or 'auto', default 'auto'
            Number of independent optimisation repeats. Repeat seeds are
            derived from ``seed`` via NumPy's SeedSequence.spawn; the
            representative repeat is returned (see ``priority``). An
            integer runs exactly that many repeats. ``'auto'`` runs
            rounds of 5 repeats up to a work cap and stops early only
            when the selected result has been found by 5 of them; it
            usually reaches the cap, so it does not make the result
            independent of the seed (see below). Two repeats found the
            same
            result when the criterion score and the priority metrics
            agree to a relative 1e-9. While no other repeat has
            reproduced the best result, each round doubles the
            optimiser's search length for the next round (``max_iter``
            for SA, ``n_restarts`` for SCE, ``Q`` for ESE), unless it was
            set with :meth:`set_optimizer`.
            The work stops at 64 repeats, or earlier at the work of 32
            repeats at the default search length, with a warning that
            gives how far the other repeats' criterion scores lie from
            the delivered one. The limit counts work, not time, so a
            seed gives the same design on every machine. Independent
            searches seldom find an identical design except on small
            problems, so ``'auto'`` usually runs to the work limit:
            on benchmark spaces of 15 to 30 runs (2 to 5 factors, all
            criteria, three optimisers, repeats at the default length),
            7 % of runs stopped early, and the best of 20 repeats
            varied across seeds by a median 2.1 % of the criterion
            score (up to 11 %). :attr:`SamplingResult.repeat_info`
            reports, per algorithm, whether the run stopped early and
            how far the other repeats' scores lie from the delivered
            one.
        priority : tuple of str, optional
            Quality-metric names used to pick the representative repeat
            when ``n_repeats > 1``, and the delivered algorithm when
            several are run. ``None`` (default) keeps the repeat
            with the lowest criterion score (classic best-of-k
            restarts). When given — e.g.
            ``('min_distance', 'max_abs_correlation')`` — the repeat is
            chosen on the raw values of these metrics by the
            ``selection`` rule instead (optimise-by-proxy,
            select-by-objective), which keeps the delivered design
            stable when near-optimal criterion scores correspond to
            different coverage trade-offs. Pass the same ``priority``,
            ``selection``, ``tol`` and ``rtol`` used in :meth:`compare`
            to reproduce its ``best_result`` exactly, unless the winner
            is a polished design (``'<algorithm>+polish'``), which
            ``run()`` does not produce; ``compare(polish=False)`` gives
            winners that ``run()`` reproduces. When several
            algorithms are run, the same rule also chooses which
            algorithm's design is delivered.
        n_jobs : int or None, default -1
            Number of parallel workers across the independent (algorithm,
            repeat) runs, following the joblib convention: ``-1`` (the
            default) uses every CPU, ``-2`` all but one, a positive
            integer that many, and ``None`` or ``1`` runs sequentially.
            The restart loop inside one run is serial (each restart
            kicks from the previous best). Results are identical for
            every ``n_jobs``.
        verbose : bool, optional
            Print progress information. Default ``True``.
        selection : ``'pareto'`` or ``'lexicographic'``, optional
            Rule for choosing among repeats (and algorithms) when
            ``priority`` is given (see :mod:`mergen.selection`).
            ``'pareto'`` (default) treats the metrics as competing
            objectives and takes the non-dominated design closest to the
            Utopia point; ``'lexicographic'`` applies them in the order
            listed.
        tol, rtol : float or dict, optional
            Absolute and relative tolerance under which two metric
            values count as equal: ``|a - b| <= tol + rtol * max(|a|,
            |b|)``. Either a number for every metric or a dict
            ``{metric: value}``. The defaults (``1e-9``, ``0``) absorb
            floating-point noise only.

        Returns
        -------
        SamplingResult
            Contains the (per-algorithm) optimised designs and quality
            metadata. When a single algorithm is requested,
            :attr:`SamplingResult.samples` exposes its design directly;
            when multiple algorithms are requested, use
            :attr:`SamplingResult.algorithm_results` to access each
            individually and :meth:`SamplingResult.comparison` to
            tabulate them.
        """
        priority = _check_priority(priority, allow_none=True)
        tol_arr, rtol_arr = _check_selection(selection, priority, tol, rtol)
        # Progress output; warnings are printed regardless.
        say = print if verbose else (lambda *a, **k: None)

        if self._loaded_design is not None:
            return self._run_loaded(seed=seed, verbose=verbose)

        # Normalise algorithm into a list (case-insensitive)
        if isinstance(algorithm, str):
            algorithm_names = [algorithm.strip().lower()]
        else:
            algorithm_names = [a.strip().lower() for a in algorithm]
        if not algorithm_names:
            _fatal("'algorithm' cannot be empty.")

        # ESE is structurally unable to optimise one-dimensional
        # designs: its within-column swap moves (Jin, Chen & Sudjianto
        # 2005) permute rows without changing the design as a point
        # set when d = 1, so it would silently return the initial
        # design. Fail early with an actionable message instead.
        if self.space.n_parameters == 1 and 'ese' in algorithm_names:
            _fatal(
                "Algorithm 'ese' cannot optimise a 1-parameter design: "
                "its column-swap moves leave a 1D point set unchanged "
                "(Jin, Chen & Sudjianto 2005).\n"
                "  Use algorithm='sa' or algorithm='sce' instead."
            )

        # Resolve criterion(ia)
        if criteria is None:
            criteria = default_criterion(self.space)
        if isinstance(criteria, str):
            crit_list  = [get_criterion(criteria)]
            crit_names = [criteria]
        elif isinstance(criteria, list):
            crit_list  = [get_criterion(c) if isinstance(c, str) else c
                          for c in criteria]
            crit_names = [c if isinstance(c, str) else repr(c)
                          for c in criteria]
        else:
            crit_list  = [criteria]
            crit_names = [repr(criteria)]

        if len(crit_list) != 1:
            _fatal(f"run() optimises one criterion; got {crit_names}. "
                   f"Use compare(criteria=[...]) to try several.")
        criterion  = crit_list[0]
        crit_name  = crit_names[0]

        # ── Nominal / criterion compatibility check ────────────────
        # Nominal factors are unordered categorical: distance- and
        # discrepancy-based criteria whose kernels assume ordered
        # levels give meaningless scores on them. Refuse to run
        # such combinations rather than silently produce a bogus
        # design. Ordinal factors are OK for every criterion —
        # their integer scoring preserves an interpretable order.
        if self.space.has_nominal and not getattr(
            criterion, 'supports_nominal', False,
        ):
            supporters = ', '.join(f"'{c}'"
                                   for c in nominal_supporting_criteria())
            nominal_cols = self.space.nominal_names
            _fatal(
                f"Criterion '{crit_name}' does not support nominal "
                f"factors.\n"
                f"  Space has nominal column(s): {nominal_cols}.\n"
                f"  Criteria that support nominal factors: "
                f"{supporters or '(none registered)'}.\n"
                f"  Ordinal factors, however, are accepted by every "
                f"criterion — declare with "
                f"('ordinal', [labels]) if the levels are ordered."
            )

        # Space metadata
        space   = self.space
        n_dims  = space.n_parameters
        gmins   = space.gmins
        granges = space.granges
        names   = space.names
        gs      = space.grid_sampler()

        # Resolve auto settings on focus points
        for fp in self._focus:
            fp.resolve_n_samples(n_dims)

        # Conflict detection
        self._check_conflicts()

        # ── Budget ─────────────────────────────────────────────────────
        n_prescribed_in  = sum(1 for _, in_d, _ in self._prescribed if in_d)
        n_prescribed_out = sum(1 for _, in_d, _ in self._prescribed if not in_d)
        n_focus_in       = sum(fp.n_samples for fp in self._focus if fp.in_design)
        n_focus_out      = sum(fp.n_samples for fp in self._focus if not fp.in_design)

        loeppky = 10 * n_dims
        if self._n_samples is None:
            n_global = loeppky
        else:
            n_global = self._n_samples
            if n_global < loeppky:
                _warn(
                    f"n_samples ({n_global}) < recommended 10*n_parameters "
                    f"({loeppky}, Loeppky et al. 2009). Design quality may "
                    f"be reduced."
                )

        n_optimised_slots = n_global - n_prescribed_in - n_focus_in
        if n_optimised_slots < 1:
            _fatal(
                f"No optimiser slots remaining: n_samples={n_global}, "
                f"prescribed_in={n_prescribed_in}, focus_in={n_focus_in}. "
                f"Increase n_samples or set in_design=False for some points."
            )

        n_validation = (self._n_validation
                        if self._n_validation is not None
                        else max(1, int(np.ceil(n_global * 0.20))))

        n_total_design = n_global + n_prescribed_out + n_focus_out

        # ── Print configuration ────────────────────────────────────────
        say()
        say("═" * 60)
        say("  MERGEN — Space-filling Design")
        say("═" * 60)
        say(f"  Parameters      : {n_dims}")
        say(f"  Candidates      : {space.n_candidates:,}")
        say(f"  n_samples       : {n_global}  "
              f"(prescribed_in={n_prescribed_in}, "
              f"focus_in={n_focus_in}, "
              f"optimised_slots={n_optimised_slots})")
        if n_prescribed_out or n_focus_out:
            say(f"  Extra points    : +{n_prescribed_out + n_focus_out}  "
                  f"(prescribed_out={n_prescribed_out}, "
                  f"focus_out={n_focus_out})")
        say(f"  Total design    : {n_total_design}")
        say(f"  Validation      : {n_validation}")
        say(f"  Criterion       : {crit_name}")
        say(f"  Algorithm(s)    : {', '.join(algorithm_names)}")
        say("─" * 60)

        # ── Reserved set + classify points ─────────────────────────────
        reserved: set = set()

        prescribed_in_pts:  List[np.ndarray] = []
        prescribed_out_pts: List[np.ndarray] = []
        prescribed_sa_pts:  List[np.ndarray] = []
        for pt, in_d, in_s in self._prescribed:
            idx = gs.point_to_index(pt)
            if idx >= 0:
                reserved.add(idx)
            (prescribed_in_pts if in_d else prescribed_out_pts).append(pt)
            if in_s:
                prescribed_sa_pts.append(pt)

        # User-supplied named sets: reserve their grid nodes so the
        # optimiser cannot re-select them.
        for _uname, (upts, _ucolor) in self._user_sets.items():
            for pt in upts:
                idx = gs.point_to_index(pt)
                if idx >= 0:
                    reserved.add(idx)

        # ── Focus / exclusion sampling ─────────────────────────────────
        if self._focus or self._exclusions:
            say("  [MERGEN]   Sampling focus / exclusion regions...",
                flush=True)
        (focus_in_pts, focus_out_pts, focus_sa_pts, repel_weights, reserved,
         focus_sa_in, focus_sa_out) = self._build_focus_exclusion(
            gs, gmins, granges, reserved,
            np.random.default_rng(seed if seed is not None else 44))
        # Delivered rows outside the optimised design; the priority
        # metrics are computed with them, as in quality_report().
        extra_rows = np.asarray(prescribed_out_pts + focus_out_pts,
                                dtype=float).reshape(-1, n_dims)
        fixed_rows = np.asarray(prescribed_in_pts + focus_in_pts
                                + prescribed_out_pts + focus_out_pts,
                                dtype=float).reshape(-1, n_dims)
        # A focus region with fewer free neighbours than requested (it
        # has already warned) hands its unused slots to the optimiser,
        # so the design keeps n_samples rows.
        n_optimised_slots += n_focus_in - len(focus_in_pts)

        # ── Assemble the design layout ─────────────────────────────────
        # Row order in the design matrix:
        #   [ prescribed_in | focus_in | sce_optimised ]
        # n_frozen     = prescribed_in + focus_in (these rows never move)
        # crit_start   = first row that is "visible" to the criterion
        #
        # When a prescribed/focus point has in_optim=True, the criterion
        # includes it; otherwise it is reserved but criterion-blind.
        # We place all "criterion-visible frozen" rows together at the
        # start of the criterion view so a single ``crit_start`` index
        # suffices.

        # Anchor for greedy maximin seeding = criterion-visible frozen rows
        anchor_parts: List[np.ndarray] = []
        in_design_sa_prescribed = [pt for pt, in_d, in_s in self._prescribed
                                   if in_d and in_s]
        if in_design_sa_prescribed:
            anchor_parts.append(np.array(in_design_sa_prescribed, dtype=float))
        # focus_in_pts contains the *sampled* focus points
        # (centres + neighbours).  For anchoring, the centres alone
        # would under-utilise the focus structure; instead we anchor on
        # all sampled in_design + in_optim focus points so the greedy step
        # truly avoids placing optimised points near them.
        # A focus region can yield fewer than n_samples points, so the
        # rows are classified as they are sampled, not by counting.
        focus_sa_in_design_sampled = focus_sa_in
        if focus_sa_in_design_sampled:
            anchor_parts.append(np.array(focus_sa_in_design_sampled,
                                         dtype=float))
        # Criterion-visible points outside the budget (in_design=False,
        # in_optim=True; fill_around's references) come last among the
        # anchors: the optimiser sees them as frozen rows, and each task
        # removes them from the design it returns.
        visible_out = ([pt for pt, in_d, in_s in self._prescribed
                        if not in_d and in_s] + focus_sa_out)
        if visible_out:
            anchor_parts.append(np.array(visible_out, dtype=float))

        _corner_used = False
        if anchor_parts:
            anchor = np.vstack(anchor_parts)
        else:
            # No criterion-visible frozen rows → seed from a corner
            # (Morris & Mitchell 1995 strategy: extreme starting point).
            corner = np.array([v[0] if i % 2 == 0 else v[-1]
                                for i, v in enumerate(space.values)])
            idx = gs.point_to_index(corner)
            if idx in reserved or space.on_grid(corner) < 0:
                # Infeasible or excluded corner: start from the nearest
                # free feasible candidate (lowest index on ties).
                pool  = space.candidate_pool
                free  = ~np.isin(gs.pool_indices(),
                                 np.fromiter(reserved, dtype=np.int64,
                                             count=len(reserved)))
                if not free.any():
                    _fatal("No free feasible grid point is left for the "
                           "design: every candidate is prescribed, "
                           "excluded or in a user-supplied set.")
                dist  = np.where(
                    free,
                    np.linalg.norm((pool - corner) / granges, axis=1),
                    np.inf)
                corner = pool[int(np.argmin(dist))].copy()
                idx    = gs.point_to_index(corner)
            if idx >= 0:
                reserved.add(idx)
            corner_idx   = idx
            anchor       = corner[np.newaxis, :]
            _corner_used = True

        # ── Initial design: delegated to each optimiser ──────────────
        # Each optimiser provides its own prepare_initial_design()
        # method so it can use the type of starting design its
        # underlying algorithm was designed for. The default in
        # BaseOptimizer is a balanced LHS (Joseph, Gul & Ba 2020),
        # which matches what SA, SCE and ESE expect from the
        # literature; subclasses may override for non-LHS algorithms.
        if self._prescribed or self._focus:
            say("  [MERGEN]   Preparing anchor points...", flush=True)

        # Anchors = prescribed_in + focus_in (any explicit user points
        # that must appear unchanged in every optimiser's design).
        if _corner_used:
            # The corner point already lives in ``anchor``; treat it the
            # same as a prescribed point.
            anchors = anchor.copy()
        elif len(anchor) > 0:
            anchors = anchor.copy()
        else:
            anchors = np.empty((0, n_dims))

        # Note: full_design is now constructed *per algorithm* inside
        # the dispatcher loop below, so each optimiser's natural
        # starting design is used.

        # n_frozen: the optimiser must not move these rows. The returned
        # designs lack the visible out-of-budget rows (n_kept frozen).
        n_kept   = len(prescribed_in_pts) + len(focus_in_pts)
        n_frozen = n_kept + len(visible_out)

        # crit_start: first row visible to the criterion
        #   Rows are: [criterion-blind frozen | criterion-visible frozen | optimised]
        # The number of criterion-blind frozen rows is the count of
        # prescribed/focus points that are in_design but NOT in_optim.
        n_blind = (len(prescribed_in_pts) - len(in_design_sa_prescribed)
                   + len(focus_in_pts) - len(focus_sa_in_design_sampled))
        crit_start = n_blind

        # Snapshot the initial reserved set so each algorithm starts
        # from the same anchor state.
        initial_reserved = reserved.copy()

        # ── Run the requested optimiser(s) ──────────────────────────────
        from .algorithms import get_optimizer

        # Independent, reproducible seeds for the repeats, derived from
        # the base seed via NumPy's SeedSequence.spawn. This is the SAME
        # derivation compare() relies on, so that run(seed=s,
        # n_repeats=k) for one (criterion, algorithm) reproduces exactly
        # what compare(seed=s, n_repeats=k) evaluates for that same
        # combination — the two entry points never disagree.
        auto_repeats = isinstance(n_repeats, str)
        if auto_repeats:
            if n_repeats != 'auto':
                _fatal(f"n_repeats must be a positive integer or 'auto'; "
                       f"got {n_repeats!r}")
            round_size = _AUTO_ROUND
        else:
            if (isinstance(n_repeats, (bool, np.bool_))
                    or not isinstance(n_repeats, numbers.Integral)
                    or n_repeats < 1):
                _fatal(f"n_repeats must be a positive integer or 'auto'; "
                       f"got {n_repeats!r}")
            round_size = int(n_repeats)
        _base_seed = seed if seed is not None else 44
        _seed_seq  = np.random.SeedSequence(_base_seed)
        rep_seeds  = [int(cs.generate_state(1)[0])
                      for cs in _seed_seq.spawn(round_size)]

        # Resolve n_jobs and decide whether to parallelise.
        # Parallelism applies across independent (algorithm, repeat)
        # jobs; the per-algorithm restart loop inside one job is
        # ILS-based and intentionally serial (each restart kicks from the
        # previous best — Lourenco, Martin & Stutzle (2003)).
        n_jobs_resolved = _resolve_n_jobs(n_jobs)
        n_algos         = len(algorithm_names)
        n_units         = n_algos * round_size
        effective_n_jobs = min(n_jobs_resolved, n_units)
        run_in_parallel = effective_n_jobs > 1 and n_units > 1

        # Validate all algorithm names up front (better error than
        # discovering a typo three workers in).
        for alg_name in algorithm_names:
            try:
                get_optimizer(alg_name)
            except KeyError:
                _fatal(
                    f"Optimiser {alg_name!r} is not registered. "
                    f"Available: "
                    f"{sorted(get_optimizer.__globals__.get('_OPTIMIZER_REGISTRY', {}).keys())}"
                )
                raise

        # ── Progress messages ──
        if verbose:
            algos_str = ", ".join(algorithm_names)
            if run_in_parallel:
                msg = (f"Optimising in parallel (criterion={crit_name}, "
                       f"algorithms={algos_str}, n_jobs={effective_n_jobs})...")
                if n_jobs_resolved > effective_n_jobs:
                    extra = n_jobs_resolved - effective_n_jobs
                    msg += (f"  [note: {n_jobs_resolved} workers requested, "
                            f"only {effective_n_jobs} needed; "
                            f"{extra} unused]")
                _ok(msg)
            else:
                if n_jobs_resolved > 1 and n_algos == 1:
                    _ok(
                        f"n_jobs={n_jobs_resolved} requested but only one "
                        f"algorithm to dispatch; running sequentially "
                        f"(per-algorithm restart loop is ILS-based and "
                        f"cannot be parallelised)."
                    )
                _ok(f"Optimising (criterion={crit_name}, "
                    f"algorithm{'s' if n_algos > 1 else ''}={algos_str})...")

        t_run_start = time.perf_counter()

        # Per-algorithm results
        algorithm_results: Dict[str, Any] = {}

        # Build the per-algorithm task arguments once; reused for both
        # the sequential and parallel paths so they share a single code
        # path for the heavy lifting.
        budget   = n_optimised_slots - (1 if _corner_used else 0)

        def task_args(alg_name, rep_seed, budget_factor):
            return dict(
                alg_name                    = alg_name,
                params                      = self._optimizer_params.get(alg_name, {}),
                anchors                     = anchors.copy(),
                budget                      = budget,
                initial_reserved            = set(initial_reserved),
                space                       = space,
                criterion                   = criterion,
                prescribed_in_pts           = prescribed_in_pts,
                focus_in_pts                = focus_in_pts,
                in_design_sa_prescribed     = in_design_sa_prescribed,
                focus_sa_in_design_sampled  = focus_sa_in_design_sampled,
                n_dims                      = n_dims,
                n_frozen                    = n_frozen,
                crit_start                  = crit_start,
                seed                        = rep_seed,
                budget_factor               = budget_factor,
                visible_out                 = len(visible_out),
                # In parallel mode, suppress per-task verbose output so
                # worker streams do not interleave.
                verbose                     = (verbose and not run_in_parallel
                                               and round_size == 1
                                               and not auto_repeats),
            )

        def run_tasks(tasks_args):
            if not (effective_n_jobs > 1 and len(tasks_args) > 1):
                return [_run_one_algorithm_task(**ta) for ta in tasks_args]
            from joblib import Parallel, delayed
            # Workers spawn fresh Python processes that re-import mergen;
            # set MERGEN_SILENT in the child env so each worker does not
            # print the package banner on import.
            _prev_silent = os.environ.get('MERGEN_SILENT')
            os.environ['MERGEN_SILENT'] = '1'
            try:
                # loky (joblib's default) on every platform: fork from a
                # multi-threaded process can deadlock, and loky's
                # cloudpickle also carries lambda constraints. The
                # context manager shuts the pool down on exit/Ctrl-C.
                with Parallel(n_jobs=min(effective_n_jobs, len(tasks_args)),
                              backend='loky') as parallel:
                    return parallel(
                        delayed(_run_one_algorithm_task)(**ta)
                        for ta in tasks_args)
            finally:
                if _prev_silent is None:
                    os.environ.pop('MERGEN_SILENT', None)
                else:
                    os.environ['MERGEN_SILENT'] = _prev_silent

        # A design and its mirror images along the axes where the whole
        # problem is symmetric are equally good. Each repeat is stored in
        # one canonical form of that family, so the delivered design does
        # not depend on which member a seed happened to find. The seed
        # corner is an ordinary optimised row.
        banned = set(initial_reserved)
        if _corner_used:
            banned.discard(corner_idx)
        sa_set   = {tuple(p) for p in in_design_sa_prescribed}
        foc_set  = {tuple(p) for p in focus_sa_in_design_sampled}
        frozen_groups = [
            [p for p in prescribed_in_pts if tuple(p) not in sa_set],
            [p for p in focus_in_pts if tuple(p) not in foc_set],
            in_design_sa_prescribed, focus_sa_in_design_sampled,
            visible_out]
        # A reference off the grid has no grid image: no mirroring then.
        mirror_axes = ([] if any(gs.point_to_index(p) < 0
                                 for p in visible_out)
                       else _mirror_axes(space, gs, banned, frozen_groups))
        visible_rows = np.asarray(visible_out, dtype=float).reshape(-1, n_dims)

        def canonical(res):
            from dataclasses import replace
            if not np.isfinite(res.score):
                # A design handed over by compare()'s polishing: score it.
                res = replace(res, score=float(criterion.evaluate(
                    _normalised(np.vstack([np.asarray(res.design)[crit_start:],
                                           visible_rows]), space), space)))
            D = _canonical_design(res.design, mirror_axes, gs, n_kept)
            s_new = float(criterion.evaluate(
                _normalised(np.vstack([D[crit_start:], visible_rows]),
                            space), space))
            if not _same_values([s_new], [res.score]):
                # The criterion is not symmetric here: keep the design
                # as found, rows sorted.
                D = _canonical_design(res.design, [], gs, n_kept)
            return replace(res, design=D)

        def values(res):
            v = [res.score]
            if priority:
                v += list(_priority_values(res.design, priority, space,
                                           extra_rows))
            return v

        # Budget scaling applies to the built-in optimisers, through the
        # parameter that sets their search length, unless the user set it.
        def scalable(alg_name):
            key = _BUDGET_PARAM.get(alg_name)
            return (key is not None
                    and key not in self._optimizer_params.get(alg_name, {}))

        # Work of one repeat in units of a repeat at the default search
        # length, so the work limit means the same with a user budget.
        n_rows = (len(prescribed_in_pts) + len(focus_in_pts)
                  + len(visible_out) + n_optimised_slots)

        def repeat_cost(alg_name):
            key = _BUDGET_PARAM.get(alg_name)
            params = self._optimizer_params.get(alg_name, {})
            if key is None or params.get(key) is None:
                return 1.0
            if alg_name == 'sa':
                default = max(2000, 100 * n_rows)
            else:
                default = get_optimizer(alg_name).get_default_params()[key]
            if default is None:
                # ESE sizes Q from the design; its default run uses at
                # least two outer iterations.
                default = 2
            return float(params[key]) / default

        # Repeat k of every algorithm uses the k-th seed spawned from
        # ``seed``. With an integer n_repeats exactly that many repeats
        # run. With 'auto', rounds of _AUTO_ROUND repeats run up to the
        # work limit, stopping earlier once the selected result has been
        # found _AUTO_HITS times; while no
        # other repeat has reproduced the best result, each round doubles
        # the budget of the next, and the work stops at
        # _AUTO_MAX_EFFORT repeats at the
        # default budget. Rounds have a fixed size and the test runs
        # between them, so the result is the same for any n_jobs. The
        # full repeat list (spawn order) is kept so compare() can reuse
        # the very same optimisations.
        _repeats_by_alg: Dict[str, list] = {a: [] for a in algorithm_names}
        rep_values: Dict[str, list] = {a: [] for a in algorithm_names}
        factor  = {a: 1 for a in algorithm_names}
        effort  = {a: 0 for a in algorithm_names}
        repeat_info: Dict[str, dict] = {}
        pending = list(algorithm_names)
        while pending:
            need = max(len(_repeats_by_alg[a]) for a in pending) + round_size
            while len(rep_seeds) < need:
                rep_seeds += [int(cs.generate_state(1)[0])
                              for cs in _seed_seq.spawn(round_size)]
            batch = [(a, rep_seeds[len(_repeats_by_alg[a]) + i])
                     for a in pending for i in range(round_size)]
            # compare() hands over the repeats it already ran for the
            # winner (same seeds, same settings), so best_result does
            # not optimise them a second time.
            reuse = getattr(self, '_reuse_repeats', None)
            if reuse is not None and all(
                    len(reuse.get(a, ())) >= len(_repeats_by_alg[a])
                    + round_size for a in pending):
                results_list = [reuse[a][len(_repeats_by_alg[a]) + i]
                                for a in pending for i in range(round_size)]
            else:
                results_list = run_tasks([task_args(a, s, factor[a])
                                          for a, s in batch])
            for (a, _s), res in zip(batch, results_list):
                res = canonical(res)
                _repeats_by_alg[a].append(res)
                rep_values[a].append(values(res))
            still = []
            for a in pending:
                reps, vals = _repeats_by_alg[a], rep_values[a]
                effort[a] += round_size * factor[a] * repeat_cost(a)
                win = _select_repeat(reps, priority, space, selection,
                                     tol_arr, rtol_arr, extra_rows)
                algorithm_results[a] = win
                k = next(i for i, r in enumerate(reps) if r is win)
                hits = sum(_same_values(v, vals[k]) for v in vals)
                # How far the other repeats' criterion scores lie from
                # the delivered one, relative: the measured seed effect.
                gaps = [abs(r.score / win.score - 1) for r in reps
                        if r is not win and win.score > 0]
                repeat_info[a] = dict(n_repeats=len(reps), hits=hits,
                                      budget_factor=factor[a],
                                      converged=hits >= _AUTO_HITS,
                                      gap_median=(float(np.median(gaps))
                                                  if gaps else 0.0),
                                      gap_max=max(gaps, default=0.0))
                if not auto_repeats or hits >= _AUTO_HITS:
                    continue
                # A best result no other repeat has reproduced means the
                # search length is too short to settle: double it.
                # A doubled round that would pass the work limit runs at
                # the current length instead.
                nxt = factor[a] * (2 if hits < 2 and scalable(a) else 1)
                if (effort[a] + round_size * nxt * repeat_cost(a)
                        > _AUTO_MAX_EFFORT):
                    nxt = factor[a]
                if (effort[a] + round_size * nxt * repeat_cost(a)
                        > _AUTO_MAX_EFFORT
                        or len(reps) + round_size > _AUTO_MAX_REPEATS):
                    continue
                factor[a] = nxt
                still.append(a)
            pending = still

        if verbose:
            for alg_name in algorithm_names:
                result = algorithm_results[alg_name]
                info   = repeat_info[alg_name]
                suffix = ""
                if info['n_repeats'] > 1:
                    suffix = (f"  (best of {info['n_repeats']}, found "
                              f"{info['hits']} times")
                    if info['budget_factor'] > 1:
                        suffix += f", budget x{info['budget_factor']}"
                    suffix += ")"
                _ok(f"{alg_name:<6} done -- score={result.score:.4g}  "
                    f"(elapsed {self._fmt_time(result.elapsed)}){suffix}")
        if auto_repeats:
            for alg_name in algorithm_names:
                info = repeat_info[alg_name]
                if not info['converged']:
                    _warn(f"{alg_name}: the best result was found by "
                          f"{info['hits']} of {info['n_repeats']} repeats "
                          f"before the work limit; the other repeats' "
                          f"criterion scores differ from it by "
                          f"{_gap_text(info)}, so another seed may give "
                          f"a different, better or worse design. "
                          f"Raise the optimiser budget with "
                          f"set_optimizer() to reduce this.")

        t_elapsed = time.perf_counter() - t_run_start

        # Pick the best result across all algorithms: the lowest score,
        # or, with a priority, the same selection rule as for repeats.
        # Equal scores go to the algorithm name, not the list order.
        best_algorithm = min(algorithm_results,
                             key=lambda k: (algorithm_results[k].score, k))
        if priority and len(algorithm_results) > 1:
            from . import metrics as _m
            from .selection import select
            _names = sorted(algorithm_results)
            _sel = select(
                np.array([_priority_values(algorithm_results[a].design,
                                           priority, space, extra_rows)
                          for a in _names]),
                [m in _m._HIGHER_BETTER for m in priority],
                rule=selection, tol=tol_arr, rtol=rtol_arr,
                tiebreak=[algorithm_results[a].score for a in _names])
            best_algorithm = _names[_sel.winner]
        best_design    = algorithm_results[best_algorithm].design
        best_score     = algorithm_results[best_algorithm].score

        if verbose and len(algorithm_names) > 1:
            _ok(f"Best: {best_algorithm}  score={best_score:.4g}  "
                f"(total elapsed {self._fmt_time(t_elapsed)})")

        # Reconstruct reserved set from the final design (used downstream
        # for validation / extra-set maximin selection). Start from
        # the initial reserved (anchors only) and add the optimiser's
        # output rows.
        best_reserved = initial_reserved.copy()
        if _corner_used:
            # The seed corner is an ordinary optimised row: once the
            # optimiser has moved it, its node is free again.
            best_reserved.discard(corner_idx)
        for row in best_design[n_kept:]:
            idx = gs.point_to_index(row)
            if idx >= 0:
                best_reserved.add(idx)

        final_design   = best_design
        final_reserved = best_reserved

        # ── Validation set (maximin set) ─────────────────────────────
        # Held-out points keep away from every delivered point: the
        # design, out-of-budget prescribed and focus points, and the
        # user-supplied sets.
        cur_ref = np.vstack(
            [final_design,
             np.asarray(prescribed_out_pts + focus_out_pts,
                        dtype=float).reshape(-1, n_dims)]
            + [upts for upts, _ucolor in self._user_sets.values()])
        val_pts = self._kennard_stone(
            gs, final_reserved, cur_ref, n_validation,
        )
        cur_reserved = final_reserved.copy()
        for vp in val_pts:
            idx = gs.point_to_index(vp)
            if idx >= 0:
                cur_reserved.add(idx)
        if len(val_pts):
            cur_ref = np.vstack([cur_ref, val_pts])

        # ── Extra sets (maximin sets) ─────────────────────────────────
        extra_dfs: Dict[str, pd.DataFrame] = {}
        for uname, (upts, ucolor) in self._user_sets.items():
            df_u = pd.DataFrame(upts, columns=names)
            df_u['point_type'] = uname
            if ucolor is not None:
                df_u['color'] = ucolor
            df_u.index.name = 'id'
            extra_dfs[uname] = df_u

        if self._extra_sets:
            for set_name, set_n in self._extra_sets.items():
                set_pts = self._kennard_stone(
                    gs, cur_reserved, cur_ref, set_n,
                )
                for sp in set_pts:
                    idx = gs.point_to_index(sp)
                    if idx >= 0:
                        cur_reserved.add(idx)
                cur_ref = (np.vstack([cur_ref, set_pts])
                           if len(set_pts) else cur_ref)
                df_set = (pd.DataFrame(set_pts, columns=names)
                          if len(set_pts) else pd.DataFrame(columns=names))
                df_set['point_type'] = set_name
                df_set.index.name = 'id'
                extra_dfs[set_name] = df_set

        # ── Append out-of-budget extras ────────────────────────────────
        extra_fixed:  List[np.ndarray] = []
        extra_labels: List[str]        = []
        for pt in prescribed_out_pts:
            extra_fixed.append(pt)
            extra_labels.append(SamplingResult.PRESCRIBED)
        for pt in focus_out_pts:
            extra_fixed.append(pt)
            extra_labels.append(SamplingResult.FOCUS)

        # ── Build the main DataFrame ───────────────────────────────────
        n_p_in = len(prescribed_in_pts)
        n_f_in = len(focus_in_pts)
        n_sce  = len(final_design) - n_p_in - n_f_in

        # Row order: criterion-blind prescribed and focus rows, then the
        # criterion-visible ones, then the optimised rows.
        n_p_vis = len(in_design_sa_prescribed)
        n_f_vis = len(focus_sa_in_design_sampled)
        labels = (
            [SamplingResult.PRESCRIBED] * (n_p_in - n_p_vis)
            + [SamplingResult.FOCUS]    * (n_f_in - n_f_vis)
            + [SamplingResult.PRESCRIBED] * n_p_vis
            + [SamplingResult.FOCUS]    * n_f_vis
            + [SamplingResult.OPTIMISED]* n_sce
        )

        if extra_fixed:
            final_with_extra = np.vstack(
                [final_design, np.array(extra_fixed, dtype=float)]
            )
            labels += extra_labels
        else:
            final_with_extra = final_design

        df_samples = pd.DataFrame(final_with_extra, columns=names)
        df_samples['point_type'] = labels
        df_samples.index.name = 'id'

        df_val = (pd.DataFrame(val_pts, columns=names)
                  if len(val_pts)
                  else pd.DataFrame(columns=names + ['point_type']))
        if len(df_val):
            df_val['point_type'] = SamplingResult.VALIDATION
            df_val.index.name = 'id'

        # ── Final status ───────────────────────────────────────────────
        say()
        say("─" * 60)
        say("  MERGEN — Final Design")
        say("─" * 60)
        say(f"  Prescribed (in)  : {n_p_in}")
        say(f"  Prescribed (out) : {len(prescribed_out_pts)}")
        say(f"  Focus (in)       : {n_f_in}")
        say(f"  Focus (out)      : {len(focus_out_pts)}")
        say(f"  Optimised        : {n_sce}")
        say(f"  Total design     : {len(df_samples)}")
        say(f"  Validation       : {len(df_val)}")
        for sn, sdf in extra_dfs.items():
            say(f"  {sn:<18}: {len(sdf)}")
        say("═" * 60)

        result                    = SamplingResult(df_samples, df_val, space)
        result.sets               = {k: space.decode_frame(v)
                                     for k, v in extra_dfs.items()}
        result.designs            = {crit_name: result.samples}
        result.algorithm_results  = algorithm_results
        result.best_algorithm     = best_algorithm
        # All repeat results per algorithm (spawn order), so compare()
        # can reuse the exact same optimisations it would trigger anyway.
        result._repeats_by_alg    = _repeats_by_alg
        result._repeat_info       = repeat_info
        # Delivered rows outside the optimised design, for compare().
        result._extra_pts         = extra_rows
        result._meta              = {
            'criteria'        : crit_name,
            'criterion_obj'   : criterion,
            'seed'            : seed,
            'algorithm'       : (algorithm_names[0] if len(algorithm_names) == 1
                                 else algorithm_names),
            'best_algorithm'  : best_algorithm,
            'n_repeats'       : n_repeats,
            'n_parameters'    : n_dims,
            'n_candidates'    : space.n_candidates,
            'n_design'        : len(df_samples),
            'n_validation'    : len(df_val),
            'best_log_score'  : float(np.log(max(best_score, _EPS))),
            'elapsed_sec'     : float(t_elapsed),
            # Exclusion centres, which the baseline of quality_report()
            # leaves out of the random designs.
            'exclusions'      : [ep.point.tolist() for ep in self._exclusions],
            # Delivered rows the optimiser did not choose; the quality
            # baseline completes them at random, from the candidates the
            # optimiser could use.
            'fixed_points'    : fixed_rows.tolist(),
            'baseline_reserved': self._baseline_reserved(),
        }
        return result

    def _run_loaded(
        self,
        seed:    Optional[int],
        verbose: bool,
    ) -> SamplingResult:
        """
        Build a :class:`SamplingResult` from an externally loaded design.

        No optimisation is performed. The loaded points are reserved on
        the grid, then the validation set and any extra sets are
        generated around them as maximin sets, exactly as
        in :meth:`run`.
        """
        if self._prescribed or self._focus or self._exclusions:
            _fatal(
                "load_design cannot be combined with add_prescribed, "
                "add_focus or add_exclusion: the loaded design is fixed. "
                "Use mergen.sequential.extend to grow a design."
            )
        if self._n_samples is not None:
            _fatal(
                "load_design cannot be combined with n_samples: the "
                "design size is fixed by the loaded points. Use "
                "mergen.sequential.extend to grow a design."
            )
        self._check_conflicts()

        t_start = time.perf_counter()
        space   = self.space
        pts, label, lcolor = self._loaded_design
        say = print if verbose else (lambda *a, **k: None)

        names   = space.names
        gs      = space.grid_sampler()
        n_dims  = space.n_parameters

        n_validation = (self._n_validation
                        if self._n_validation is not None
                        else max(1, int(np.ceil(len(pts) * 0.20))))

        say()
        say("═" * 60)
        say("  MERGEN — Loaded Design")
        say("═" * 60)
        say(f"  Parameters      : {n_dims}")
        say(f"  Candidates      : {space.n_candidates:,}")
        say(f"  Loaded points   : {len(pts)}  (label='{label}')")
        say(f"  Validation      : {n_validation}")
        say("─" * 60)

        # Reserve loaded points so validation / extra sets avoid them.
        reserved: set = set()
        for pt in pts:
            idx = gs.point_to_index(pt)
            if idx >= 0:
                reserved.add(idx)
        for _uname, (upts, _ucolor) in self._user_sets.items():
            for pt in upts:
                idx = gs.point_to_index(pt)
                if idx >= 0:
                    reserved.add(idx)

        # ── Validation set (maximin set) ─────────────────────────────
        cur_ref = np.vstack(
            [pts] + [upts for upts, _ucolor in self._user_sets.values()])
        val_pts = self._kennard_stone(
            gs, reserved, cur_ref, n_validation,
        )
        cur_reserved = reserved.copy()
        for vp in val_pts:
            idx = gs.point_to_index(vp)
            if idx >= 0:
                cur_reserved.add(idx)
        if len(val_pts):
            cur_ref = np.vstack([cur_ref, val_pts])

        # ── Extra sets: user-supplied first, then generated ────────────
        extra_dfs: Dict[str, pd.DataFrame] = {}
        for uname, (upts, ucolor) in self._user_sets.items():
            df_u = pd.DataFrame(upts, columns=names)
            df_u['point_type'] = uname
            if ucolor is not None:
                df_u['color'] = ucolor
            df_u.index.name = 'id'
            extra_dfs[uname] = df_u
        if self._extra_sets:
            for set_name, set_n in self._extra_sets.items():
                set_pts = self._kennard_stone(
                    gs, cur_reserved, cur_ref, set_n,
                )
                for sp in set_pts:
                    idx = gs.point_to_index(sp)
                    if idx >= 0:
                        cur_reserved.add(idx)
                cur_ref = (np.vstack([cur_ref, set_pts])
                           if len(set_pts) else cur_ref)
                df_set = (pd.DataFrame(set_pts, columns=names)
                          if len(set_pts) else pd.DataFrame(columns=names))
                df_set['point_type'] = set_name
                df_set.index.name = 'id'
                extra_dfs[set_name] = df_set

        # ── Assemble result ────────────────────────────────────────────
        df_samples = pd.DataFrame(pts, columns=names)
        df_samples['point_type'] = label
        if label != SamplingResult.OPTIMISED:
            df_samples['color'] = lcolor if lcolor is not None else "#3a86ff"
        df_samples.index.name = 'id'

        df_val = (pd.DataFrame(val_pts, columns=names)
                  if len(val_pts)
                  else pd.DataFrame(columns=names + ['point_type']))
        if len(df_val):
            df_val['point_type'] = SamplingResult.VALIDATION
            df_val.index.name = 'id'

        t_elapsed = time.perf_counter() - t_start

        say()
        say("─" * 60)
        say("  MERGEN — Final Design")
        say("─" * 60)
        say(f"  {label:<17}: {len(df_samples)}")
        say(f"  Total design     : {len(df_samples)}")
        say(f"  Validation       : {len(df_val)}")
        for sn, sdf in extra_dfs.items():
            say(f"  {sn:<18}: {len(sdf)}")
        say("═" * 60)

        result                    = SamplingResult(df_samples, df_val, space)
        result.sets               = {k: space.decode_frame(v)
                                     for k, v in extra_dfs.items()}
        result.designs            = {'loaded': result.samples}
        result.algorithm_results  = {}
        result.best_algorithm     = 'loaded'
        result._meta              = {
            'criteria'        : 'loaded',
            'seed'            : seed,
            'algorithm'       : 'loaded',
            'best_algorithm'  : 'loaded',
            'n_parameters'    : n_dims,
            'n_candidates'    : space.n_candidates,
            'n_design'        : len(df_samples),
            'n_validation'    : len(df_val),
            'best_log_score'  : float('nan'),
            'elapsed_sec'     : float(t_elapsed),
            # Exclusion centres, which the baseline of quality_report()
            # leaves out of the random designs.
            'exclusions'      : [ep.point.tolist() for ep in self._exclusions],
            # Nothing was optimised: the baseline stays unconditioned.
            'fixed_points'    : [],
        }
        return result

    def _baseline_reserved(self) -> List[list]:
        """Grid points the optimiser may not use besides exclusions:
        add_set() points and focus centres with include_center=False."""
        pts = [list(map(float, p)) for upts, _c in self._user_sets.values()
               for p in upts]
        pts += [list(map(float, fp.point)) for fp in self._focus
                if fp.include_center is False]
        return pts

    def compare(
        self,
        criteria:   Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None,
        priority:   Tuple[str, ...] = ('min_distance', 'max_abs_correlation'),
        mc_samples: int = 300,
        seed:       int = 44,
        n_repeats:  Union[int, str] = 'auto',
        n_jobs:     Optional[int] = -1,
        verbose:    bool = True,
        selection:  str = 'pareto',
        tol:        Union[float, Dict[str, float]] = 1e-9,
        rtol:       Union[float, Dict[str, float]] = 0.0,
        polish:     bool = True,
        polish_kicks: int = 0,
    ) -> "ComparisonResult":
        """
        Run a criterion/algorithm sweep and select the best design.

        Every (criterion, algorithm) combination is optimised with the
        current sampler configuration. Raw criterion scores are on
        incomparable scales, so the designs are compared on
        criterion-agnostic quality metrics, computed on normalised
        coordinates. The winner is chosen on the raw values of the
        ``priority`` metrics by the ``selection`` rule (see
        :mod:`mergen.selection`). Percentile ranks against a shared
        Monte Carlo baseline of random designs of the same size are
        reported alongside as a description of how each design compares
        with random designs; they are not used for selection, because
        well-optimised designs all saturate at the top of that scale.

        Parameters
        ----------
        criteria : list of str, optional
            Criteria to sweep. If None, all criteria compatible with the
            space are used: nominal-supporting criteria when the space
            contains a nominal factor, all remaining criteria otherwise.
            An explicit list is used as given (with a warning if it
            mixes in criteria that do not match the factor types).
        algorithms : list of str, optional
            Optimisers to sweep. If None, ``['sa']``.
        priority : tuple of str
            Quality metrics that decide the winner. A single metric may
            be given as a string. Available metrics: min_distance,
            minimax, max_abs_correlation, projection_cd2, cd2.
            mean_distance and cv_distances are reported but cannot
            decide, as a design can improve them without filling the
            space better.
        mc_samples : int, default 300
            Size of the shared Monte Carlo baseline used for the
            percentile columns. ``0`` skips it (the columns are then
            NaN; the selection is unaffected).
        seed : int, default 44
            Base seed. All randomness (the repeats and the Monte Carlo
            baseline) is derived from it, so the whole comparison
            reproduces exactly from this one number. ``None`` is the
            same as 44.
        n_repeats : int or 'auto', default 'auto'
            Number of independent optimiser runs per combination, as in
            :meth:`run` (``'auto'``: up to a work cap, stopping early
            only when the best result has been found by 5 repeats,
            which is rare; see there and :attr:`ComparisonResult.repeat_info`).
            Each
            run uses a distinct seed derived from ``seed`` via
            NumPy's SeedSequence.spawn (independent, not consecutive
            integers). The representative repeat of each combination is
            chosen with the same ``priority``, ``selection`` and
            tolerances, exactly as ``run(seed, n_repeats, priority,
            selection, tol, rtol)`` would deliver it.
        n_jobs : int or None, default -1
            Number of parallel workers across the combinations,
            following the joblib convention (-1 -> all cores, the
            default; None or 1 -> one worker). Results are identical
            regardless of n_jobs.
        verbose : bool, default True
            Print progress and the final table.
        selection : ``'pareto'`` or ``'lexicographic'``, default ``'pareto'``
            ``'lexicographic'`` applies the priority metrics in the
            order listed: the first decides, and each later metric only
            decides among designs tied on the earlier ones within the
            tolerance. ``'pareto'`` treats them as unordered, competing
            objectives: the non-dominated designs are kept and the one
            closest to the Utopia point is chosen.
        tol, rtol : float or dict, optional
            Absolute and relative tolerance under which two metric
            values count as equal: ``|a - b| <= tol + rtol * max(|a|,
            |b|)``. Either a number for every metric or a dict
            ``{metric: value}``. The defaults (``1e-9``, ``0``) absorb
            floating-point noise only. Metrics are computed on
            normalised coordinates, so ``tol={'min_distance': 0.01}``
            treats designs whose minimum distances differ by less than
            1 % of the parameter range as equally good.
        polish : bool, default True
            Polish the design of every combination on the priority
            metrics before the selection (:mod:`mergen.polish`): rows
            are moved one at a time to the free feasible grid node that
            improves the priority most, until no single move helps.
            Prescribed and focus rows stay. A polished design is never
            worse than the design it started from (beyond ``tol``) on
            the first priority metric under ``'lexicographic'``, and on
            every priority metric under ``'pareto'``. A polished design
            that differs from its start is added to the table as
            ``'<algorithm>+polish'`` and competes under the same
            selection rule. The winner is therefore never worse on the
            priority than without polishing under ``'lexicographic'``;
            under ``'pareto'`` it is never dominated by the winner
            without polishing, though the added designs move the ideal
            point and can shift the choice along the front. With
            ``min_distance`` in the priority, the exchange alternates
            with moves that lower the Morris-Mitchell phi_p. The
            result is a local optimum, not a proven global one. On
            five test spaces the winner's first priority metric lay,
            from the best known value, a median 0 % (at most 11 %) for
            ``minimax``, 1 % (5 %) for ``cd2``, 3 % (12 %) for
            ``min_distance`` under ``'lexicographic'`` and 10 % (38 %)
            under the default Pareto priority; ``min_distance`` is a
            known limit, and compare() warns when it is a priority
            metric.
            ``best_result.summary()`` names the combination a polished
            winner started from. ``False`` compares the optimiser
            designs only.
        polish_kicks : int, default 0
            Iterated-local-search kicks after the exchange: three random
            rows are moved and the exchange runs again; the result is
            kept only if it is better. Kicks follow the seed, so the
            design does not depend on the machine, but each kick costs
            about as much as the first exchange, which can be minutes
            on large candidate sets.

        Returns
        -------
        ComparisonResult
            ``.best`` (winning key), ``.best_result`` (its full
            SamplingResult), ``.metrics`` (raw values), ``.table``
            (percentile ranks), ``.ties``, ``.front`` and
            :meth:`~ComparisonResult.reselect`.
        """
        if (isinstance(polish_kicks, bool)
                or not isinstance(polish_kicks, (int, np.integer))
                or polish_kicks < 0):
            _fatal(f"polish_kicks must be a non-negative integer; got "
                   f"{polish_kicks!r}")
        if self._loaded_design is not None:
            _fatal("compare() cannot run on a loaded design: load_design() "
                   "makes run() return the loaded points without "
                   "optimising. Compare on a Sampler without "
                   "load_design(), or use mergen.sequential.extend().")

        # ── Resolve the sweep lists ────────────────────────────────────
        has_nominal = any(self.space.is_nominal(p) for p in self.space.names)
        nominal_ok  = set(nominal_supporting_criteria())
        if criteria is None:
            if has_nominal:
                crit_list = sorted(nominal_ok)
            else:
                crit_list = sorted(set(list_criteria()) - nominal_ok)
        else:
            if isinstance(criteria, str):
                criteria = [criteria]
            if not all(isinstance(c, str) for c in criteria):
                _fatal(f"criteria must be names (str); got {criteria!r}")
            crit_list = [c.strip().lower() for c in criteria]
            if has_nominal:
                bad = [cr for cr in crit_list if cr not in nominal_ok]
                if bad:
                    _warn(f"Criteria {bad} do not support nominal factors; "
                          f"their scores may be meaningless for this space.")
            else:
                bad = [cr for cr in crit_list if cr in nominal_ok]
                if bad:
                    _warn(f"Criteria {bad} target nominal factors, which "
                          f"this space does not contain; their extra terms "
                          f"are dead weight here.")
        if isinstance(algorithms, str):
            algorithms = [algorithms]
        if algorithms is not None and not all(isinstance(a, str)
                                              for a in algorithms):
            _fatal(f"algorithms must be names (str); got {algorithms!r}")
        alg_list = ([a.strip().lower() for a in algorithms]
                    if algorithms is not None else ['sa'])
        from .criteria import _REGISTRY as _CRIT
        for label, lst in (('criteria', crit_list), ('algorithms', alg_list)):
            if not lst:
                _fatal(f"{label} cannot be empty.")
            # Aliases of one criterion (phi_p / phip) count as the same.
            key = ((lambda x: _CRIT.get(x, x)) if label == 'criteria'
                   else (lambda x: x))
            seen, kept, dupes = set(), [], []
            for x in lst:
                if key(x) in seen:
                    dupes.append(x)
                else:
                    seen.add(key(x))
                    kept.append(x)
            if dupes:
                _warn(f"{label} lists {sorted(set(dupes))} more than once "
                      f"(or as an alias); each is run once.")
            lst[:] = kept

        priority = _check_priority(priority, allow_none=False)
        _check_selection(selection, priority, tol, rtol)
        if not (n_repeats == 'auto'
                or (isinstance(n_repeats, numbers.Integral)
                    and not isinstance(n_repeats, (bool, np.bool_))
                    and n_repeats >= 1)):
            _fatal(f"n_repeats must be a positive integer or 'auto'; "
                   f"got {n_repeats!r}")
        if (not isinstance(mc_samples, (int, np.integer))
                or isinstance(mc_samples, bool) or mc_samples < 0):
            _fatal(f"mc_samples must be a non-negative integer; "
                   f"got {mc_samples!r}")

        # One job per (criterion, algorithm) combination. Each job calls
        # Sampler.run(seed, n_repeats), which owns the single seed
        # derivation (SeedSequence(seed).spawn) and returns every repeat.
        # compare() therefore triggers exactly the optimisations a
        # follow-up run() would, so their designs always agree.
        combos = [(cr, alg) for cr in crit_list for alg in alg_list]

        def only_priority(arg):
            if isinstance(arg, Mapping):
                return {m: v for m, v in arg.items() if m in priority}
            return arg
        job_sel = (priority, selection, only_priority(tol),
                   only_priority(rtol))

        effective_n_jobs = _resolve_n_jobs(n_jobs)
        # job_results: list of ((cr, alg), [repeat OptimisationResults])
        job_results = []
        settle: Dict[Tuple[str, str], dict] = {}
        if effective_n_jobs > 1 and len(combos) > 1:
            from joblib import Parallel, delayed
            _prev_silent = os.environ.get('MERGEN_SILENT')
            os.environ['MERGEN_SILENT'] = '1'
            n_combos = len(combos)
            # Cores the combinations leave idle run their repeats.
            inner_jobs = max(1, effective_n_jobs // n_combos)
            # loky (joblib's default) on every platform: fork from a
            # multi-threaded process can deadlock, and loky's cloudpickle
            # also carries lambda constraints. It streams, so each
            # combination is reported as it completes.
            if verbose:
                print(f"  [COMPARE]  running {n_combos} combination"
                      f"{'s' if n_combos > 1 else ''} "
                      f"({_repeats_text(n_repeats)} each) "
                      f"on {effective_n_jobs} workers ...", flush=True)
            try:
                done = 0
                with Parallel(n_jobs=effective_n_jobs, backend='loky',
                              return_as='generator') as parallel:
                    stream = parallel(
                        delayed(_compare_one_job)(self, cr, alg,
                                                  seed, n_repeats,
                                                  *job_sel, inner_jobs)
                        for cr, alg in combos)
                    for (cr, alg), reps, info, extra_pts, fixed in stream:
                        job_results.append(((cr, alg), reps))
                        settle[(cr, alg)] = info
                        done += 1
                        if verbose:
                            print(f"  [COMPARE]  [{done}/{n_combos}]"
                                  f"  {cr} + {alg}  done", flush=True)
            finally:
                if _prev_silent is None:
                    os.environ.pop('MERGEN_SILENT', None)
                else:
                    os.environ['MERGEN_SILENT'] = _prev_silent
        else:
            for cr, alg in combos:
                if verbose:
                    print(f"  [COMPARE]  {cr} + {alg} "
                          f"({_repeats_text(n_repeats)}) ...",
                          flush=True)
                key, reps, info, extra_pts, fixed = _compare_one_job(
                    self, cr, alg, seed, n_repeats, *job_sel,
                    effective_n_jobs)
                job_results.append((key, reps))
                settle[key] = info
        repeats = dict(job_results)
        unsettled = [f"{cr} + {alg} ({i['hits']} of {i['n_repeats']}; "
                     f"other repeats {_gap_text(i)})"
                     for (cr, alg), i in settle.items()
                     if n_repeats == 'auto' and not i['converged']]
        if unsettled:
            _warn(f"{len(unsettled)} of {len(combos)} combinations did not "
                  f"find their best result five times within the work "
                  f"limit, so another seed may give them a different "
                  f"design: {', '.join(unsettled)}.")

        # ── Shared Monte Carlo baseline ────────────────────────────────
        baseline = None
        if mc_samples > 0:
            from . import metrics as _m
            n_ref = (len(np.asarray(next(iter(repeats.values()))[0].design))
                     + (0 if extra_pts is None else len(extra_pts)))
            if verbose:
                print(f"  [COMPARE]  Monte Carlo baseline "
                      f"({mc_samples} random designs, n={n_ref}) ...",
                      flush=True)
            baseline = _m._mc_baseline(
                self.space.grid_sampler(), self.space.gmins,
                self.space.granges, n_ref,
                # Every metric a priority may name, so reselect() can
                # rank on any of them.
                metric_names=list(dict.fromkeys(
                    list(_COMPARE_METRICS) + list(_PRIORITY_METRICS))),
                crit_names=[], mc_samples=mc_samples,
                seed=44 if seed is None else seed,
                space=self.space,
                exclude=([ep.point for ep in self._exclusions]
                         + self._baseline_reserved()),
                fixed=fixed,
            )

        return self._finish_compare(
            repeats=repeats, crit_list=crit_list, alg_list=alg_list,
            priority=priority, selection=selection, tol=tol, rtol=rtol,
            baseline=baseline, seed=seed, n_repeats=n_repeats,
            verbose=verbose, reuse=True, n_jobs=n_jobs,
            extra_pts=extra_pts, polish=polish, polish_kicks=polish_kicks,
            settle=settle)

    def _finish_compare(self, repeats, crit_list, alg_list, priority,
                        selection, tol, rtol, baseline, seed, n_repeats,
                        verbose, reuse=False, n_jobs=-1,
                        extra_pts=None, polish=False,
                        polish_kicks=0, settle=None) -> "ComparisonResult":
        """
        Select the representative repeat of every combination, choose
        the winner and build the ComparisonResult. Shared by compare()
        and ComparisonResult.reselect(). ``extra_pts`` are the delivered
        out-of-budget rows; the metrics include them, as in
        quality_report().
        """
        import contextlib
        import io
        from . import metrics as _m
        from .selection import select

        priority = _check_priority(priority, allow_none=False)

        def only_priority(arg):
            if isinstance(arg, Mapping):
                return {m: v for m, v in arg.items() if m in priority}
            return arg
        tol_arr, rtol_arr = _check_selection(
            selection, priority, only_priority(tol), only_priority(rtol))
        metric_names = list(dict.fromkeys(
            list(priority) + list(_COMPARE_METRICS)))
        combos = [(cr, alg) for cr in crit_list for alg in alg_list]

        # Representative repeat of each combination: the same rule
        # run(seed, n_repeats, priority, selection, tol, rtol) applies,
        # so the table describes exactly the design each combination
        # delivers.
        results: Dict[Tuple[str, str], Any] = {}
        settle_now: Dict[Tuple[str, str], dict] = {}
        raw_rows, pct_rows, designs_n = [], [], []
        for key in combos:
            rep = _select_repeat(repeats[key], priority, self.space,
                                 selection, tol_arr, rtol_arr, extra_pts)
            results[key] = rep
            if settle and key in settle:
                # Measured against the repeat this priority delivers.
                settle_now[key] = _measured_repeat_info(
                    settle[key], repeats[key], rep, priority, self.space,
                    extra_pts)
            design = np.asarray(rep.design, dtype=float)
            if extra_pts is not None and len(extra_pts):
                design = np.vstack([design, extra_pts])
            Xn = _normalised(design, self.space)
            designs_n.append(Xn)
            raw = {'criterion': key[0], 'algorithm': key[1]}
            pct = {'criterion': key[0], 'algorithm': key[1]}
            for m in metric_names:
                val = _m._compute_metric(m, Xn, space=self.space)
                raw[m] = float(val)
                pct[m] = (float(_m._percentile_rank(
                              val, baseline[m], m in _m._HIGHER_BETTER))
                          if baseline is not None else np.nan)
            raw_rows.append(raw)
            pct_rows.append(pct)
        # Polishing: each combination design, moved row by row on the
        # feasible grid to improve the priority (mergen.polish). A
        # polished design that differs from its start joins the
        # candidates as '<algorithm>+polish'; the selection below is
        # unchanged, so the winner is never lexicographically worse, or
        # Pareto-dominated by, the winner without polishing.
        polished: Dict[Tuple[str, str], Tuple[str, str]] = {}
        if polish:
            from dataclasses import replace
            from .polish import polish as _polish
            hb = [m in _m._HIGHER_BETTER for m in priority]
            banned = ([ep.point for ep in self._exclusions]
                      + self._baseline_reserved()
                      + [pt for pt, _d, _o in self._prescribed])
            for k, key in enumerate(list(combos)):
                rep = results[key]
                if verbose:
                    print(f"  [COMPARE]  polishing {key[0]} + {key[1]} ...",
                          flush=True)
                rng = np.random.default_rng(np.random.SeedSequence(
                    [44 if seed is None else int(seed), 7, k]))
                D, info = _polish(
                    rep.design, int(rep.metadata.get('n_kept', 0)),
                    extra_pts, self.space, priority, hb, selection,
                    tol_arr, rtol_arr, banned, rng,
                    n_kicks=int(polish_kicks))
                if not info['improved']:
                    continue
                pkey = (key[0], key[1] + '+polish')
                combos.append(pkey)
                polished[pkey] = key
                results[pkey] = replace(
                    rep, design=D, score=float('nan'),
                    metadata=dict(rep.metadata, polished_from=key,
                                  polish=info))
                design = D
                if extra_pts is not None and len(extra_pts):
                    design = np.vstack([design, extra_pts])
                Xn = _normalised(design, self.space)
                designs_n.append(Xn)
                raw = {'criterion': pkey[0], 'algorithm': pkey[1]}
                pct = {'criterion': pkey[0], 'algorithm': pkey[1]}
                for m in metric_names:
                    val = _m._compute_metric(m, Xn, space=self.space)
                    raw[m] = float(val)
                    pct[m] = (float(_m._percentile_rank(
                                  val, baseline[m], m in _m._HIGHER_BETTER))
                              if baseline is not None else np.nan)
                raw_rows.append(raw)
                pct_rows.append(pct)
        metrics_df = pd.DataFrame(raw_rows)
        table      = pd.DataFrame(pct_rows)

        # A tie on the priority metrics goes to the better design on the
        # other metrics that may serve as a priority (never on those in
        # _NOT_PRIORITY), then to the criterion and algorithm names, so
        # the winner does not depend on the order of the criteria and
        # algorithms lists.
        others = [m for m in _PRIORITY_METRICS if m not in priority]
        def tie_key(i):
            vals = []
            for m in others:
                if m not in raw_rows[i]:
                    raw_rows[i][m] = float(_m._compute_metric(
                        m, designs_n[i], space=self.space))
                vals.append(-raw_rows[i][m] if m in _m._HIGHER_BETTER
                            else raw_rows[i][m])
            return tuple(v if np.isfinite(v) else np.inf
                         for v in vals) + combos[i]
        tie_rank = np.argsort(sorted(range(len(combos)), key=tie_key))
        sel = select(metrics_df[list(priority)].to_numpy(dtype=float),
                     [m in _m._HIGHER_BETTER for m in priority],
                     rule=selection, tol=tol_arr, rtol=rtol_arr,
                     tiebreak=tie_rank)
        if sel.excluded:
            _warn("Excluded from selection (NaN or infinite priority "
                  "metric): "
                  + ", ".join(f"{combos[i][0]}/{combos[i][1]}"
                              for i in sel.excluded))
        if 'min_distance' in priority:
            # A known limit (0.1.1), measured on five test spaces, seeds
            # 5-9, under the default Pareto priority and under
            # ('min_distance',).
            _warn("min_distance is a priority metric, and compare() does "
                  "not reliably reach its best value: on test spaces the "
                  "winner's min_distance lay a median "
                  + ("10 % (up to 38 %) below the best known under the "
                     "default Pareto priority and 3 % (up to 12 %) with "
                     "min_distance alone under 'lexicographic'"
                     if polish else
                     "15 % (up to 38 %) below the best known under the "
                     "default Pareto priority and 13 % (up to 38 %) with "
                     "min_distance alone under 'lexicographic', without "
                     "polishing")
                  + ". minimax and cd2 are reached more closely; see the "
                  "guide on comparing designs.")
        best_key = combos[sel.winner]

        table.insert(0, 'best', [' *' if k == best_key else ''
                                 for k in combos])
        cmp = ComparisonResult(
            table=table, metrics=metrics_df, results=results,
            best=best_key, priority=priority, selection=selection,
            tol=dict(zip(priority, tol_arr.tolist())),
            rtol=dict(zip(priority, rtol_arr.tolist())),
            ties=[combos[i] for i in sel.ties],
            front=[combos[i] for i in sel.front],
            decided_by=sel.decided_by)
        cmp._context = dict(sampler=self, repeats=repeats,
                            crit_list=crit_list, alg_list=alg_list,
                            baseline=baseline, seed=seed,
                            n_repeats=n_repeats, tol=tol, rtol=rtol,
                            extra_pts=extra_pts, polish=polish,
                            polish_kicks=polish_kicks, settle=settle)
        # A polished row reports the repeats of the combination it
        # started from.
        for pkey, origin in polished.items():
            if origin in settle_now:
                settle_now[pkey] = settle_now[origin]
        cmp.repeat_info = {k: _public_repeat_info(v, n_repeats == 'auto')
                           for k, v in settle_now.items()}

        # best_result must be a full SamplingResult that matches a
        # standalone run() exactly (a polished winner excepted: run()
        # scores its design, handed over below). Produce it by calling run() once for
        # the winning combination with the same settings; the single
        # seed derivation guarantees identical optimisations. Right
        # after the sweep (reuse=True) the winner's stored repeats are
        # handed to run() instead of being optimised again. Its run
        # banner is hidden; its warnings are passed on.
        # reselect() (reuse=False) also hands over the stored repeats, all
        # of them: under a new priority, 'auto' could stop after a
        # different number of repeats than the table was built from.
        _bc, _ba = polished.get(best_key, best_key)
        buf = io.StringIO()
        if best_key in polished:
            # The polished design is delivered as the only repeat; run()
            # scores it and builds the validation and extra sets.
            self._reuse_repeats = {_ba: [results[best_key]]}
            best_repeats = 1
        else:
            self._reuse_repeats = {_ba: repeats[best_key]}
            best_repeats = n_repeats if reuse else len(repeats[best_key])
        try:
            with contextlib.redirect_stdout(buf):
                cmp._best_result = self.run(
                    criteria=_bc, algorithm=_ba, seed=seed,
                    n_repeats=best_repeats, priority=priority,
                    selection=selection, tol=cmp.tol, rtol=cmp.rtol,
                    n_jobs=n_jobs, verbose=False)
        finally:
            self._reuse_repeats = None
        for line in buf.getvalue().splitlines():
            # compare() has already listed the unsettled combinations.
            if '[WARNING]' in line and not (
                    reuse and 'repeats before the work limit' in line):
                print(line)
        # best_result was built from the stored repeats (one, for a
        # polished winner); it reports the sweep's repeats instead.
        if best_key in settle_now:
            cmp._best_result._repeat_info[_ba] = settle_now[best_key]
            cmp._best_result._meta['n_repeats'] = n_repeats
        delivered = np.asarray(cmp._best_result.algorithm_results[_ba].design)
        compared  = np.asarray(results[best_key].design)
        if best_key in polished:
            # run() delivers the design in canonical form: rows sorted
            # and, where the problem is symmetric, possibly mirrored,
            # which leaves every metric unchanged. The table keeps that
            # form once the priority metrics are confirmed equal.
            cmp._best_result._meta['polished_from'] = polished[best_key]
            same = delivered.shape == compared.shape and bool(np.allclose(
                _priority_values(delivered, priority, self.space, extra_pts),
                _priority_values(compared, priority, self.space, extra_pts),
                rtol=_SAME_RTOL, atol=1e-12))
            if same:
                from dataclasses import replace
                results[best_key] = replace(results[best_key],
                                            design=delivered.copy())
        else:
            same = np.array_equal(delivered, compared)
        if not same:
            _warn("best_result differs from the compared design: the "
                  "sampler was reconfigured after compare(). Re-run "
                  "compare() to obtain a consistent result.")
        if verbose:
            cmp.summary()
        elif cmp._tie_text():
            _warn(f"Tie: {cmp._tie_text()}")
        return cmp

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f} min"
        return f"{seconds / 3600:.1f} hr"

    # ------------------------------------------------------------------ #
    # Private: layout helpers                                              #
    # ------------------------------------------------------------------ #

    def _assemble_initial_design(
        self,
        prescribed_in_pts:          List[np.ndarray],
        focus_in_pts:               List[np.ndarray],
        in_design_sa_prescribed:    List[np.ndarray],
        focus_sa_in_design_sampled: List[np.ndarray],
        seed_design:                np.ndarray,
        n_dims:                     int,
    ) -> np.ndarray:
        """
        Build the initial design matrix in canonical row order::

            [ criterion-blind frozen rows | criterion-visible frozen rows | optimised rows ]

        The greedy seed begins with the anchor (criterion-visible
        frozen) rows and continues with the optimised rows.  We
        prepend the criterion-blind frozen rows so a single
        ``crit_start`` index suffices in :meth:`run`.
        """
        # Criterion-blind = in_design but NOT in_optim
        sa_set = {tuple(p) for p in in_design_sa_prescribed}
        blind_prescribed = [p for p in prescribed_in_pts
                            if tuple(p) not in sa_set]

        sa_focus_set = {tuple(p) for p in focus_sa_in_design_sampled}
        blind_focus  = [p for p in focus_in_pts
                        if tuple(p) not in sa_focus_set]

        parts: List[np.ndarray] = []
        if blind_prescribed:
            parts.append(np.array(blind_prescribed, dtype=float))
        if blind_focus:
            parts.append(np.array(blind_focus, dtype=float))
        # seed_design already starts with anchor (criterion-visible
        # frozen rows) followed by the optimised rows.
        parts.append(seed_design)

        out = np.vstack(parts) if parts else np.empty((0, n_dims))

        # Sanity check on row count
        expected = (len(prescribed_in_pts) + len(focus_in_pts)
                    + max(0, len(seed_design) - len(in_design_sa_prescribed)
                          - len(focus_sa_in_design_sampled)))
        if len(out) != expected:
            _warn(
                f"Initial design size mismatch: expected ~{expected}, "
                f"got {len(out)}. Continuing."
            )
        return out

    def _check_conflicts(self) -> None:
        """
        Reject configurations where the same grid point appears in more
        than one role, and reject within-role duplicates.

        A given point may be prescribed *or* a focus centre *or* an
        exclusion centre — never two of these at once.
        """
        p_pts = [pt for pt, _, _ in self._prescribed]
        f_pts = [fp.point for fp in self._focus]
        e_pts = [ep.point for ep in self._exclusions]

        tol = self.space.grid_tol

        def _same(a: np.ndarray, b: np.ndarray) -> bool:
            return bool(np.all(np.abs(a - b) <= tol))

        def _overlap(a: np.ndarray, lst: List[np.ndarray]) -> bool:
            return any(_same(a, b) for b in lst)

        # Cross-role conflicts
        for fp in f_pts:
            if _overlap(fp, p_pts):
                _fatal(
                    f"Point {fp.tolist()} appears in both add_prescribed() "
                    f"and add_focus(). A point cannot be both."
                )
        for ep in e_pts:
            if _overlap(ep, p_pts):
                _fatal(
                    f"Point {ep.tolist()} appears in both add_prescribed() "
                    f"and add_exclusion(). A prescribed point cannot be "
                    f"excluded."
                )
            if _overlap(ep, f_pts):
                _fatal(
                    f"Point {ep.tolist()} appears in both add_focus() and "
                    f"add_exclusion(). A point cannot be both attracted "
                    f"and repelled."
                )

        # A point of a user-supplied set is never part of the design.
        for sname, (upts, _ucolor) in self._user_sets.items():
            for u in upts:
                if _overlap(u, p_pts):
                    _fatal(f"Point {np.asarray(u).tolist()} is both "
                           f"prescribed and in the set '{sname}' added "
                           f"with add_set(); a set point cannot be in "
                           f"the design.")
                if _overlap(u, f_pts):
                    _fatal(f"Point {np.asarray(u).tolist()} is both a "
                           f"focus centre and in the set '{sname}' added "
                           f"with add_set().")
                if _overlap(u, e_pts):
                    _fatal(f"Point {np.asarray(u).tolist()} is both "
                           f"excluded and in the set '{sname}' added with "
                           f"add_set(); an excluded point cannot be in "
                           f"any set.")

        # Within-role duplicates
        def _check_dupes(pts: List[np.ndarray], label: str) -> None:
            for i, a in enumerate(pts):
                for b in pts[i + 1:]:
                    if _same(a, b):
                        _fatal(
                            f"Point {a.tolist()} appears in {label} more "
                            f"than once. Merge into a single entry."
                        )
        _check_dupes(p_pts, "add_prescribed()")
        _check_dupes(f_pts, "add_focus()")
        _check_dupes(e_pts, "add_exclusion()")
        _check_dupes([u for upts, _c in self._user_sets.values()
                      for u in upts], "the sets added with add_set()")
        if self._loaded_design is not None:
            loaded = list(self._loaded_design[0])
            _check_dupes(loaded, "load_design()")
            for sname, (upts, _ucolor) in self._user_sets.items():
                for u in upts:
                    if _overlap(u, loaded):
                        _fatal(f"Point {np.asarray(u).tolist()} is both in "
                               f"the loaded design and in the set "
                               f"'{sname}' added with add_set().")

    # ------------------------------------------------------------------ #
    # Private: focus / exclusion sampling                                  #
    # ------------------------------------------------------------------ #

    def _build_focus_exclusion(
        self,
        gs:       GridSampler,
        gmins:    np.ndarray,
        granges:  np.ndarray,
        reserved: set,
        rng:      np.random.Generator,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], dict,
               set, List[np.ndarray]]:
        """
        Sample focus neighbours and build exclusion repulsion weights.

        Focus neighbours are drawn with ``rng`` from feasible grid points
        that are neither reserved nor exclusion centres.

        Returns
        -------
        focus_in_pts  : list[np.ndarray]
            Sampled focus points (centre + neighbours) for entries with
            ``in_design=True``.
        focus_out_pts : list[np.ndarray]
            Sampled focus points for entries with ``in_design=False``.
        focus_sa_pts  : list[np.ndarray]
            All sampled focus points with ``in_optim=True`` (criterion-visible).
        repel_weights : dict[int, float]
            Sparse repulsion weights keyed by grid index. Computed but
            not used by any optimiser in this version.
        reserved      : updated set of reserved grid indices
        focus_sa_in   : list[np.ndarray]
            The ``focus_in_pts`` with ``in_optim=True``, in the same order.
        focus_sa_out  : list[np.ndarray]
            The ``focus_out_pts`` with ``in_optim=True``, in the same order.
        """
        focus_in_pts:  List[np.ndarray] = []
        focus_sa_in:   List[np.ndarray] = []
        excluded = {gs.point_to_index(ep.point) for ep in self._exclusions}
        constraints, names = self.space._constraints, self.space.names
        focus_out_pts: List[np.ndarray] = []
        focus_sa_pts:  List[np.ndarray] = []
        focus_sa_out:  List[np.ndarray] = []
        repel_weights: Dict[int, float] = {}

        mask = self.space.is_mask

        # ── Focus points ──────────────────────────────────────────────
        for fp in self._focus:
            pt       = fp.point
            self_idx = gs.point_to_index(pt)
            n_draw   = fp.n_samples

            if fp.include_center is True:
                # Centre is guaranteed in the design
                if self_idx >= 0 and self_idx not in reserved:
                    reserved.add(self_idx)
                    (focus_in_pts if fp.in_design else focus_out_pts).append(
                        pt.copy())
                    if fp.in_optim:
                        focus_sa_pts.append(pt.copy())
                        (focus_sa_in if fp.in_design
                         else focus_sa_out).append(pt.copy())
                    n_draw = max(fp.n_samples - 1, 0)
                # A centre already taken (e.g. as a neighbour of another
                # focus region) does not count; all n_samples points
                # are drawn from its neighbourhood.
            elif fp.include_center is False:
                # Centre is never selected
                if self_idx >= 0 and self_idx not in reserved:
                    reserved.add(self_idx)

            # Gaussian-weighted neighbour sampling
            if n_draw == 0:
                continue
            # Neighbours are the grid nodes within four kernel widths of
            # the centre on every ordered axis, weighted by a Gaussian
            # whose width is ``spread`` grid steps; a nominal axis keeps
            # the centre's category.
            reach  = max(1, int(np.ceil(4 * fp.spread)))
            centre = [gs.level_of(j, pt[j]) for j in range(len(pt))]

            def box(r):
                return [np.array([centre[j]]) if mask[j] else
                        np.arange(max(0, centre[j] - r),
                                  min(gs.n_levels[j], centre[j] + r + 1))
                        for j in range(len(pt))]
            # Keep the box enumerable in many dimensions.
            while reach > 1 and np.prod([len(a) for a in box(reach)],
                                        dtype=float) > _FOCUS_BOX_MAX:
                reach -= 1
            axes = box(reach)
            candidates: List[Tuple[float, int, np.ndarray]] = []
            for lev in itertools.product(*axes):
                idx = int(sum(lv * st for lv, st in zip(lev, gs._strides)))
                if idx in reserved or idx in excluded:
                    continue
                cpt = gs.index_to_point(idx)
                if constraints and not all(
                        c(dict(zip(names, cpt))) for c in constraints):
                    continue
                steps = np.array(lev, dtype=float) - centre
                w = float(np.exp(-0.5 * np.sum((steps / fp.spread) ** 2)))
                candidates.append((w + 1e-300, idx, cpt))

            if not candidates:
                _warn(f"FocusPoint {pt.tolist()}: no free candidates found.")
                continue

            weights = np.array([c[0] for c in candidates])
            weights = weights / weights.sum()
            n_take  = min(n_draw, len(candidates))
            if n_take < n_draw:
                _warn(
                    f"FocusPoint {pt.tolist()}: requested {n_draw} "
                    f"neighbours but only {n_take} available."
                )

            chosen = rng.choice(
                len(candidates), size=n_take, replace=False, p=weights,
            )
            for ci in chosen:
                _, cidx, cpt = candidates[ci]
                reserved.add(cidx)
                (focus_in_pts if fp.in_design else focus_out_pts).append(cpt)
                if fp.in_optim:
                    focus_sa_pts.append(cpt)
                    (focus_sa_in if fp.in_design
                     else focus_sa_out).append(cpt)

        # ── Exclusion points ──────────────────────────────────────────
        for ep in self._exclusions:
            pt       = ep.point
            self_idx = gs.point_to_index(pt)
            if self_idx >= 0 and self_idx not in reserved:
                reserved.add(self_idx)
                _info(f"ExclusionPoint {pt.tolist()}: removed from pool.")


        return (focus_in_pts, focus_out_pts, focus_sa_pts, repel_weights,
                reserved, focus_sa_in, focus_sa_out)

    # ------------------------------------------------------------------ #
    # Private: Stochastic Coordinate Exchange (SCE) core                   #
    # ------------------------------------------------------------------ #

    def _kennard_stone(
        self,
        gs:       GridSampler,
        reserved: set,
        ref:      np.ndarray,
        n:        int,
    ) -> np.ndarray:
        """
        Holdout selection from the feasible candidate pool.

        Picks the ``n`` pool points that, as a set, lie farthest (HEOM
        distance in normalised coordinates) from ``ref`` and from each
        other: the maximin set of
        :func:`mergen.distances.maximin_subset`, which starts from the
        Kennard-Stone selection; when the exact search is too large it
        keeps the best set found and warns. Grid indices in
        ``reserved`` are never picked.

        References
        ----------
        Johnson, M. E., Moore, L. M. & Ylvisaker, D. (1990).
            *J. Statist. Plann. Inference*, 26(2), 131–148.
        Kennard, R. W. & Stone, L. A. (1969).
            *Technometrics*, 11(1), 137–148.
        """
        from .distances import maximin_subset

        space = self.space
        if n <= 0:
            return np.empty((0, space.n_parameters))
        pool    = space.candidate_pool
        gmins   = space.gmins
        granges = space.granges
        free    = ~np.isin(gs.pool_indices(),
                           np.fromiter(reserved, dtype=np.int64))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rows = maximin_subset((pool - gmins) / granges, n,
                                  ref=(np.asarray(ref, dtype=float) - gmins)
                                  / granges,
                                  available=free, space=space)
        for w in caught:
            warnings.warn(f"Holdout selection of {n} points: "
                          f"{str(w.message).split(': ', 1)[-1]}",
                          stacklevel=3)
        if len(rows) < n:
            _warn(f"Only {len(rows)} free feasible points are left for a "
                  f"holdout set of {n}; the set has {len(rows)} points.")
        return pool[rows].copy()

    def _snapshot_state(self) -> dict:
        """Capture the mutable design-spec state for safe restoration."""
        return {
            'prescribed'     : list(self._prescribed),
            'focus'          : list(self._focus),
            'exclusions'     : list(self._exclusions),
            'n_samples'      : self._n_samples,
            'n_validation'   : self._n_validation,
            'extra_sets'     : self._extra_sets,
            'loaded_design'  : self._loaded_design,
        }

    def _restore_state(self, snap: dict) -> None:
        """Restore the state captured by :meth:`_snapshot_state`."""
        self._prescribed[:]  = snap['prescribed']
        self._focus[:]       = snap['focus']
        self._exclusions[:]  = snap['exclusions']
        self._n_samples      = snap['n_samples']
        self._n_validation   = snap['n_validation']
        self._extra_sets     = snap['extra_sets']
        self._loaded_design  = snap['loaded_design']
