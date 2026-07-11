"""
mergen.sampler
==============
Space-filling design via the Stochastic Coordinate Exchange (SCE) engine.

This module exposes four objects:

  - :class:`Sampler`        — configure and run the design construction.
  - :class:`SamplingResult` — container for design + validation + extras.
  - :class:`FocusPoint`     — denser sampling around a critical region.
  - :class:`ExclusionPoint` — repel sampling away from a region.

Quick start
-----------
    from mergen.space   import ParameterSpace
    from mergen.sampler import Sampler

    space   = ParameterSpace({'temperature': range(100, 400, 10),
                              'pressure'   : ('continuous', 0.5, 5.0)})
    sampler = Sampler(space)
    sampler.add_prescribed([[200, 2.5]], in_design=True,  in_optim=False)
    sampler.add_focus    ([350, 4.5], spread=1.5,         in_design=True, in_optim=True)
    sampler.add_exclusion([100, 0.5], spread=1.0)
    sampler.set_design(n_samples=30)
    sampler.set_sce(n_restarts=5)
    result  = sampler.run(criteria='umaxpro', seed=44)
    result.summary()

Algorithm
---------
The optimisation is a Stochastic Coordinate Exchange (SCE) outer-inner loop:

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

import os
import random
import time
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
_RED    = "\033[0;31m"
_GREEN  = "\033[0;32m"
_YELLOW = "\033[1;33m"
_CYAN   = "\033[0;36m"
_RESET  = "\033[0m"

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
):
    """
    Run a single optimiser end-to-end (initial design + optimise).

    Top-level (module-scope) function so that joblib's pickling-based
    multiprocessing backend can serialise the task. Called sequentially
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

    return optimiser.optimize(
        initial_design = full_design,
        space          = space,
        criterion      = criterion,
        banned         = initial_reserved,
        n_frozen       = n_frozen,
        crit_start     = crit_start,
        seed           = seed,
        verbose        = verbose,
    )


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
        Gaussian kernel width in normalised grid steps.
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
    A region to avoid: the centre is hard-excluded from the candidate
    pool and its neighbourhood receives a Gaussian soft-repulsion
    penalty during coordinate-exchange candidate selection.

    Parameters
    ----------
    point : array-like, shape (n_parameters,)
        Grid coordinates of the exclusion centre.
    spread : float
        Gaussian repulsion kernel width in normalised grid steps.
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
        Kennard-Stone validation set.
    sets       : dict[str, pd.DataFrame]
        Extra named sets (e.g. ``{'test': df, 'holdout': df}``).
    designs    : dict[str, pd.DataFrame]
        Designs per criterion when multiple criteria are run.
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
        self.samples        = samples
        self.validation     = validation
        self.space          = space
        self.sets:    Dict[str, pd.DataFrame] = {}
        self.designs: Dict[str, pd.DataFrame] = {}
        # Multi-algorithm results — populated by Sampler.run() when
        # ``algorithm=[...]`` is requested. Keys: algorithm name,
        # Values: OptimisationResult instance (algorithms/base.py).
        self.algorithm_results: Dict[str, Any] = {}
        # The algorithm whose design is exposed as ``self.samples`` (the
        # lowest-scoring one when multiple were run).
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
        Compare quality metrics across all criteria in ``self.designs``.
        Returns a DataFrame (criteria × metrics).
        """
        from . import metrics as _metrics
        return _metrics.comparison(self)

    def plot(self, kind: str = 'pairplot', **kwargs) -> None:
        """
        Visualise the design.  Delegates to :mod:`mergen.output`.

        Parameters
        ----------
        kind : ``'pairplot'`` | ``'1d'`` | ``'2d'`` | ``'distances'``
               | ``'quality'`` | ``'all'``
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
        One row per (criterion, algorithm) combination, sorted by the
        priority metrics; columns hold percentile ranks (0-100, higher
        is better) against the shared Monte Carlo baseline. The winner
        is flagged in the ``best`` column.
    results : dict
        Maps ``(criterion, algorithm)`` to the full
        :class:`SamplingResult`, so any candidate design can be
        inspected or exported, not only the winner.
    best : tuple of (str, str)
        The winning (criterion, algorithm) pair.
    priority : tuple of str
        The metrics that defined the ranking, in order.
    """

    def __init__(self, table, results, best, priority):
        self.table    = table
        self.results  = results
        self.best     = best
        self.priority = priority

    @property
    def best_result(self) -> "SamplingResult":
        """The SamplingResult of the winning combination."""
        return self.results[self.best]

    def summary(self) -> None:
        """Print the ranked comparison table."""
        print()
        print("═" * 72)
        print("  MERGEN — Criterion / Algorithm Comparison")
        print("═" * 72)
        print(f"  Priority: {' > '.join(self.priority)}   "
              f"(percentile vs shared MC baseline, higher is better)")
        print("─" * 72)
        with pd.option_context('display.width', 100,
                               'display.float_format',
                               lambda v: f"{v:6.1f}"):
            print(self.table.to_string(index=False))
        print("─" * 72)
        print(f"  Best: criteria='{self.best[0]}', "
              f"algorithm='{self.best[1]}'")
        print("═" * 72)

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
        Save the ranked comparison table as a Markdown file under the
        output directory.

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
                f"priority={self.priority})")


class Sampler:
    """
    Space-filling design sampler using Stochastic Coordinate Exchange (SCE).

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
        in_optim:     bool = False,
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
            ``True``  → the optimiser sees these points and pushes
            optimised points away from them.
            ``False`` → the optimiser ignores them (the points are
            only reserved against being re-selected).

        Returns
        -------
        self
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        if pts.ndim != 2 or pts.shape[1] != self.space.n_parameters:
            _fatal(
                f"Each prescribed point must have {self.space.n_parameters} "
                f"coordinates, got shape {pts.shape}."
            )
        for row in pts:
            validated = self.space.validate_point(row, label="Prescribed point")
            self._prescribed.append((validated, bool(in_design), bool(in_optim)))
        return self

    def _encode_points(self, points, context: str) -> np.ndarray:
        """
        Convert a list/array of points to the numeric grid encoding.

        Nominal factors are stored internally as integer category
        indices, so string category labels (e.g. 'adam') are mapped to
        their index before the array is built. Numeric factors pass
        through unchanged. Accepts a single point or a list of points.
        """
        rows = list(points)
        if len(rows) and np.ndim(rows[0]) == 0:
            rows = [rows]                      # a single point
        names = self.space.names
        encoded = []
        for row in rows:
            if len(row) != len(names):
                _fatal(
                    f"{context}: each point must have {len(names)} "
                    f"coordinates, got {len(row)}."
                )
            vals = []
            for name, v in zip(names, row):
                if self.space.is_nominal(name) and isinstance(v, str):
                    labels = self.space.category_labels(name)
                    if v not in labels:
                        _fatal(
                            f"{context}: '{v}' is not a level of nominal "
                            f"factor '{name}'. Valid levels: {labels}."
                        )
                    vals.append(float(labels.index(v)))
                else:
                    vals.append(float(v))
            encoded.append(vals)
        return np.asarray(encoded, dtype=float)

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
            (e.g. ``'#3a86ff'``). If omitted, a neutral fallback colour
            is assigned by the plotting layer.

        Returns
        -------
        self
        """
        builtin = {"Optimised", "Validation", "Prescribed", "Focus"}
        taken   = builtin | set(self._user_sets) | set(self._extra_sets or {})
        if name in taken:
            _fatal(f"Set name '{name}' is already in use.")
        pts = self._encode_points(points, context=f"Set '{name}'")
        validated = np.vstack([
            self.space.validate_point(row, label=f"Set '{name}' point")
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
        validation set and any extra sets (Kennard-Stone) around them.
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
        validated = np.vstack([
            self.space.validate_point(row, label="Loaded design point")
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
        spread         : Gaussian kernel width in normalised grid steps
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
        Add an exclusion point — sampling avoids this region.

        The centre is hard-excluded from the candidate pool.  Its
        neighbourhood receives a Gaussian soft-repulsion weight that
        downweights candidate values during coordinate-exchange
        selection.

        Parameters
        ----------
        point  : grid coordinates of the exclusion centre
        spread : Gaussian repulsion kernel width

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
                       Kennard-Stone holdout sets, e.g.
                       ``{'test': 10, 'holdout': 5}``.

        Returns
        -------
        self
        """
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
        >>> sampler.set_optimizer('sce', n_complexes=4)
        >>> sampler.set_optimizer('ese', n_outer=20, n_inner=100)
        """
        # Import here to avoid circular imports at module load time
        from .algorithms import get_optimizer

        # Normalise the name (registry is case-insensitive on user input)
        name = name.strip().lower()

        # Look up the optimiser class (raises KeyError if unknown)
        opt_cls = get_optimizer(name)

        # Validate kwargs against the optimiser's parameter schema by
        # instantiating it; this raises ValueError on unknown keys.
        opt_cls(**kwargs)

        # Store the hyperparameters
        self._optimizer_params[name] = dict(kwargs)
        return self

    def set_dimension_weights(
        self,
        weights: Union[Dict[str, float], List[float]],
    ) -> "Sampler":
        """
        Set per-dimension importance weights for greedy maximin seeding.

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
        if len(w) != self.space.n_parameters:
            _fatal(
                f"dimension_weights length ({len(w)}) must match "
                f"n_parameters ({self.space.n_parameters})."
            )
        self._dim_weights = w / w.sum()
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
        criteria:  Union[str, List[str], BaseCriterion] = 'umaxpro',
        algorithm: Union[str, List[str]]                = 'sa',
        seed:      Optional[int]                        = 44,
        n_jobs:    Optional[int]                        = None,
        verbose:   bool                                  = True,
    ) -> SamplingResult:
        """
        Generate the space-filling design.

        Parameters
        ----------
        criteria : str, list of str, or BaseCriterion instance
            Optimisation criterion (or list of criteria).
            Available built-in names: ``'umaxpro'``, ``'maxpro'``,
            ``'phi_p'``, ``'cd2'``, ``'stratified'``.  Default
            ``'umaxpro'``.
        algorithm : str or list of str, optional
            Name of the optimiser (or list of names) to run. Each name
            must be a registered optimiser
            (see :func:`mergen.list_optimizers`). When a list is given,
            every algorithm is run independently and the results are
            collected in :attr:`SamplingResult.algorithm_results`.
            Default ``'sa'``.
        seed : int or None
            Random seed for reproducibility (default 44).
        n_jobs : int or None, optional
            Number of parallel workers for the multi-algorithm
            dispatcher. ``None`` (default) and ``1`` run sequentially.
            Positive integers request that many workers (up to the
            number of CPUs). Negative integers follow the joblib
            convention: ``-1`` for all CPUs, ``-2`` for all but one,
            and so on. Parallelism applies only across distinct
            algorithms; the per-algorithm restart loop is ILS-based
            and therefore intentionally serial. When ``n_jobs`` is
            greater than the number of algorithms, the extra workers
            are unused.
        verbose : bool, optional
            Print progress information. Default ``True``.

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
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if self._loaded_design is not None:
            return self._run_loaded(seed=seed, verbose=verbose)

        # Normalise algorithm into a list (case-insensitive)
        if isinstance(algorithm, str):
            algorithm_names = [algorithm.strip().lower()]
        else:
            algorithm_names = [a.strip().lower() for a in algorithm]
        if not algorithm_names:
            _fatal("'algorithm' cannot be empty.")

        # Resolve criterion(ia)
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

        # NOTE: multi-criterion support is reserved for a future release.
        # For now we run the first criterion only.
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
        print()
        print("═" * 60)
        print("  MERGEN — Space-filling Design")
        print("═" * 60)
        print(f"  Parameters      : {n_dims}")
        print(f"  Candidates      : {space.n_candidates:,}")
        print(f"  n_samples       : {n_global}  "
              f"(prescribed_in={n_prescribed_in}, "
              f"focus_in={n_focus_in}, "
              f"optimised_slots={n_optimised_slots})")
        if n_prescribed_out or n_focus_out:
            print(f"  Extra points    : +{n_prescribed_out + n_focus_out}  "
                  f"(prescribed_out={n_prescribed_out}, "
                  f"focus_out={n_focus_out})")
        print(f"  Total design    : {n_total_design}")
        print(f"  Validation      : {n_validation}")
        print(f"  Criterion       : {crit_name}")
        print(f"  Algorithm(s)    : {', '.join(algorithm_names)}")
        print("─" * 60)

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
            print("  [MERGEN]   Sampling focus / exclusion regions...",
                  flush=True)
        focus_in_pts, focus_out_pts, focus_sa_pts, repel_weights, reserved = \
            self._build_focus_exclusion(gs, gmins, granges, reserved)

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
        focus_sa_in_design_sampled = []
        if focus_in_pts and any(fp.in_optim for fp in self._focus):
            sa_indices_in = [k for k, fp in enumerate(self._focus)
                             if fp.in_design and fp.in_optim]
            cursor = 0
            for k, fp in enumerate(self._focus):
                count = fp.n_samples if fp.in_design else 0
                if k in sa_indices_in:
                    focus_sa_in_design_sampled.extend(
                        focus_in_pts[cursor:cursor + count]
                    )
                cursor += count
        if focus_sa_in_design_sampled:
            anchor_parts.append(np.array(focus_sa_in_design_sampled,
                                         dtype=float))

        _corner_used = False
        if anchor_parts:
            anchor = np.vstack(anchor_parts)
        else:
            # No criterion-visible frozen rows → seed from a corner
            # (Morris & Mitchell 1995 strategy: extreme starting point).
            corner = np.array([v[0] if i % 2 == 0 else v[-1]
                                for i, v in enumerate(space.values)])
            idx = gs.point_to_index(corner)
            if idx >= 0:
                reserved.add(idx)
            anchor       = corner[np.newaxis, :]
            _corner_used = True

        # ── Initial design: delegated to each optimiser ──────────────
        # Each optimiser provides its own prepare_initial_design()
        # method so it can use the type of starting design its
        # underlying algorithm was designed for. The default in
        # BaseOptimizer is a balanced LHS (Joseph, Gul & Ba 2018),
        # which matches what SA, SCE and ESE expect from the
        # literature; subclasses may override for non-LHS algorithms.
        if self._prescribed or self._focus:
            print("  [MERGEN]   Preparing anchor points...", flush=True)

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

        # n_frozen: the optimiser must not move these rows
        n_frozen = len(prescribed_in_pts) + len(focus_in_pts)

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

        # Resolve n_jobs and decide whether to parallelise.
        # Parallelism applies *only* across distinct algorithms; the
        # per-algorithm restart loop is ILS-based and intentionally
        # serial (each restart kicks from the previous best — see
        # Lourenco, Martin & Stutzle (2003) Handbook of Metaheuristics).
        n_jobs_resolved = _resolve_n_jobs(n_jobs)
        n_algos         = len(algorithm_names)
        effective_n_jobs = min(n_jobs_resolved, n_algos)
        run_in_parallel = effective_n_jobs > 1 and n_algos > 1

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
        eff_seed = seed if seed is not None else 44
        tasks_args = [
            dict(
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
                seed                        = eff_seed,
                # In parallel mode, suppress per-task verbose output so
                # worker streams do not interleave.
                verbose                     = (verbose and not run_in_parallel),
            )
            for alg_name in algorithm_names
        ]

        if run_in_parallel:
            from joblib import Parallel, delayed
            # Workers spawn fresh Python processes that re-import mergen;
            # set MERGEN_SILENT in the child env so each worker does not
            # print the package banner on import.
            _prev_silent = os.environ.get('MERGEN_SILENT')
            os.environ['MERGEN_SILENT'] = '1'
            try:
                results_list = Parallel(n_jobs=effective_n_jobs)(
                    delayed(_run_one_algorithm_task)(**ta) for ta in tasks_args
                )
            finally:
                if _prev_silent is None:
                    os.environ.pop('MERGEN_SILENT', None)
                else:
                    os.environ['MERGEN_SILENT'] = _prev_silent
            for alg_name, result in zip(algorithm_names, results_list):
                algorithm_results[alg_name] = result
                if verbose:
                    _ok(f"{alg_name:<6} done -- score={result.score:.4g}  "
                        f"(elapsed {self._fmt_time(result.elapsed)})")
        else:
            for ta, alg_name in zip(tasks_args, algorithm_names):
                result = _run_one_algorithm_task(**ta)
                algorithm_results[alg_name] = result
                if verbose:
                    _ok(f"{alg_name:<6} done -- score={result.score:.4g}  "
                        f"(elapsed {self._fmt_time(result.elapsed)})")

        t_elapsed = time.perf_counter() - t_run_start

        # Pick the best result across all algorithms (lowest score)
        best_algorithm = min(algorithm_results,
                             key=lambda k: algorithm_results[k].score)
        best_design    = algorithm_results[best_algorithm].design
        best_score     = algorithm_results[best_algorithm].score

        if verbose and len(algorithm_names) > 1:
            _ok(f"Best: {best_algorithm}  score={best_score:.4g}  "
                f"(total elapsed {self._fmt_time(t_elapsed)})")

        # Reconstruct reserved set from the final design (used downstream
        # for validation / extra-set Kennard-Stone selection). Start from
        # the initial reserved (anchors only) and add the optimiser's
        # output rows.
        best_reserved = initial_reserved.copy()
        for row in best_design[n_frozen:]:
            idx = gs.point_to_index(row)
            if idx >= 0:
                best_reserved.add(idx)

        final_design   = best_design
        final_reserved = best_reserved

        # ── Validation set (Kennard-Stone) ─────────────────────────────
        val_pts = self._kennard_stone(
            gs, final_reserved, final_design, n_validation, gmins, granges,
        )
        val_reserved = final_reserved.copy()
        for vp in val_pts:
            idx = gs.point_to_index(vp)
            if idx >= 0:
                val_reserved.add(idx)

        # ── Extra sets (Kennard-Stone) ─────────────────────────────────
        extra_dfs: Dict[str, pd.DataFrame] = {}
        cur_reserved = val_reserved.copy()
        cur_ref      = (np.vstack([final_design, val_pts])
                        if len(val_pts) else final_design)

        # User-supplied sets go in first so the generated extra sets
        # below are pushed away from them as well.
        for uname, (upts, ucolor) in self._user_sets.items():
            df_u = pd.DataFrame(upts, columns=names)
            df_u['point_type'] = uname
            if ucolor is not None:
                df_u['color'] = ucolor
            df_u.index.name = 'id'
            extra_dfs[uname] = df_u
            for up in upts:
                idx = gs.point_to_index(up)
                if idx >= 0:
                    cur_reserved.add(idx)
            cur_ref = np.vstack([cur_ref, upts])

        if self._extra_sets:
            for set_name, set_n in self._extra_sets.items():
                set_pts = self._kennard_stone(
                    gs, cur_reserved, cur_ref, set_n, gmins, granges,
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

        labels = (
            [SamplingResult.PRESCRIBED] * n_p_in
            + [SamplingResult.FOCUS]    * n_f_in
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
        print()
        print("─" * 60)
        print("  MERGEN — Final Design")
        print("─" * 60)
        print(f"  Prescribed (in)  : {n_p_in}")
        print(f"  Prescribed (out) : {len(prescribed_out_pts)}")
        print(f"  Focus (in)       : {n_f_in}")
        print(f"  Focus (out)      : {len(focus_out_pts)}")
        print(f"  Optimised        : {n_sce}")
        print(f"  Total design     : {len(df_samples)}")
        print(f"  Validation       : {len(df_val)}")
        for sn, sdf in extra_dfs.items():
            print(f"  {sn:<18}: {len(sdf)}")
        print("═" * 60)

        result                    = SamplingResult(df_samples, df_val, space)
        result.sets               = extra_dfs
        result.designs            = {crit_name: df_samples}
        result.algorithm_results  = algorithm_results
        result.best_algorithm     = best_algorithm
        result._meta              = {
            'criteria'        : crit_name,
            'seed'            : seed,
            'algorithm'       : (algorithm_names[0] if len(algorithm_names) == 1
                                 else algorithm_names),
            'best_algorithm'  : best_algorithm,
            'n_parameters'    : n_dims,
            'n_candidates'    : space.n_candidates,
            'n_design'        : len(df_samples),
            'n_validation'    : len(df_val),
            'best_log_score'  : float(np.log(max(best_score, _EPS))),
            'elapsed_sec'     : float(t_elapsed),
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
        generated around them with Kennard-Stone selection, exactly as
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

        t_start = time.perf_counter()
        space   = self.space
        pts, label, lcolor = self._loaded_design

        gmins   = space.gmins
        granges = space.granges
        names   = space.names
        gs      = space.grid_sampler()
        n_dims  = space.n_parameters

        n_validation = (self._n_validation
                        if self._n_validation is not None
                        else max(1, int(np.ceil(len(pts) * 0.20))))

        print()
        print("═" * 60)
        print("  MERGEN — Loaded Design")
        print("═" * 60)
        print(f"  Parameters      : {n_dims}")
        print(f"  Candidates      : {space.n_candidates:,}")
        print(f"  Loaded points   : {len(pts)}  (label='{label}')")
        print(f"  Validation      : {n_validation}")
        print("─" * 60)

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

        # ── Validation set (Kennard-Stone) ─────────────────────────────
        val_pts = self._kennard_stone(
            gs, reserved, pts, n_validation, gmins, granges,
        )
        cur_reserved = reserved.copy()
        for vp in val_pts:
            idx = gs.point_to_index(vp)
            if idx >= 0:
                cur_reserved.add(idx)
        cur_ref = np.vstack([pts, val_pts]) if len(val_pts) else pts

        # ── Extra sets: user-supplied first, then generated ────────────
        extra_dfs: Dict[str, pd.DataFrame] = {}
        for uname, (upts, ucolor) in self._user_sets.items():
            df_u = pd.DataFrame(upts, columns=names)
            df_u['point_type'] = uname
            if ucolor is not None:
                df_u['color'] = ucolor
            df_u.index.name = 'id'
            extra_dfs[uname] = df_u
            cur_ref = np.vstack([cur_ref, upts])
        if self._extra_sets:
            for set_name, set_n in self._extra_sets.items():
                set_pts = self._kennard_stone(
                    gs, cur_reserved, cur_ref, set_n, gmins, granges,
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

        print()
        print("─" * 60)
        print("  MERGEN — Final Design")
        print("─" * 60)
        print(f"  {label:<17}: {len(df_samples)}")
        print(f"  Total design     : {len(df_samples)}")
        print(f"  Validation       : {len(df_val)}")
        for sn, sdf in extra_dfs.items():
            print(f"  {sn:<18}: {len(sdf)}")
        print("═" * 60)

        result                    = SamplingResult(df_samples, df_val, space)
        result.sets               = extra_dfs
        result.designs            = {'loaded': df_samples}
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
        }
        return result

    def compare(
        self,
        criteria:   Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None,
        priority:   Tuple[str, ...] = ('min_distance', 'max_abs_correlation'),
        mc_samples: int = 300,
        seed:       int = 44,
        verbose:    bool = True,
    ) -> "ComparisonResult":
        """
        Run a criterion/algorithm sweep and rank the resulting designs.

        Every (criterion, algorithm) combination is optimised with the
        current sampler configuration. Because raw criterion scores are
        on incomparable scales, designs are ranked on criterion-agnostic
        quality metrics expressed as percentile ranks against a single
        shared Monte Carlo baseline of random designs of the same size
        (Joseph 2016; Pronzato & Mueller 2012).

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
            Quality metrics treated as competing objectives when
            selecting the best design. All are on a common 0-100
            percentile scale (higher is better) and are honoured jointly
            via a Pareto/Utopia rule (Lu, Anderson-Cook & Robinson
            2011): the non-dominated designs are kept, and the one
            closest to the Utopia point (every metric = 100) is chosen.
            No weights are needed and no single metric dominates.
            Available metrics: min_distance, mean_distance, cv_distances,
            minimax, max_abs_correlation, projection_cd2.
        mc_samples : int, default 300
            Size of the shared Monte Carlo baseline.
        seed : int, default 44
            Seed used for every run and for the baseline.
        verbose : bool, default True
            Print progress and the final ranking table.

        Returns
        -------
        ComparisonResult
            ``.table`` (ranked DataFrame of percentile ranks),
            ``.results`` (dict mapping (criterion, algorithm) to the
            full SamplingResult), ``.best`` (winning key) and
            ``.priority``.
        """
        from . import metrics as _m

        # ── Resolve the sweep lists ────────────────────────────────────
        has_nominal = any(self.space.is_nominal(p) for p in self.space.names)
        nominal_ok  = set(nominal_supporting_criteria())
        if criteria is None:
            if has_nominal:
                crit_list = sorted(nominal_ok)
            else:
                crit_list = sorted(set(list_criteria()) - nominal_ok)
        else:
            crit_list = list(criteria)
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
        alg_list = list(algorithms) if algorithms is not None else ['sa']

        for m in priority:
            if m not in _m._METRIC_FN:
                _fatal(f"Unknown priority metric '{m}'. "
                       f"Available: {sorted(_m._METRIC_FN)}")
        metric_names = list(dict.fromkeys(
            list(priority) + ['min_distance', 'minimax',
                              'max_abs_correlation', 'projection_cd2',
                              'cv_distances', 'mean_distance']))

        # ── Run every combination silently ─────────────────────────────
        import contextlib
        import io
        results: Dict[Tuple[str, str], SamplingResult] = {}
        for cr in crit_list:
            for alg in alg_list:
                if verbose:
                    print(f"  [COMPARE]  {cr} + {alg} ...", flush=True)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    res = self.run(criteria=cr, algorithm=alg,
                                   seed=seed, verbose=False)
                results[(cr, alg)] = res

        # ── Shared Monte Carlo baseline ────────────────────────────────
        gs      = self.space.grid_sampler()
        gmins   = self.space.gmins
        granges = self.space.granges
        n_ref   = len(next(iter(results.values())).best_design)
        if verbose:
            print(f"  [COMPARE]  Monte Carlo baseline "
                  f"({mc_samples} random designs, n={n_ref}) ...",
                  flush=True)
        baseline = _m._mc_baseline(
            gs, gmins, granges, n_ref,
            metric_names=metric_names, crit_names=[],
            mc_samples=mc_samples, seed=seed, space=self.space,
        )

        # ── Percentile ranks against the shared baseline ──────────────
        rows = []
        for (cr, alg), res in results.items():
            X = res.best_design[self.space.names].to_numpy(dtype=float)
            X_norm = (X - gmins) / np.where(granges > 1e-12, granges, 1.0)
            row = {'criterion': cr, 'algorithm': alg}
            for m in metric_names:
                val = _m._compute_metric(m, X_norm, space=self.space)
                row[m] = _m._percentile_rank(
                    val, baseline[m], m in _m._HIGHER_BETTER)
            rows.append(row)
        table = pd.DataFrame(rows)

        # ── Pareto-frontier ranking with a Utopia-point selection ─────
        # The priority metrics are treated as competing objectives, all
        # already on a common 0-100 percentile scale (higher is better).
        # Rather than collapsing them with arbitrary weights (a weighted
        # sum favours extreme corner solutions and needs weights that are
        # not natural to set) or with a strict lexicographic order (which
        # lets the first metric dominate and all but ignores the rest),
        # designs are ranked with the Pareto/Utopia approach of Lu,
        # Anderson-Cook & Robinson (2011): keep the non-dominated
        # (Pareto-optimal) designs, then pick the one closest to the
        # Utopia point where every metric reaches 100. This honours every
        # priority metric without a weight choice.
        obj = list(priority)
        pts = table[obj].to_numpy(dtype=float)
        n = len(pts)

        def _dominates(a, b):
            # a dominates b: no worse on any objective, better on at least
            # one (all objectives are "higher is better" here).
            return bool(np.all(a >= b) and np.any(a > b))

        pareto = []
        for i in range(n):
            if not any(_dominates(pts[j], pts[i]) for j in range(n)
                       if j != i):
                pareto.append(i)

        # Utopia point = best achievable on each objective (100 for a
        # percentile). Choose the Pareto design closest to it.
        utopia = np.full(len(obj), 100.0)
        dists = {i: float(np.linalg.norm(utopia - pts[i])) for i in pareto}
        best_pos = min(dists, key=dists.get)
        best_idx = table.index[best_pos]
        best_key = (table.loc[best_idx, 'criterion'],
                    table.loc[best_idx, 'algorithm'])

        # Order rows by production order: criteria as the user listed
        # them, and within each criterion the algorithms in their listed
        # order. This keeps the table (and the heat map) grouped and
        # readable, rather than shuffled by score.
        crit_order = {cr: i for i, cr in enumerate(crit_list)}
        alg_order = {al: i for i, al in enumerate(alg_list)}
        table['_c'] = table['criterion'].map(crit_order)
        table['_a'] = table['algorithm'].map(alg_order)
        table = table.sort_values(['_c', '_a'], ignore_index=True)
        table = table.drop(columns=['_c', '_a'])
        table.insert(0, 'best',
                     [' *' if (r.criterion, r.algorithm) == best_key
                      else '' for r in table.itertuples()])

        cmp = ComparisonResult(table=table, results=results,
                               best=best_key, priority=tuple(priority))
        if verbose:
            cmp.summary()
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
        ``crit_start`` index suffices in :meth:`_run_sce`.
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

        def _overlap(a: np.ndarray, lst: List[np.ndarray]) -> bool:
            return any(np.allclose(a, b) for b in lst)

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

        # Within-role duplicates
        def _check_dupes(pts: List[np.ndarray], label: str) -> None:
            for i, a in enumerate(pts):
                for b in pts[i + 1:]:
                    if np.allclose(a, b):
                        _fatal(
                            f"Point {a.tolist()} appears in {label} more "
                            f"than once. Merge into a single entry."
                        )
        _check_dupes(p_pts, "add_prescribed()")
        _check_dupes(f_pts, "add_focus()")
        _check_dupes(e_pts, "add_exclusion()")

    # ------------------------------------------------------------------ #
    # Private: focus / exclusion sampling                                  #
    # ------------------------------------------------------------------ #

    def _build_focus_exclusion(
        self,
        gs:       GridSampler,
        gmins:    np.ndarray,
        granges:  np.ndarray,
        reserved: set,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], dict, set]:
        """
        Sample focus neighbours and build exclusion repulsion weights.

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
            Sparse repulsion weights keyed by grid index, used by the
            coordinate-exchange candidate selector.
        reserved      : updated set of reserved grid indices
        """
        focus_in_pts:  List[np.ndarray] = []
        focus_out_pts: List[np.ndarray] = []
        focus_sa_pts:  List[np.ndarray] = []
        repel_weights: Dict[int, float] = {}

        def _gaussian_weight(cpt: np.ndarray, centre: np.ndarray,
                             spread: float) -> float:
            """Unnormalised Gaussian weight in normalised space."""
            sigma = spread / max(granges.mean(), 1e-9)
            d2    = float(np.sum(((cpt - centre) / granges) ** 2))
            return float(np.exp(-d2 / (2 * sigma ** 2)))

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
                n_draw = max(fp.n_samples - 1, 0)
            elif fp.include_center is False:
                # Centre is never selected
                if self_idx >= 0 and self_idx not in reserved:
                    reserved.add(self_idx)

            # Gaussian-weighted neighbour sampling
            if n_draw == 0:
                continue
            K          = min(gs.n_candidates - len(reserved),
                             max(n_draw * 50, 2_000))
            candidates: List[Tuple[float, int, np.ndarray]] = []
            seen: set  = set()
            for _ in range(K * 3):
                idx = random.randint(0, gs.n_candidates - 1)
                if idx in reserved or idx in seen:
                    continue
                seen.add(idx)
                cpt = gs.index_to_point(idx)
                w   = _gaussian_weight(cpt, pt, fp.spread) + 1e-12
                candidates.append((w, idx, cpt))
                if len(candidates) >= K:
                    break

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

            chosen = np.random.choice(
                len(candidates), size=n_take, replace=False, p=weights,
            )
            for ci in chosen:
                _, cidx, cpt = candidates[ci]
                reserved.add(cidx)
                (focus_in_pts if fp.in_design else focus_out_pts).append(cpt)
                if fp.in_optim:
                    focus_sa_pts.append(cpt)

        # ── Exclusion points ──────────────────────────────────────────
        for ep in self._exclusions:
            pt       = ep.point
            self_idx = gs.point_to_index(pt)
            if self_idx >= 0 and self_idx not in reserved:
                reserved.add(self_idx)
                _info(f"ExclusionPoint {pt.tolist()}: removed from pool.")

            # Sample a neighbourhood to estimate repulsion weights
            K_repel  = min(gs.n_candidates, 10_000)
            seen     = set()
            local_w  = []
            for _ in range(K_repel * 2):
                idx = random.randint(0, gs.n_candidates - 1)
                if idx in seen:
                    continue
                seen.add(idx)
                cpt = gs.index_to_point(idx)
                w   = _gaussian_weight(cpt, pt, ep.spread)
                if w > 1e-6:
                    local_w.append((idx, w))
                if len(local_w) >= K_repel:
                    break

            if local_w:
                max_w = max(w for _, w in local_w)
                for idx, w in local_w:
                    repel_weights[idx] = repel_weights.get(idx, 0.0) + w / max_w

        return focus_in_pts, focus_out_pts, focus_sa_pts, repel_weights, reserved

    # ------------------------------------------------------------------ #
    # Private: Stochastic Coordinate Exchange (SCE) core                   #
    # ------------------------------------------------------------------ #

    def _kennard_stone(
        self,
        gs:       GridSampler,
        reserved: set,
        selected: np.ndarray,
        n:        int,
        gmins:    np.ndarray,
        granges:  np.ndarray,
        K_step:   int = 5_000,
    ) -> np.ndarray:
        """
        Kennard-Stone holdout selection via :class:`GridSampler`.

        At each step we pick the unreserved grid point farthest (in
        normalised Euclidean distance) from the current reference set.
        For small grids every non-reserved index is iterated exactly;
        for larger grids we sample ``K_step`` candidates per step.

        References
        ----------
        Kennard, R. W. & Stone, L. A. (1969).
            *Technometrics*, 11(1), 137–148.
        """
        if n <= 0:
            return np.empty((0, self.space.n_parameters))

        reserved = reserved.copy()
        ref      = selected.copy()
        chosen:  List[np.ndarray] = []

        for _ in range(n):
            best_pt, best_dist, best_idx = None, -1.0, -1
            norm_ref = (ref - gmins) / granges

            if gs.n_candidates <= GridSampler.FULL_GREEDY_THRESHOLD:
                # Small grid: exhaustive scan
                for idx in range(gs.n_candidates):
                    if idx in reserved:
                        continue
                    pt   = gs.index_to_point(idx)
                    pt_n = (pt - gmins) / granges
                    d    = float(np.min(
                        np.linalg.norm(norm_ref - pt_n, axis=1)))
                    if d > best_dist:
                        best_dist, best_pt, best_idx = d, pt, idx
            else:
                # Large grid: sample K_step candidates
                seen: set = set()
                for _ in range(K_step):
                    idx = random.randint(0, gs.n_candidates - 1)
                    if idx in reserved or idx in seen:
                        continue
                    seen.add(idx)
                    pt   = gs.index_to_point(idx)
                    pt_n = (pt - gmins) / granges
                    d    = float(np.min(
                        np.linalg.norm(norm_ref - pt_n, axis=1)))
                    if d > best_dist:
                        best_dist, best_pt, best_idx = d, pt, idx

            if best_pt is None:
                break
            chosen.append(best_pt)
            ref = np.vstack([ref, best_pt])
            reserved.add(best_idx)

        return (np.array(chosen) if chosen
                else np.empty((0, self.space.n_parameters)))

    def _snapshot_state(self) -> dict:
        """Capture the mutable design-spec state for safe restoration."""
        return {
            'prescribed_len' : len(self._prescribed),
            'focus_len'      : len(self._focus),
            'exclusions_len' : len(self._exclusions),
            'n_samples'      : self._n_samples,
            'n_validation'   : self._n_validation,
        }

    def _restore_state(self, snap: dict) -> None:
        """Roll back any state added after ``snap`` was taken."""
        del self._prescribed[snap['prescribed_len']:]
        del self._focus[snap['focus_len']:]
        del self._exclusions[snap['exclusions_len']:]
        self._n_samples    = snap['n_samples']
        self._n_validation = snap['n_validation']
