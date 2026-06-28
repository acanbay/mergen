"""
mergen.sampler
==============
Sampler       : orchestrates space-filling design via Simulated Annealing.
SamplingResult: container for design, validation, and extra sets.
FocusPoint    : attract sampling toward a critical region.
ExclusionPoint: repel sampling away from a region.

Quick start
-----------
    import numpy as np
    from mergen.space   import ParameterSpace
    from mergen.sampler import Sampler

    space = ParameterSpace({
        'temperature': range(100, 400, 10),
        'pressure':    ('continuous', 0.5, 5.0),
    })

    sampler = Sampler(space)
    sampler.add_prescribed([[200, 2.5]], in_design=True,  in_sa=False)
    sampler.add_focus([350, 4.5],        spread=1.5,      in_design=True, in_sa=True)
    sampler.add_exclusion([100, 0.5],    spread=1.0)
    sampler.set_design(n_samples=30)
    sampler.set_sa(n_restarts=5)

    result = sampler.run(criteria='umaxpro', seed=44)
    result.summary()

References
----------
Vorechovsky & Elias (2026), Computers & Structures.        [uMaxPro]
Joseph, Gul & Ba (2015), Biometrika 102(2).                [MaxPro]
Morris & Mitchell (1995), J. Statist. Plan. Infer. 43.     [SA for LHD]
Kirkpatrick, Gelatt & Vecchi (1983), Science 220.          [SA cooling]
Kennard & Stone (1969), Technometrics 11(1).               [holdout set]
Loeppky, Sacks & Welch (2009), Technometrics 51(4).        [10p rule]
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .space    import ParameterSpace, GridSampler
from .criteria import BaseCriterion, get_criterion

# ── Terminal colours ──────────────────────────────────────────────────────
_RED    = "\033[0;31m"
_GREEN  = "\033[0;32m"
_YELLOW = "\033[1;33m"
_CYAN   = "\033[0;36m"
_RESET  = "\033[0m"

_EPS = 1e-300


def _info(msg: str)  -> None: print(f"  {_CYAN}[INFO]{_RESET}     {msg}")
def _warn(msg: str)  -> None: print(f"  {_YELLOW}[WARNING]{_RESET}  {msg}")
def _ok(msg: str)    -> None: print(f"  {_GREEN}[SA]{_RESET}       {msg}")
def _fatal(msg: str) -> None:
    raise ValueError(f"\n{_RED}[MERGEN ERROR]{_RESET}  {msg}")


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
        None → auto: ``max(1, int((2*d + 1) * spread))``.
    include_center : bool or None
        True  → centre point is guaranteed in the design.
        False → centre is excluded from the pool.
        None  → stochastic (centre competes with neighbours).
    in_design : bool
        True  → focus points count toward ``n_samples`` budget.
        False → focus points are added on top of ``n_samples`` (total grows).
    in_sa : bool
        True  → SA sees these points and pushes optimised points away.
        False → SA ignores these points (they are only reserved).
    """

    def __init__(
        self,
        point:          np.ndarray,
        spread:         float            = 1.0,
        n_samples:      Optional[int]    = None,
        include_center: Optional[bool]   = None,
        in_design:      bool             = True,
        in_sa:          bool             = True,
    ) -> None:
        self.point          = np.asarray(point, dtype=float).ravel()
        self.spread         = float(spread)
        self.n_samples      = n_samples
        self.include_center = include_center
        self.in_design      = bool(in_design)
        self.in_sa          = bool(in_sa)

        if self.spread <= 0:
            _fatal(f"FocusPoint spread must be > 0, got {self.spread}.")
        if self.n_samples is not None and self.n_samples < 1:
            _fatal(f"FocusPoint n_samples must be >= 1, got {self.n_samples}.")

    def resolve_n_samples(self, n_dims: int) -> None:
        """Set n_samples from auto formula if not user-supplied."""
        if self.n_samples is None:
            self.n_samples = max(1, int((2 * n_dims + 1) * self.spread))

    def __repr__(self) -> str:
        ic = {None: 'stochastic', True: 'guaranteed', False: 'excluded'}
        return (
            f"FocusPoint(point={self.point.tolist()}, spread={self.spread}, "
            f"n_samples={self.n_samples}, center={ic[self.include_center]}, "
            f"in_design={self.in_design}, in_sa={self.in_sa})"
        )


# ======================================================================
# ExclusionPoint
# ======================================================================

class ExclusionPoint:
    """
    A region to avoid: the centre is hard-excluded from the candidate
    pool and its neighbourhood receives a Gaussian soft-repulsion penalty
    during SA candidate sampling.

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
        Extra named sets (e.g. ``{'test': df}``).
    designs    : dict[str, pd.DataFrame]
        Design per criterion when multiple criteria are run.
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
        self.output_dir     = "outputs"
        self._meta: dict    = {}

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def summary(self) -> None:
        """Print a concise design summary to stdout."""
        vc  = self.samples["point_type"].value_counts()
        W   = 52
        sep = "─" * W
        print(f"\n{sep}")
        print(f"  MERGEN Design Summary")
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
        print(f"{sep}\n")

    def quality_report(
        self,
        metrics:          Union[str, list]   = 'default',
        criteria_metrics: Optional[list]     = None,
        mc_samples:       int                = 0,
    ) -> dict:
        """
        Compute and print design quality metrics.

        Delegates to :func:`mergen.metrics.quality_report`.

        Parameters
        ----------
        metrics          : 'default' | list of metric names
        criteria_metrics : list of criterion names to evaluate post-hoc
        mc_samples       : number of random designs for MC baseline (0 = off)

        Returns
        -------
        dict of metric values and (optionally) baseline stats
        """
        from . import metrics as _metrics
        return _metrics.quality_report(
            self,
            metrics=metrics,
            criteria_metrics=criteria_metrics,
            mc_samples=mc_samples,
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
        kind : 'pairplot' | '1d' | '2d' | 'distances' | 'quality' | 'all'
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

class Sampler:
    """
    Space-filling design sampler using Simulated Annealing.

    Parameters
    ----------
    space : ParameterSpace

    Examples
    --------
    >>> space = ParameterSpace({'x': range(1, 21), 'y': range(1, 21)})
    >>> sampler = Sampler(space)
    >>> sampler.set_design(n_samples=20)
    >>> result = sampler.run(seed=44)
    >>> result.summary()
    """

    def __init__(self, space: ParameterSpace) -> None:
        if not isinstance(space, ParameterSpace):
            _fatal(f"space must be a ParameterSpace, got {type(space).__name__}.")
        if not space.is_valid():
            _fatal("ParameterSpace has no parameters or no feasible candidates.")

        self.space = space

        # Points
        self._prescribed:  List[Tuple[np.ndarray, bool, bool]] = []
        # Each entry: (point, in_design, in_sa)
        self._focus:       List[FocusPoint]      = []
        self._exclusions:  List[ExclusionPoint]  = []

        # Design settings
        self._n_samples:    Optional[int]         = None
        self._n_validation: Optional[int]         = None
        self._extra_sets:   Optional[Dict]        = None
        self._dim_weights:  Optional[np.ndarray]  = None

        # SA settings
        self._n_restarts:     int            = 5
        self._max_iter:       Optional[int]  = None
        self._init_temp:      Optional[float]= None
        self._cooling:        Optional[float]= None
        self._greedy_polish:  bool           = False

    # ------------------------------------------------------------------ #
    # Public: configuration                                                #
    # ------------------------------------------------------------------ #

    def add_prescribed(
        self,
        points:    Union[Sequence, np.ndarray],
        in_design: bool = True,
        in_sa:     bool = False,
    ) -> "Sampler":
        """
        Add one or more prescribed (fixed) points.

        Parameters
        ----------
        points : array-like — single point or list of points
        in_design : bool
            True  → counted within ``n_samples`` budget.
            False → added on top of ``n_samples`` (total increases).
        in_sa : bool
            True  → SA sees these points and avoids their vicinity.
            False → SA ignores them (points are only reserved).

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
            self._prescribed.append((validated, bool(in_design), bool(in_sa)))
        return self

    def add_focus(
        self,
        point:          Sequence[float],
        spread:         float          = 1.0,
        n_samples:      Optional[int]  = None,
        include_center: Optional[bool] = None,
        in_design:      bool           = True,
        in_sa:          bool           = True,
    ) -> "Sampler":
        """
        Add a focus point — denser sampling near a critical region.

        Parameters
        ----------
        point          : grid coordinates of the focus centre
        spread         : Gaussian kernel width in normalised grid steps
        n_samples      : number of focus samples (None → auto)
        include_center : True/False/None (guaranteed/excluded/stochastic)
        in_design      : True → within budget; False → extra on top
        in_sa          : True → SA sees these points

        Returns
        -------
        self
        """
        pt = self.space.validate_point(point, label="Focus point")
        self._focus.append(FocusPoint(
            pt, spread=spread, n_samples=n_samples,
            include_center=include_center,
            in_design=in_design, in_sa=in_sa,
        ))
        return self

    def add_exclusion(
        self,
        point:  Sequence[float],
        spread: float = 1.0,
    ) -> "Sampler":
        """
        Add an exclusion point — sampling avoids this region.

        The centre is hard-excluded from the candidate pool.
        Its neighbourhood receives a Gaussian soft-repulsion weight
        during SA candidate sampling.

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
        n_samples    : total design size (None → 10 × n_parameters)
        n_validation : validation set size (None → 20% of n_samples)
        extra_sets   : dict of {name: size} for additional Kennard-Stone sets
                       e.g. ``{'test': 10, 'holdout': 5}``

        Returns
        -------
        self
        """
        self._n_samples    = n_samples
        self._n_validation = n_validation
        self._extra_sets   = extra_sets
        return self

    def set_sa(
        self,
        n_restarts:    int            = 5,
        max_iter:      Optional[int]  = None,
        init_temp:     Optional[float]= None,
        cooling:       Optional[float]= None,
        greedy_polish: bool           = False,
    ) -> "Sampler":
        """
        Configure Simulated Annealing hyperparameters.

        Parameters
        ----------
        n_restarts    : number of independent SA restarts (best is kept)
        max_iter      : SA iterations per restart (None → auto)
        init_temp     : initial temperature (None → Kirkpatrick auto-tune)
        cooling       : geometric cooling rate (None → auto)
        greedy_polish : single greedy improvement pass after SA

        Returns
        -------
        self
        """
        self._n_restarts    = max(1, int(n_restarts))
        self._max_iter      = max_iter
        self._init_temp     = init_temp
        self._cooling       = cooling
        self._greedy_polish = bool(greedy_polish)
        return self

    def set_dimension_weights(
        self,
        weights: Union[Dict[str, float], List[float]],
    ) -> "Sampler":
        """
        Set per-dimension importance weights for greedy maximin seeding.

        Parameters
        ----------
        weights : dict {name: weight} or list of floats (one per parameter)

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

    # ------------------------------------------------------------------ #
    # Public: run                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        criteria: Union[str, List[str], BaseCriterion] = 'umaxpro',
        seed:     Optional[int] = 44,
        n_cores:  int           = 1,
    ) -> SamplingResult:
        """
        Generate the space-filling design.

        Parameters
        ----------
        criteria : str, list of str, or BaseCriterion instance
            Optimisation criterion (or list of criteria).
            Available: ``'umaxpro'``, ``'maxpro'``, ``'phi_p'``,
            ``'cd2'``, ``'stratified'``.
            Default: ``'umaxpro'``.
        seed     : int or None — random seed (default 44)
        n_cores  : int — parallel restarts (-1 = all cores, -2 = all-1)

        Returns
        -------
        SamplingResult
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Resolve criterion(ia)
        if isinstance(criteria, str):
            crit_list = [get_criterion(criteria)]
            crit_names = [criteria]
        elif isinstance(criteria, list):
            crit_list  = [get_criterion(c) if isinstance(c, str) else c
                          for c in criteria]
            crit_names = [c if isinstance(c, str) else repr(c)
                          for c in criteria]
        else:
            crit_list  = [criteria]
            crit_names = [repr(criteria)]

        # Resolve auto settings
        space   = self.space
        n_dims  = space.n_parameters
        gmins   = space.gmins
        granges = space.granges
        names   = space.names
        gs      = space.grid_sampler()

        # Focus n_samples auto-resolve
        for fp in self._focus:
            fp.resolve_n_samples(n_dims)

        # Conflict check
        self._check_conflicts()

        # Budget
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
                    f"n_samples ({n_global}) < recommended 10×n_parameters "
                    f"({loeppky}). Design quality may be reduced."
                )

        # SA optimises this many slots
        n_sa_slots = n_global - n_prescribed_in - n_focus_in
        if n_sa_slots < 1:
            _fatal(
                f"No SA slots remaining: n_samples={n_global}, "
                f"prescribed_in={n_prescribed_in}, focus_in={n_focus_in}. "
                f"Increase n_samples or set in_design=False for some points."
            )

        n_validation = (self._n_validation
                        if self._n_validation is not None
                        else max(1, int(np.ceil(n_global * 0.20))))

        # Print status
        n_total_design = (n_global
                          + n_prescribed_out
                          + n_focus_out)
        print()
        print("═" * 60)
        print("  MERGEN — Space-filling Design")
        print("═" * 60)
        print(f"  Parameters      : {n_dims}")
        print(f"  Candidates      : {space.n_candidates:,}")
        print(f"  n_samples       : {n_global}  "
              f"(prescribed_in={n_prescribed_in}, "
              f"focus_in={n_focus_in}, "
              f"sa_slots={n_sa_slots})")
        if n_prescribed_out or n_focus_out:
            print(f"  Extra points    : +{n_prescribed_out + n_focus_out}  "
                  f"(prescribed_out={n_prescribed_out}, "
                  f"focus_out={n_focus_out})")
        print(f"  Total design    : {n_total_design}")
        print(f"  Validation      : {n_validation}")
        print(f"  Criterion       : {crit_names[0]}")
        print(f"  SA restarts     : {self._n_restarts}")
        print("─" * 60)

        # ── Reserved set ───────────────────────────────────────────────
        reserved: set = set()

        # Prescribed points → reserve
        prescribed_in_pts  = []
        prescribed_out_pts = []
        prescribed_sa_pts  = []
        for pt, in_d, in_s in self._prescribed:
            idx = gs.point_to_index(pt)
            if idx >= 0:
                reserved.add(idx)
            if in_d:
                prescribed_in_pts.append(pt)
            else:
                prescribed_out_pts.append(pt)
            if in_s:
                prescribed_sa_pts.append(pt)

        # ── Focus sampling ─────────────────────────────────────────────
        focus_in_pts, focus_out_pts, focus_sa_pts, repel_weights, reserved = \
            self._build_focus_exclusion(gs, gmins, granges, reserved)

        # ── Greedy maximin seed ────────────────────────────────────────
        # Anchor = points visible to SA (in_sa=True) that are IN the design
        # out-of-design points are reserved but NOT used as anchor
        sa_anchor_parts = []
        if prescribed_sa_pts:
            # Only in_design=True + in_sa=True prescribed points as anchor
            in_design_sa = [pt for pt, in_d, in_s in self._prescribed
                            if in_d and in_s]
            if in_design_sa:
                sa_anchor_parts.append(np.array(in_design_sa, dtype=float))
        if focus_sa_pts:
            in_design_focus_sa = [fp_pt for fp_pt, fp in
                                   zip(focus_sa_pts,
                                       [fp for fp in self._focus if fp.in_sa])
                                   if fp.in_design]
            if in_design_focus_sa:
                sa_anchor_parts.append(np.array(in_design_focus_sa, dtype=float))

        _corner_used = False
        if sa_anchor_parts:
            anchor = np.vstack(sa_anchor_parts)
        else:
            # Corner start — counts as one SA slot
            corner = np.array([v[0] if i % 2 == 0 else v[-1]
                                for i, v in enumerate(space.values)])
            idx = gs.point_to_index(corner)
            if idx >= 0:
                reserved.add(idx)
            anchor       = corner[np.newaxis, :]
            _corner_used = True

        # Greedy seed fills remaining SA slots
        # anchor rows are already selected; seed_design will contain anchor + new
        seed_design, reserved = gs.greedy_maximin_seed(
            anchor,
            n_sa_slots - (1 if _corner_used else 0),
            reserved, self._dim_weights
        )

        # SA-optimised points = seed_design minus the anchor rows
        # seed_design = [anchor_rows | new_greedy_rows]
        n_anchor   = len(anchor)
        sa_only    = seed_design[n_anchor:] if len(seed_design) > n_anchor \
                     else np.empty((0, n_dims))

        parts = []
        if prescribed_in_pts:
            parts.append(np.array(prescribed_in_pts, dtype=float))
        if focus_in_pts:
            parts.append(np.array(focus_in_pts, dtype=float))
        # Add anchor (corner or prescribed/focus SA points) + greedy SA points
        parts.append(seed_design)   # full seed: anchor + greedy
        full_design = np.vstack(parts) if parts else seed_design

        # n_frozen: rows that SA does NOT swap
        n_frozen = (len(prescribed_in_pts)
                    + len(focus_in_pts))

        # SA scope start: which rows enter the criterion
        # in_sa=True prescribed/focus → SA sees them; others → frozen-blind
        n_sa_visible = len(prescribed_sa_pts) + len(focus_sa_pts)
        if n_sa_visible > 0:
            crit_start = (len(prescribed_in_pts)
                          - sum(1 for _, in_d, in_s in self._prescribed
                                if in_d and not in_s))
        else:
            crit_start = n_frozen

        # ── Runtime estimate ───────────────────────────────────────────
        self._print_runtime_estimate(full_design, gs, reserved,
                                     gmins, granges, n_frozen, crit_start,
                                     crit_list[0])

        # ── SA with restarts ───────────────────────────────────────────
        best_design   = None
        best_reserved = None
        best_score    = float('inf')

        for r_idx in range(self._n_restarts):
            if self._n_restarts > 1:
                # Fresh greedy seed for each restart
                r_reserved = set(
                    gs.point_to_index(pt)
                    for pt, _, _ in self._prescribed
                    if gs.point_to_index(pt) >= 0
                )
                for pts_list in [focus_in_pts, focus_out_pts]:
                    for pt in pts_list:
                        idx = gs.point_to_index(pt)
                        if idx >= 0:
                            r_reserved.add(idx)

                corner = np.array(
                    [v[0] if i % 2 == r_idx % 2 else v[-1]
                     for i, v in enumerate(space.values)]
                )
                r_anchor = anchor if sa_anchor_parts else corner[np.newaxis, :]
                if not sa_anchor_parts:
                    idx_ = gs.point_to_index(corner)
                    if idx_ >= 0:
                        r_reserved.add(idx_)

                r_seed, r_reserved = gs.greedy_maximin_seed(
                    r_anchor,
                    n_sa_slots - (0 if sa_anchor_parts else 1),
                    r_reserved, self._dim_weights
                )
                r_sa_only = r_seed[len(r_anchor):]
                r_parts   = []
                if prescribed_in_pts:
                    r_parts.append(np.array(prescribed_in_pts, dtype=float))
                if focus_in_pts:
                    r_parts.append(np.array(focus_in_pts, dtype=float))
                r_parts.append(r_sa_only)
                r_design = np.vstack(r_parts) if r_parts else r_sa_only

                if len(r_design) != len(full_design):
                    r_design   = full_design.copy()
                    r_reserved = reserved.copy()
            else:
                r_design   = full_design.copy()
                r_reserved = reserved.copy()

            if self._n_restarts > 1:
                _ok(f"Restart {r_idx + 1}/{self._n_restarts}")

            out_design, out_reserved = self._run_sa(
                r_design, gs, r_reserved, gmins, granges,
                n_frozen    = n_frozen,
                crit_start  = crit_start,
                criterion   = crit_list[0],
                repel_weights = repel_weights,
            )

            # Score this restart
            X_norm = (out_design[crit_start:] - gmins) / granges
            score  = crit_list[0].evaluate(X_norm, space)
            if score < best_score:
                best_score   = score
                best_design  = out_design
                best_reserved= out_reserved
                if self._n_restarts > 1:
                    _ok(f"Restart {r_idx + 1}: new best "
                        f"log(score)={np.log(max(score, _EPS)):.3f}")

        if self._n_restarts > 1:
            _ok(f"Best across {self._n_restarts} restarts: "
                f"log(score)={np.log(max(best_score, _EPS)):.3f}")

        final_design   = best_design
        final_reserved = best_reserved

        # ── Append out-of-budget points ────────────────────────────────
        extra_fixed = []
        extra_labels= []
        for pt in prescribed_out_pts:
            extra_fixed.append(pt)
            extra_labels.append(self.PRESCRIBED if hasattr(self, 'PRESCRIBED')
                                else SamplingResult.PRESCRIBED)
        for pt in focus_out_pts:
            extra_fixed.append(pt)
            extra_labels.append(SamplingResult.FOCUS)

        # ── Validation set (Kennard-Stone) ─────────────────────────────
        val_pts = self._kennard_stone(
            gs, final_reserved, final_design, n_validation, gmins, granges
        )
        val_reserved = final_reserved.copy()
        for vp in val_pts:
            idx = gs.point_to_index(vp)
            if idx >= 0:
                val_reserved.add(idx)

        # ── Extra sets (Kennard-Stone) ─────────────────────────────────
        extra_dfs   = {}
        cur_reserved = val_reserved.copy()
        cur_ref      = (np.vstack([final_design, val_pts])
                        if len(val_pts) else final_design)
        if self._extra_sets:
            for set_name, set_n in self._extra_sets.items():
                set_pts = self._kennard_stone(
                    gs, cur_reserved, cur_ref, set_n, gmins, granges
                )
                for sp in set_pts:
                    idx = gs.point_to_index(sp)
                    if idx >= 0:
                        cur_reserved.add(idx)
                cur_ref = np.vstack([cur_ref, set_pts]) if len(set_pts) else cur_ref

                df_set = pd.DataFrame(set_pts, columns=names) if len(set_pts) \
                         else pd.DataFrame(columns=names)
                df_set['point_type'] = set_name
                df_set.index.name = 'id'
                extra_dfs[set_name] = df_set

        # ── Build DataFrames ───────────────────────────────────────────
        n_p_in   = len(prescribed_in_pts)
        n_f_in   = len(focus_in_pts)
        n_sa     = len(final_design) - n_p_in - n_f_in

        labels = (
            [SamplingResult.PRESCRIBED] * n_p_in
            + [SamplingResult.FOCUS]    * n_f_in
            + [SamplingResult.OPTIMISED]* n_sa
        )

        # Append out-of-budget points
        if extra_fixed:
            final_with_extra = np.vstack([final_design,
                                          np.array(extra_fixed, dtype=float)])
            labels += extra_labels
        else:
            final_with_extra = final_design

        df_samples = pd.DataFrame(final_with_extra, columns=names)
        df_samples['point_type'] = labels
        df_samples.index.name = 'id'

        df_val = pd.DataFrame(val_pts, columns=names) if len(val_pts) \
                 else pd.DataFrame(columns=names + ['point_type'])
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
        print(f"  Optimised        : {n_sa}")
        print(f"  Total design     : {len(df_samples)}")
        print(f"  Validation       : {len(df_val)}")
        for sn, sdf in extra_dfs.items():
            print(f"  {sn:<18}: {len(sdf)}")
        print("═" * 60)

        result          = SamplingResult(df_samples, df_val, space)
        result.sets     = extra_dfs
        result.designs  = {crit_names[0]: df_samples}
        result._meta    = {
            'criteria'       : crit_names[0],
            'seed'           : seed,
            'n_restarts'     : self._n_restarts,
            'n_parameters'   : n_dims,
            'n_candidates'   : space.n_candidates,
            'n_design'       : len(df_samples),
            'n_validation'   : len(df_val),
            'best_log_score' : float(np.log(max(best_score, _EPS))),
        }
        return result

    # ------------------------------------------------------------------ #
    # Private: conflict detection                                          #
    # ------------------------------------------------------------------ #

    def _check_conflicts(self) -> None:
        """Raise if any point appears in more than one category."""
        p_pts = [pt for pt, _, _ in self._prescribed]
        f_pts = [fp.point for fp in self._focus]
        e_pts = [ep.point for ep in self._exclusions]

        def _overlap(a, lst):
            return any(np.allclose(a, b) for b in lst)

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
                    f"and add_exclusion()."
                )
            if _overlap(ep, f_pts):
                _fatal(
                    f"Point {ep.tolist()} appears in both add_focus() "
                    f"and add_exclusion()."
                )

    # ------------------------------------------------------------------ #
    # Private: focus / exclusion sampling                                  #
    # ------------------------------------------------------------------ #

    def _build_focus_exclusion(
        self, gs, gmins, granges, reserved
    ):
        """
        Sample focus neighbours and build exclusion repulsion weights.

        Returns
        -------
        focus_in_pts  : list of np.ndarray — in_design=True focus points
        focus_out_pts : list of np.ndarray — in_design=False focus points
        focus_sa_pts  : list of np.ndarray — in_sa=True focus points
        repel_weights : dict {index: weight} — sparse repulsion
        reserved      : updated set
        """
        focus_in_pts  = []
        focus_out_pts = []
        focus_sa_pts  = []
        repel_weights = {}

        def _gaussian_w(cpt, centre, spread):
            sigma = spread / max(granges.mean(), 1e-9)
            d2    = np.sum(((cpt - centre) / granges) ** 2)
            return float(np.exp(-d2 / (2 * sigma ** 2))) + 1e-8

        # ── Focus points ──────────────────────────────────────────────
        for fp in self._focus:
            pt       = fp.point
            self_idx = gs.point_to_index(pt)
            n_draw   = fp.n_samples

            if fp.include_center is True:
                if self_idx >= 0 and self_idx not in reserved:
                    reserved.add(self_idx)
                    pts_list = focus_in_pts if fp.in_design else focus_out_pts
                    pts_list.append(pt.copy())
                    if fp.in_sa:
                        focus_sa_pts.append(pt.copy())
                n_draw = max(fp.n_samples - 1, 0)
            elif fp.include_center is False:
                if self_idx >= 0 and self_idx not in reserved:
                    reserved.add(self_idx)

            # Gaussian-weighted neighbour sampling
            K          = min(gs.n_candidates - len(reserved),
                             max(n_draw * 50, 2_000))
            candidates = []
            seen: set  = set()

            for _ in range(K * 3):
                idx = random.randint(0, gs.n_candidates - 1)
                if idx in reserved or idx in seen:
                    continue
                seen.add(idx)
                cpt = gs.index_to_point(idx)
                w   = _gaussian_w(cpt, pt, fp.spread)
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
                    f"FocusPoint {pt.tolist()}: requested {n_draw} neighbours "
                    f"but only {n_take} available."
                )

            chosen = np.random.choice(len(candidates), size=n_take,
                                      replace=False, p=weights)
            for ci in chosen:
                _, cidx, cpt = candidates[ci]
                reserved.add(cidx)
                pts_list = focus_in_pts if fp.in_design else focus_out_pts
                pts_list.append(cpt)
                if fp.in_sa:
                    focus_sa_pts.append(cpt)

        # ── Exclusion points ──────────────────────────────────────────
        for ep in self._exclusions:
            pt       = ep.point
            self_idx = gs.point_to_index(pt)
            if self_idx >= 0 and self_idx not in reserved:
                reserved.add(self_idx)
                _info(f"ExclusionPoint {pt.tolist()}: removed from pool.")

            K_repel  = min(gs.n_candidates, 10_000)
            seen: set = set()
            local_w  = []
            for _ in range(K_repel * 2):
                idx = random.randint(0, gs.n_candidates - 1)
                if idx in seen:
                    continue
                seen.add(idx)
                cpt = gs.index_to_point(idx)
                sigma = ep.spread / max(granges.mean(), 1e-9)
                d2    = np.sum(((cpt - pt) / granges) ** 2)
                w     = float(np.exp(-d2 / (2 * sigma ** 2)))
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
    # Private: runtime estimate                                            #
    # ------------------------------------------------------------------ #

    def _print_runtime_estimate(
        self, design, gs, reserved, gmins, granges,
        n_frozen, crit_start, criterion
    ) -> None:
        """
        Estimate SA runtime via Kirkpatrick auto-tuning probe.
        Prints estimate without blocking.
        """
        n_swap = len(design) - n_frozen
        if n_swap == 0:
            return

        X_norm     = (design[crit_start:] - gmins) / granges
        score      = criterion.evaluate(X_norm, self.space)
        max_iter   = self._max_iter or max(2_000, 100 * len(design))

        t0 = time.perf_counter()
        n_probe = min(20, n_swap)
        for _ in range(n_probe):
            i      = random.randint(n_frozen, len(design) - 1)
            new_pt_raw, _ = gs.random_point_excluding(reserved)
            if new_pt_raw is None:
                continue
            new_pt = (new_pt_raw - gmins) / granges
            i_rel  = i - crit_start
            if 0 <= i_rel < len(X_norm):
                criterion.incremental(X_norm, i_rel, new_pt, self.space, score)

        t_probe  = time.perf_counter() - t0
        t_iter   = t_probe / max(n_probe, 1)
        t_restart= t_iter * max_iter
        t_total  = t_restart * self._n_restarts

        def _fmt(sec):
            if sec < 60:   return f"{sec:.0f}s"
            if sec < 3600: return f"{sec/60:.1f} min"
            return f"{sec/3600:.1f} hr"

        _ok(
            f"Estimated runtime: ~{_fmt(t_total)}  "
            f"({self._n_restarts} restart(s) × ~{_fmt(t_restart)} each)"
        )

    # ------------------------------------------------------------------ #
    # Private: Simulated Annealing                                         #
    # ------------------------------------------------------------------ #

    def _auto_temp(
        self, design, gs, reserved, gmins, granges,
        crit_start, criterion,
        n_probe: int = 50, target_accept: float = 0.80,
    ):
        """
        Kirkpatrick et al. (1983) automatic T_start.

        Samples *n_probe* random swaps, collects positive log-deltas,
        and sets T_start so that the acceptance probability for a
        typical worsening move equals *target_accept*.

        References
        ----------
        Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P. (1983).
            *Science*, 220, 671–680.
        """
        X_norm    = (design[crit_start:] - gmins) / granges
        score     = criterion.evaluate(X_norm, self.space)
        n         = len(design)
        pos_deltas= []

        for _ in range(n_probe):
            i          = random.randint(max(crit_start, 0), n - 1)
            new_raw, _ = gs.random_point_excluding(reserved)
            if new_raw is None:
                continue
            new_pt  = (new_raw - gmins) / granges
            i_rel   = i - crit_start
            if not (0 <= i_rel < len(X_norm)):
                continue
            ld, _   = criterion.incremental(X_norm, i_rel, new_pt,
                                             self.space, score)
            if ld > 0:
                pos_deltas.append(ld)

        if pos_deltas:
            T_start = -np.mean(pos_deltas) / np.log(target_accept)
        else:
            T_start = 1.0

        max_iter = max(2_000, 100 * n)
        T_end    = 1e-4 * T_start
        cooling  = (T_end / T_start) ** (1.0 / max_iter)
        return float(T_start), float(cooling), int(max_iter), float(score)

    def _run_sa(
        self,
        design,
        gs,
        reserved,
        gmins,
        granges,
        n_frozen:      int,
        crit_start:    int,
        criterion:     BaseCriterion,
        repel_weights: dict,
    ):
        """
        Core SA loop: coordinate swap on the discrete grid.

        Parameters
        ----------
        design        : np.ndarray (n, d) — raw (unnormalised) design
        gs            : GridSampler
        reserved      : set of reserved grid indices
        gmins, granges: normalisation arrays
        n_frozen      : number of rows SA must not swap
        crit_start    : first row entering the criterion
        criterion     : BaseCriterion instance
        repel_weights : dict {idx: weight} — soft exclusion repulsion

        Returns
        -------
        best_design : np.ndarray
        reserved    : updated set
        """
        design   = design.copy()
        reserved = reserved.copy()
        n_swap   = len(design) - n_frozen

        if n_swap == 0:
            _warn("Nothing to swap — all points are frozen. Skipping SA.")
            return design, reserved

        # Normalised view for criterion
        X_norm = (design[crit_start:] - gmins) / granges

        # Auto-tune temperature
        T_start, cooling, max_iter, raw_score = self._auto_temp(
            design, gs, reserved, gmins, granges, crit_start, criterion
        )
        if self._init_temp  is not None: T_start  = self._init_temp
        if self._cooling    is not None: cooling   = self._cooling
        if self._max_iter   is not None: max_iter  = self._max_iter

        best_score  = raw_score
        best_design = design.copy()
        T           = T_start

        # Soft repulsion: probabilistically add highly-weighted indices
        repel_reserved: set = set()
        if repel_weights:
            for idx, w in repel_weights.items():
                if w > 0.5 and random.random() < w:
                    repel_reserved.add(idx)

        _ok(
            f"Start  log(score)={np.log(max(raw_score, _EPS)):.3f}  "
            f"T={T_start:.4f}  iters={max_iter}  swappable={n_swap}"
        )

        log_interval = max(max_iter // 10, 500)

        for it in range(max_iter):
            # Pick a swappable row
            i = random.randint(n_frozen, len(design) - 1)

            # Draw candidate
            excl          = reserved | repel_reserved
            new_raw, new_idx = gs.random_point_excluding(excl)
            if new_raw is None:
                continue

            # Duplicate check
            if np.any(np.all(np.abs(design - new_raw) < 1e-9, axis=1)):
                continue

            # Criterion incremental update
            new_pt = (new_raw - gmins) / granges
            i_rel  = i - crit_start
            if not (0 <= i_rel < len(X_norm)):
                continue

            log_delta, new_score = criterion.incremental(
                X_norm, i_rel, new_pt, self.space, raw_score
            )

            # Metropolis acceptance
            if (log_delta < 0
                    or random.random() < np.exp(
                        float(np.clip(-log_delta / T, -700, 0)))):
                old_idx = gs.point_to_index(design[i])
                reserved.discard(old_idx)
                reserved.add(new_idx)
                design[i]     = new_raw
                X_norm[i_rel] = new_pt
                raw_score     = new_score
                if raw_score < best_score:
                    best_score  = raw_score
                    best_design = design.copy()

            T *= cooling

            if (it + 1) % log_interval == 0:
                print(f"    iter {it+1:>{len(str(max_iter))}}/{max_iter}  "
                      f"T={T:.5f}  "
                      f"best log(score)={np.log(max(best_score, _EPS)):.3f}")

        _ok(
            f"Done   log(score)={np.log(max(best_score, _EPS)):.3f}  "
            f"(raw={best_score:.4e})"
        )

        # Optional greedy polish
        if self._greedy_polish:
            best_design, reserved = self._greedy_polish_pass(
                best_design, gs, reserved, gmins, granges,
                n_frozen, crit_start, criterion, best_score
            )

        return best_design, reserved

    # ------------------------------------------------------------------ #
    # Private: greedy polish                                               #
    # ------------------------------------------------------------------ #

    def _greedy_polish_pass(
        self, design, gs, reserved, gmins, granges,
        n_frozen, crit_start, criterion, current_score
    ):
        _ok("Greedy polish pass...")
        design    = design.copy()
        X_norm    = (design[crit_start:] - gmins) / granges
        raw_score = current_score
        K_polish  = min(gs.n_candidates - len(reserved),
                        max(5_000, 20 * len(design)))

        for i in range(n_frozen, len(design)):
            best_pt, best_idx_p, best_delta = None, -1, 0.0
            i_rel = i - crit_start
            if not (0 <= i_rel < len(X_norm)):
                continue
            seen: set = set()
            for _ in range(K_polish):
                cand_raw, cand_idx = gs.random_point_excluding(
                    reserved | seen)
                if cand_raw is None:
                    break
                seen.add(cand_idx)
                cand_pt   = (cand_raw - gmins) / granges
                ld, _     = criterion.incremental(
                    X_norm, i_rel, cand_pt, self.space, raw_score)
                if ld < best_delta:
                    best_delta, best_pt, best_idx_p = ld, (cand_raw, cand_pt), cand_idx

            if best_pt is not None:
                old_idx = gs.point_to_index(design[i])
                reserved.discard(old_idx)
                reserved.add(best_idx_p)
                design[i]     = best_pt[0]
                X_norm[i_rel] = best_pt[1]
                raw_score     = criterion.evaluate(X_norm, self.space)

        _ok(f"After polish: log(score)={np.log(max(raw_score, _EPS)):.3f}")
        return design, reserved

    # ------------------------------------------------------------------ #
    # Private: Kennard-Stone                                               #
    # ------------------------------------------------------------------ #

    def _kennard_stone(
        self,
        gs,
        reserved: set,
        selected: np.ndarray,
        n:        int,
        gmins:    np.ndarray,
        granges:  np.ndarray,
        K_step:   int = 5_000,
    ) -> np.ndarray:
        """
        Kennard-Stone holdout selection via GridSampler.

        At each step, samples up to *K_step* random candidates from the
        non-reserved pool and picks the one farthest from all previously
        selected points.

        For small grids (≤ GridSampler.FULL_GREEDY_THRESHOLD) every
        non-reserved index is iterated exactly.

        References
        ----------
        Kennard, R. W. & Stone, L. A. (1969).
            *Technometrics*, 11(1), 137–148.
        """
        if n <= 0:
            return np.empty((0, self.space.n_parameters))

        reserved = reserved.copy()
        ref      = selected.copy()
        chosen   = []

        for _ in range(n):
            best_pt, best_dist, best_idx = None, -1.0, -1
            norm_ref = (ref - gmins) / granges

            if gs.n_candidates <= GridSampler.FULL_GREEDY_THRESHOLD:
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

        return np.array(chosen) if chosen else np.empty(
            (0, self.space.n_parameters))

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"Sampler("
            f"space={self.space.n_parameters}D, "
            f"prescribed={len(self._prescribed)}, "
            f"focus={len(self._focus)}, "
            f"exclusions={len(self._exclusions)})"
        )