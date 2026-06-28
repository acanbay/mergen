"""
mergen.space
============
ParameterSpace  : defines the parameter grid for the sampling design.
GridSampler     : memory-efficient bijective grid representation.

Supported parameter types
-------------------------
Discrete (explicit values):
    [100, 200, 300]                          list / tuple
    range(10, 110, 10)                       Python range
    np.arange(0.1, 1.1, 0.1)               numpy array
    np.linspace(0, 1, 10)                   any 1-D iterable

Continuous (Mergen builds the grid):
    ('continuous', 0.5, 5.0)                linear grid
    ('continuous', 1e-4, 1e-1, 'log')       log-spaced grid

Integer (Mergen builds the grid):
    ('integer', 2, 10)                       all integers in [min, max]
    ('integer', 8, 256, 'log')              log-spaced integers (rounded, unique)

Usage
-----
    # Dict — all at once
    space = ParameterSpace({
        'voltage':  [100, 200, 300, 400],
        'pressure': ('continuous', 0.5, 5.0),
        'lr':       ('continuous', 1e-4, 1e-1, 'log'),
        'n_layers': ('integer', 2, 10),
        'batch':    ('integer', 8, 256, 'log'),
    })

    # Fluent — one at a time
    space = ParameterSpace()
    space.add_parameter('voltage', [100, 200, 300])
    space.add_parameter('pressure', ('continuous', 0.5, 5.0))

    # Constrained
    space.add_constraint(lambda p: p['x'] + p['y'] <= 10)

References
----------
Morris & Mitchell (1995), J. Statist. Plan. Infer. 43.   [greedy maximin seed]
"""

from __future__ import annotations

import itertools
import random
import warnings
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# ── Terminal colours (degrades gracefully on plain terminals) ─────────────
_RED    = "\033[0;31m"
_YELLOW = "\033[1;33m"
_RESET  = "\033[0m"


def _warn(msg: str) -> None:
    """User-facing warning — yellow prefix, always visible."""
    warnings.warn(f"\n{_YELLOW}[MERGEN WARNING]{_RESET}  {msg}",
                  UserWarning, stacklevel=3)


def _fatal(msg: str) -> None:
    """Unrecoverable error — red prefix."""
    raise ValueError(f"\n{_RED}[MERGEN ERROR]{_RESET}  {msg}")


# ── Default resolution for continuous / integer log parameters ────────────
_DEFAULT_RESOLUTION = 100   # overridden to max(n_samples*10, 100) at run time


# ======================================================================
# Parameter type parsing
# ======================================================================

def _parse_parameter(
    name: str,
    spec,
    resolution: int = _DEFAULT_RESOLUTION,
) -> np.ndarray:
    """
    Convert any supported parameter specification to a sorted 1-D float array.

    Parameters
    ----------
    name       : parameter name (for error messages)
    spec       : one of the supported specification types (see module docstring)
    resolution : grid size for continuous / integer-log parameters

    Returns
    -------
    arr : np.ndarray, dtype=float, ndim=1, len >= 1
    """

    # ── Tuple specs: ('continuous'|'integer', min, max [, 'log'] [, opts]) ──
    if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[0], str):
        kind = spec[0].lower().strip()

        if kind not in ('continuous', 'integer'):
            _fatal(
                f"Parameter '{name}': unknown type '{spec[0]}'. "
                f"Use 'continuous' or 'integer', or pass a list/array of values."
            )

        if len(spec) < 3:
            _fatal(
                f"Parameter '{name}': tuple spec requires at least "
                f"('{kind}', min, max), got {spec}."
            )

        lo, hi = float(spec[1]), float(spec[2])

        if lo >= hi:
            _fatal(
                f"Parameter '{name}': min ({lo}) must be strictly less than "
                f"max ({hi})."
            )

        # Scale and optional resolution override
        scale = 'linear'
        res   = resolution
        for item in spec[3:]:
            if isinstance(item, str):
                if item.lower() in ('log', 'log10', 'logarithmic'):
                    scale = 'log'
                else:
                    _fatal(
                        f"Parameter '{name}': unrecognised scale '{item}'. "
                        f"Use 'log' for logarithmic spacing."
                    )
            elif isinstance(item, dict):
                res = int(item.get('resolution', res))

        if lo <= 0 and scale == 'log':
            _fatal(
                f"Parameter '{name}': log scale requires min > 0, got {lo}."
            )

        # Build grid
        if kind == 'continuous':
            if scale == 'log':
                arr = np.logspace(np.log10(lo), np.log10(hi), res)
            else:
                arr = np.linspace(lo, hi, res)

        else:  # integer
            if scale == 'log':
                raw = np.logspace(np.log10(max(lo, 1)), np.log10(hi), res)
                arr = np.unique(np.round(raw)).astype(float)
                arr = arr[(arr >= lo) & (arr <= hi)]
            else:
                arr = np.arange(int(round(lo)), int(round(hi)) + 1, dtype=float)

        if len(arr) == 0:
            _fatal(
                f"Parameter '{name}': grid is empty after construction "
                f"(min={lo}, max={hi}, scale={scale}). "
                f"Check your bounds."
            )

        if len(arr) == 1:
            _warn(
                f"Parameter '{name}' has only 1 grid level. "
                f"It will not contribute to space-filling."
            )

        return arr.astype(float)

    # ── Discrete: list, tuple (of numbers), range, np.ndarray, any iterable ──
    try:
        # Tuples of numbers fall here (not caught above since spec[0] is numeric)
        arr = np.asarray(list(spec), dtype=float)
    except (TypeError, ValueError) as exc:
        _fatal(
            f"Parameter '{name}': cannot convert values to a numeric array. "
            f"Accepted types: list, range, np.ndarray, or a "
            f"('continuous'|'integer', min, max) tuple.\n"
            f"Original error: {exc}"
        )

    if arr.ndim != 1:
        _fatal(
            f"Parameter '{name}': values must be 1-D, got shape {arr.shape}."
        )

    if len(arr) == 0:
        _fatal(f"Parameter '{name}': values must not be empty.")

    # Deduplicate and sort
    arr_unique = np.unique(arr)
    if len(arr_unique) < len(arr):
        _warn(
            f"Parameter '{name}': {len(arr) - len(arr_unique)} duplicate "
            f"value(s) removed."
        )
    arr = arr_unique

    if len(arr) == 1:
        _warn(
            f"Parameter '{name}' has only 1 grid level. "
            f"It will not contribute to space-filling."
        )

    return arr.astype(float)


# ======================================================================
# ParameterSpace
# ======================================================================

class ParameterSpace:
    """
    N-dimensional discrete parameter space for space-filling design.

    Each parameter is internally represented as a sorted 1-D float array
    (the grid).  Continuous and integer parameters are automatically
    discretised to a grid; SA operates on this grid via coordinate swap.

    Parameters
    ----------
    parameters : dict, optional
        Mapping of ``{name: spec}`` where *spec* is any supported
        parameter specification (see module docstring).

    Examples
    --------
    >>> import numpy as np
    >>> from mergen.space import ParameterSpace
    >>> space = ParameterSpace({
    ...     'voltage':  [100, 200, 300, 400],
    ...     'pressure': ('continuous', 0.5, 5.0),
    ...     'lr':       ('continuous', 1e-4, 1e-1, 'log'),
    ... })
    >>> space.n_parameters
    3
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, object]] = None,
        resolution: int = _DEFAULT_RESOLUTION,
    ) -> None:
        self._parameters:     Dict[str, np.ndarray] = {}
        self._param_types:    Dict[str, str]         = {}  # 'discrete'|'continuous'|'integer'
        self._constraints:    List[Callable]          = []
        self._candidate_pool: Optional[np.ndarray]   = None   # lazy cache
        self._resolution      = resolution

        if parameters is not None:
            if not isinstance(parameters, dict):
                _fatal(
                    f"ParameterSpace expects a dict of {{name: spec}}, "
                    f"got {type(parameters).__name__}."
                )
            for name, spec in parameters.items():
                self.add_parameter(name, spec)

    # ------------------------------------------------------------------ #
    # Public: building the space                                           #
    # ------------------------------------------------------------------ #

    def add_parameter(
        self,
        name: str,
        spec,
        resolution: Optional[int] = None,
    ) -> "ParameterSpace":
        """
        Add a parameter axis to the space.

        Parameters
        ----------
        name : str
            Parameter name — used as column header in output DataFrames.
            May contain spaces, Greek letters, LaTeX, or any Unicode.
        spec : array-like or tuple
            Supported formats::

                [100, 200, 300]                      # discrete explicit
                range(10, 110, 10)                   # discrete range
                np.arange(0.1, 1.1, 0.1)            # discrete numpy
                ('continuous', 0.5, 5.0)             # linear grid
                ('continuous', 1e-4, 1e-1, 'log')   # log grid
                ('integer', 2, 10)                   # integer grid
                ('integer', 8, 256, 'log')           # log-integer grid
                ('continuous', 0.5, 5.0, {'resolution': 500})  # custom res

        resolution : int, optional
            Grid size for continuous / integer-log parameters.
            Overrides the space-level resolution for this parameter only.
            Default: space-level resolution (set at construction).

        Returns
        -------
        self — enables fluent chaining.
        """
        if not isinstance(name, str) or not name.strip():
            _fatal("Parameter name must be a non-empty string.")

        if name in self._parameters:
            _fatal(
                f"Parameter '{name}' already exists. "
                f"Use a different name or create a new ParameterSpace."
            )

        res = resolution if resolution is not None else self._resolution
        arr = _parse_parameter(name, spec, resolution=res)

        # Determine type label
        if isinstance(spec, tuple) and isinstance(spec[0], str):
            kind = spec[0].lower().strip()
            self._param_types[name] = kind   # 'continuous' or 'integer'
        else:
            self._param_types[name] = 'discrete'

        self._parameters[name]  = arr
        self._candidate_pool    = None   # invalidate cache
        return self

    def add_constraint(self, fn: Callable) -> "ParameterSpace":
        """
        Add a feasibility constraint.

        The callable receives a single dict mapping parameter names to
        their current values and must return ``True`` for feasible points.

        Parameters
        ----------
        fn : callable
            Signature: ``fn(p: dict) -> bool``

            Examples::

                lambda p: p['x'] + p['y'] <= 10
                lambda p: p['pressure'] * p['temperature'] < 1000
                lambda p: p['lr'] < 1e-2 or p['n_layers'] <= 3

        Returns
        -------
        self
        """
        if not callable(fn):
            _fatal("Constraint must be callable: lambda p: p['x'] + p['y'] <= 10.")
        self._constraints.append(fn)
        self._candidate_pool = None
        return self

    def set_resolution(self, resolution: int) -> "ParameterSpace":
        """
        Update the default grid resolution for continuous / integer-log
        parameters and rebuild affected grids.

        Note: parameters added with an explicit ``resolution`` override
        are not affected.

        Parameters
        ----------
        resolution : int
            New default resolution (>= 2).
        """
        if resolution < 2:
            _fatal(f"Resolution must be >= 2, got {resolution}.")
        self._resolution = resolution
        self._candidate_pool = None
        return self

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def names(self) -> List[str]:
        """Ordered list of parameter names."""
        return list(self._parameters.keys())

    @property
    def n_parameters(self) -> int:
        """Number of parameter axes."""
        return len(self._parameters)

    @property
    def values(self) -> List[np.ndarray]:
        """List of grid arrays, one per parameter (in insertion order)."""
        return list(self._parameters.values())

    @property
    def n_levels(self) -> List[int]:
        """Number of grid levels for each parameter."""
        return [len(v) for v in self._parameters.values()]

    @property
    def bounds(self) -> List[tuple]:
        """(min, max) for each parameter."""
        return [(float(v.min()), float(v.max()))
                for v in self._parameters.values()]

    @property
    def param_types(self) -> Dict[str, str]:
        """Dict mapping parameter name → type ('discrete'|'continuous'|'integer')."""
        return dict(self._param_types)

    @property
    def n_candidates(self) -> int:
        """Number of feasible candidate points (after constraints)."""
        return len(self.candidate_pool)

    @property
    def candidate_pool(self) -> np.ndarray:
        """
        Full Cartesian product of all parameter grids, filtered by
        constraints.  Shape: (n_candidates, n_parameters).

        The result is cached; invalidated automatically when parameters
        or constraints are added.
        """
        if self._candidate_pool is None:
            self._candidate_pool = self._build_pool()
        return self._candidate_pool

    @property
    def gmins(self) -> np.ndarray:
        """Per-dimension minimum of the feasible candidate pool."""
        return self.candidate_pool.min(axis=0)

    @property
    def granges(self) -> np.ndarray:
        """
        Per-dimension range of the feasible candidate pool.
        Never zero (degenerate dimensions get range = 1e-9).
        """
        r = self.candidate_pool.max(axis=0) - self.candidate_pool.min(axis=0)
        r[r == 0] = 1e-9
        return r

    # ------------------------------------------------------------------ #
    # Normalisation                                                        #
    # ------------------------------------------------------------------ #

    def normalise(self, points: np.ndarray) -> np.ndarray:
        """
        Map *points* to [0, 1]^d using the feasible pool's min/range.

        Parameters
        ----------
        points : array-like, shape (n,) or (n, d)

        Returns
        -------
        np.ndarray, same shape as input
        """
        pts = np.asarray(points, dtype=float)
        return (pts - self.gmins) / self.granges

    def denormalise(self, points: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`normalise`."""
        pts = np.asarray(points, dtype=float)
        return pts * self.granges + self.gmins

    # ------------------------------------------------------------------ #
    # Distance                                                             #
    # ------------------------------------------------------------------ #

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Normalised Euclidean distance between two points in [0, 1]^d.

        Both points are normalised before distance computation so that
        all parameter axes contribute equally regardless of scale.

        Parameters
        ----------
        x, y : array-like, shape (d,)

        Returns
        -------
        float in [0, sqrt(d)]
        """
        xn = self.normalise(np.asarray(x, dtype=float))
        yn = self.normalise(np.asarray(y, dtype=float))
        return float(np.sqrt(np.sum((xn - yn) ** 2)))

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def is_valid(self) -> bool:
        """Return True if the space has at least one parameter and one candidate."""
        return self.n_parameters > 0 and self.n_candidates > 0

    def on_grid(self, point: Sequence[float]) -> int:
        """
        Return the row index of *point* in the candidate pool, or -1.

        Parameters
        ----------
        point : array-like, shape (n_parameters,)
        """
        pt = np.asarray(point, dtype=float).ravel()
        if len(pt) != self.n_parameters:
            _fatal(
                f"Point has {len(pt)} coordinate(s) but space has "
                f"{self.n_parameters} parameter(s)."
            )
        pool    = self.candidate_pool
        matches = np.where(np.all(np.isclose(pool, pt, rtol=1e-9, atol=1e-9),
                                  axis=1))[0]
        return int(matches[0]) if len(matches) > 0 else -1

    def validate_point(
        self,
        point: Sequence[float],
        label: str = "Point",
    ) -> np.ndarray:
        """
        Assert that *point* lies on the grid.

        Parameters
        ----------
        point : array-like, shape (n_parameters,)
        label : str, prefix for the error message

        Returns
        -------
        pt : np.ndarray, shape (n_parameters,)

        Raises
        ------
        ValueError if the point is not on the grid.
        """
        pt = np.asarray(point, dtype=float).ravel()
        if self.on_grid(pt) < 0:
            lo = [f"{v.min():.4g}" for v in self._parameters.values()]
            hi = [f"{v.max():.4g}" for v in self._parameters.values()]
            _fatal(
                f"{label} {pt.tolist()} is not on the parameter grid.\n"
                f"  Parameter bounds: "
                + ", ".join(
                    f"{n}=[{a}, {b}]"
                    for n, a, b in zip(self.names, lo, hi)
                )
                + "\n  Note: the point must match a grid node exactly."
            )
        return pt

    # ------------------------------------------------------------------ #
    # GridSampler factory                                                  #
    # ------------------------------------------------------------------ #

    def grid_sampler(self) -> "GridSampler":
        """Return a :class:`GridSampler` for this space."""
        return GridSampler(self)

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        lines = []
        for name, vals in self._parameters.items():
            kind = self._param_types.get(name, 'discrete')
            lines.append(
                f"  {name!r:30s} {kind:12s}  "
                f"{len(vals):5d} levels  "
                f"[{vals.min():.4g}, {vals.max():.4g}]"
            )
        nc    = self.n_candidates
        total = int(np.prod(self.n_levels)) if self.n_parameters else 0
        feas  = f"{nc}/{total}" if self._constraints else str(nc)
        header = f"ParameterSpace  ({self.n_parameters} parameters,  {feas} candidates)"
        sep    = "─" * max(len(header), 60)
        return "\n".join([sep, header, sep] + lines + [sep])

    def __len__(self) -> int:
        return self.n_candidates

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _build_pool(self) -> np.ndarray:
        """Build the Cartesian product and apply constraints."""
        if self.n_parameters == 0:
            return np.empty((0, 0), dtype=float)

        pool = np.array(
            list(itertools.product(*self._parameters.values())),
            dtype=float,
        )

        if self._constraints:
            names = self.names
            mask  = np.ones(len(pool), dtype=bool)
            for i, row in enumerate(pool):
                p = dict(zip(names, row))
                try:
                    mask[i] = all(c(p) for c in self._constraints)
                except TypeError as exc:
                    _fatal(
                        f"Constraint raised a TypeError for point {row.tolist()}.\n"
                        f"  Constraint must accept a single dict argument:\n"
                        f"    lambda p: p['param'] > value\n"
                        f"  Parameter names: {names}\n"
                        f"  Original error: {exc}"
                    )
            pool = pool[mask]

        if len(pool) == 0:
            _fatal(
                "No feasible candidates remain after applying constraints.\n"
                "  Check your constraint functions — they may be too restrictive."
            )

        n_removed = int(np.prod(self.n_levels)) - len(pool)
        if n_removed > 0 and self._constraints:
            # Informational only — not a warning, just context in __repr__
            pass

        return pool


# ======================================================================
# GridSampler — memory-efficient bijective grid representation
# ======================================================================

class GridSampler:
    """
    Bijective mapping between integer indices and grid points.

    Represents the full Cartesian product grid without materialising it
    in memory.  Uses a mixed-radix number system:

    Mathematical basis
    ------------------
    For a grid with n_1 × n_2 × ... × n_d points, define strides::

        s_k = prod(n_{k+1}, ..., n_d)

    Then for index i::

        level_k = (i // s_k) mod n_k
        x_k     = values_k[level_k]

    This is the standard mixed-radix decomposition — exact and reversible.

    Memory
    ------
    O(d × max_levels) for the value arrays only.
    A 10^10-point grid uses ~48 bytes vs ~800 GB for a full array.

    Parameters
    ----------
    space : ParameterSpace

    References
    ----------
    Morris & Mitchell (1995), J. Statist. Plan. Infer. 43.
    """

    #: Grids with fewer candidates than this are iterated exactly in greedy.
    #: Above the threshold, random sampling is used (unbiased in expectation).
    FULL_GREEDY_THRESHOLD: int = 500_000

    def __init__(self, space: ParameterSpace) -> None:
        self._space       = space
        self.n_dims       = space.n_parameters
        self.values       = space.values      # list[np.ndarray]
        self.n_levels     = space.n_levels    # list[int]
        self.n_candidates = space.n_candidates
        self.gmins        = space.gmins
        self.granges      = space.granges

        # Mixed-radix strides (precomputed for O(d) bijection)
        self._strides: List[int] = []
        stride = 1
        for n in reversed(self.n_levels):
            self._strides.insert(0, stride)
            stride *= n

    # ------------------------------------------------------------------ #
    # Core bijection                                                       #
    # ------------------------------------------------------------------ #

    def index_to_point(self, idx: int) -> np.ndarray:
        """
        O(d) bijection: integer index → grid point.

        Parameters
        ----------
        idx : int in [0, n_candidates)

        Returns
        -------
        np.ndarray, shape (d,)
        """
        return np.array([
            self.values[d][(idx // self._strides[d]) % self.n_levels[d]]
            for d in range(self.n_dims)
        ])

    def point_to_index(self, point: np.ndarray) -> int:
        """
        O(d) bijection: grid point → integer index.

        Parameters
        ----------
        point : array-like, shape (d,)

        Returns
        -------
        int — index in [0, n_candidates), or -1 if off-grid.
        """
        idx = 0
        for d in range(self.n_dims):
            li = np.where(np.isclose(self.values[d], point[d],
                                     rtol=1e-9, atol=1e-9))[0]
            if len(li) == 0:
                return -1
            idx += int(li[0]) * self._strides[d]
        return idx

    # ------------------------------------------------------------------ #
    # Sampling                                                             #
    # ------------------------------------------------------------------ #

    def random_point_excluding(
        self,
        reserved: set,
        max_tries: int = 100_000,
    ):
        """
        Draw a uniformly random feasible grid point not in *reserved*.

        For large grids where ``len(reserved) << n_candidates`` the
        rejection rate is negligible and this terminates in O(1) expected
        tries.

        Parameters
        ----------
        reserved  : set of int — indices to skip
        max_tries : int — upper bound on rejection sampling attempts

        Returns
        -------
        (point, index) : (np.ndarray, int) or (None, None) if exhausted
        """
        constraints = self._space._constraints
        names       = self._space.names

        for _ in range(max_tries):
            idx = random.randint(0, self.n_candidates - 1)
            if idx in reserved:
                continue
            pt = self.index_to_point(idx)
            if constraints:
                p = dict(zip(names, pt))
                if not all(c(p) for c in constraints):
                    continue
            return pt, idx

        return None, None

    def greedy_maximin_seed(
        self,
        selected: np.ndarray,
        budget: int,
        reserved: set,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Greedy farthest-point seeding for SA initialisation.

        Adds *budget* points to *selected* by iteratively choosing the
        candidate that maximises the minimum distance to already-selected
        points (weighted maximin criterion).

        Small grids (≤ :attr:`FULL_GREEDY_THRESHOLD`):
            Exact — iterates every non-reserved index.  O(N × n × d).

        Large grids (> :attr:`FULL_GREEDY_THRESHOLD`):
            Samples ``K = min(N, max(10_000, 50 × n_selected))`` random
            candidates per step.  Unbiased in expectation; every point
            has positive probability of selection.

        Parameters
        ----------
        selected : np.ndarray, shape (n, d) — anchor points
        budget   : int — number of additional points to add
        reserved : set of int — indices already in use (updated in-place)
        weights  : array-like, shape (d,), optional
            Per-dimension importance weights for distance computation.
            None → uniform weights.

        Returns
        -------
        selected : np.ndarray, shape (n + budget, d)
        reserved : set (updated)

        References
        ----------
        Morris & Mitchell (1995), J. Statist. Plan. Infer. 43.
        """
        w = (np.ones(self.n_dims, dtype=float) if weights is None
             else np.asarray(weights, dtype=float))
        w = w / w.sum()

        for _ in range(budget):

            # Edge case: no anchor yet — pick any random point
            if len(selected) == 0:
                pt, idx = self.random_point_excluding(reserved)
                if pt is None:
                    break
                selected = pt[np.newaxis, :]
                reserved.add(idx)
                continue

            norm_sel             = (selected - self.gmins) / self.granges
            best_idx, best_dist  = -1, -1.0

            if self.n_candidates <= self.FULL_GREEDY_THRESHOLD:
                # ── Exact greedy ──────────────────────────────────────────
                for idx in range(self.n_candidates):
                    if idx in reserved:
                        continue
                    pt   = self.index_to_point(idx)
                    pt_n = (pt - self.gmins) / self.granges
                    d    = float(np.min(
                        np.sqrt(np.sum(w * (norm_sel - pt_n) ** 2, axis=1))
                    ))
                    if d > best_dist:
                        best_dist, best_idx = d, idx

            else:
                # ── Large-grid random sampling ────────────────────────────
                K    = min(
                    self.n_candidates - len(reserved),
                    max(10_000, 50 * len(selected)),
                )
                seen: set = set()
                for _ in range(K):
                    idx = random.randint(0, self.n_candidates - 1)
                    if idx in reserved or idx in seen:
                        continue
                    seen.add(idx)
                    pt   = self.index_to_point(idx)
                    pt_n = (pt - self.gmins) / self.granges
                    d    = float(np.min(
                        np.sqrt(np.sum(w * (norm_sel - pt_n) ** 2, axis=1))
                    ))
                    if d > best_dist:
                        best_dist, best_idx = d, idx

            if best_idx < 0:
                break

            pt       = self.index_to_point(best_idx)
            selected = np.vstack([selected, pt])
            reserved.add(best_idx)

        return selected, reserved

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"GridSampler("
            f"n_dims={self.n_dims}, "
            f"n_candidates={self.n_candidates:,})"
        )