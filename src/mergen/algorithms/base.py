"""
mergen.algorithms.base
======================
Abstract base class for space-filling design optimisation algorithms.

This module defines the contract that every optimisation algorithm in
Mergen must satisfy. New algorithms (SA, SCE, ESE, GA, …) are added by
subclassing :class:`BaseOptimizer` and registering them via
:func:`mergen.algorithms.register_optimizer`.

The design follows scikit-learn's template/strategy pattern
(``BaseEstimator``): the constructor only stores hyperparameters, no
work is done until :meth:`BaseOptimizer.optimize` is called. The result
object follows scipy.optimize's :class:`OptimizeResult` convention but is
implemented as a typed dataclass for static analysis support.

References
----------
Buitinck, L. et al. (2013). API design for machine learning software:
    experiences from the scikit-learn project. *ECML PKDD Workshop:
    Languages for Data Mining and Machine Learning*, 108-122.
Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for
    scientific computing in Python. *Nature Methods*, 17, 261-272.
    (:class:`scipy.optimize.OptimizeResult` convention)
"""

from __future__ import annotations

from abc      import ABC, abstractmethod
from dataclasses import dataclass, field
from typing   import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    from mergen.space    import ParameterSpace
    from mergen.criteria import BaseCriterion


# ── Optimisation result ──────────────────────────────────────────────────
@dataclass
class OptimisationResult:
    """
    Result of a single optimisation run.

    Modelled on :class:`scipy.optimize.OptimizeResult` but implemented as
    a dataclass for type safety and IDE support. Algorithm-specific data
    (acceptance rates, temperatures, restart counts, etc.) are stored
    under :attr:`metadata`.

    Attributes
    ----------
    design : np.ndarray, shape (n, d)
        Final optimised design in the original parameter space.
    score : float
        Final criterion value at the returned design.
        Lower is better (all criteria are minimised).
    converged : bool
        Whether the optimiser believes it converged to a (local) optimum.
        Algorithm-specific definition; see each algorithm's documentation.
    n_iter : int
        Total number of iterations performed (across all restarts if any).
    elapsed : float
        Wall-clock time of the optimisation in seconds.
    metadata : dict
        Algorithm-specific information. Standard keys (when applicable):

        - ``"n_accepted"`` : int — accepted moves (SA, SCE, ESE)
        - ``"n_rejected"`` : int — rejected moves
        - ``"n_restarts"`` : int — number of restarts performed
        - ``"score_history"`` : list[float] — best score per restart
        - ``"T_start"`` : float — initial temperature (SA, ESE)
        - ``"T_end"`` : float — final temperature
        - ``"acceptance_rate"`` : float — fraction of accepted moves
        - ``"algorithm"`` : str — algorithm name (e.g. 'sa', 'sce')
        - ``"params"`` : dict — hyperparameters used

    Notes
    -----
    Use ``result.x`` as an alias for ``result.design`` and ``result.fun``
    for ``result.score`` to maintain partial compatibility with the
    ``scipy.optimize.OptimizeResult`` interface.
    """

    design:    np.ndarray
    score:     float
    converged: bool                      = False
    n_iter:    int                       = 0
    elapsed:   float                     = 0.0
    metadata:  Dict[str, Any]            = field(default_factory=dict)

    # ── SciPy-compatible aliases ─────────────────────────────────────
    @property
    def x(self) -> np.ndarray:
        """Alias for :attr:`design` (scipy.optimize compatibility)."""
        return self.design

    @property
    def fun(self) -> float:
        """Alias for :attr:`score` (scipy.optimize compatibility)."""
        return self.score

    @property
    def nit(self) -> int:
        """Alias for :attr:`n_iter` (scipy.optimize compatibility)."""
        return self.n_iter

    @property
    def success(self) -> bool:
        """Alias for :attr:`converged` (scipy.optimize compatibility)."""
        return self.converged

    def __repr__(self) -> str:
        n, d = self.design.shape
        return (f"OptimisationResult(design=(n={n}, d={d}), "
                f"score={self.score:.6g}, converged={self.converged}, "
                f"n_iter={self.n_iter}, elapsed={self.elapsed:.2f}s)")


# ── Abstract base class for optimisers ────────────────────────────────────
class BaseOptimizer(ABC):
    """
    Abstract base class for space-filling design optimisers.

    Every optimisation algorithm in Mergen (SA, SCE, ESE, …) inherits from
    this class and implements two methods:

    1. :meth:`get_default_params` (classmethod) — return default hyperparams
    2. :meth:`optimize` — run the optimisation

    Hyperparameters are stored as instance attributes; the constructor
    populates them from the class defaults and any keyword arguments
    supplied by the user. Validation of parameter *values* (range checks,
    type checks) happens lazily inside :meth:`optimize`, not in
    :meth:`__init__`, following the scikit-learn convention. This allows
    :func:`sklearn.base.clone`-style cloning and grid search over
    hyperparameters.

    Subclass requirements
    ---------------------
    - Set the class attribute :attr:`name` to a short identifier
      (``'sa'``, ``'sce'``, ``'ese'``, …). This is the key used in the
      registry and by ``Sampler.set_optimizer(name, **kwargs)``.
    - Implement :meth:`get_default_params` to return a ``dict`` of
      hyperparameter names → default values. Every key here becomes a
      valid keyword argument to :meth:`__init__` and :meth:`set_params`.
    - Implement :meth:`optimize` to perform the actual optimisation.

    Examples
    --------
    >>> class DummyOptimizer(BaseOptimizer):
    ...     name = 'dummy'
    ...
    ...     @classmethod
    ...     def get_default_params(cls):
    ...         return {'max_iter': 1000, 'seed_offset': 0}
    ...
    ...     def optimize(self, initial_design, space, criterion,
    ...                  n_frozen=0, crit_start=0, seed=44, verbose=True):
    ...         # ... actual algorithm here ...
    ...         return OptimisationResult(
    ...             design=initial_design,
    ...             score=0.0,
    ...             metadata={'algorithm': self.name},
    ...         )

    See Also
    --------
    mergen.algorithms.register_optimizer : Register a subclass.
    mergen.algorithms.get_optimizer      : Look up a registered algorithm.
    """

    # Subclasses MUST override this with their short identifier.
    name: str = ""

    # ── Initialisation ───────────────────────────────────────────────
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialise the optimiser with default + user-supplied parameters.

        Parameters
        ----------
        **kwargs
            Hyperparameter overrides. Only keys present in
            :meth:`get_default_params` are accepted; unknown keys raise
            :class:`ValueError`. Value validation is deferred to
            :meth:`optimize`.

        Raises
        ------
        ValueError
            If ``kwargs`` contains an unknown hyperparameter.
        """
        defaults = self.get_default_params()

        # Populate from defaults first
        for key, value in defaults.items():
            setattr(self, key, value)

        # Apply user overrides
        for key, value in kwargs.items():
            if key not in defaults:
                raise ValueError(
                    f"{self.__class__.__name__}: unknown parameter "
                    f"{key!r}. Valid parameters: {sorted(defaults)}"
                )
            setattr(self, key, value)

    # ── Class-level: default parameters ──────────────────────────────
    @classmethod
    @abstractmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Return default hyperparameters for this optimiser.

        Subclasses override this to declare their hyperparameter schema.
        The dict keys define the legal keyword arguments to
        :meth:`__init__` and :meth:`set_params`; the values are used when
        the user does not supply an override.

        Returns
        -------
        dict
            Mapping from parameter name to default value.
        """

    # ── Instance-level: parameter management ─────────────────────────
    def set_params(self, **kwargs: Any) -> 'BaseOptimizer':
        """
        Set hyperparameters on this instance (scikit-learn convention).

        Parameters
        ----------
        **kwargs
            Hyperparameter values to update. Unknown keys raise
            :class:`ValueError`.

        Returns
        -------
        self
            Enables fluent chaining: ``opt.set_params(max_iter=1000).optimize(...)``.

        Raises
        ------
        ValueError
            If any key is not a valid hyperparameter for this optimiser.
        """
        defaults = self.get_default_params()
        for key, value in kwargs.items():
            if key not in defaults:
                raise ValueError(
                    f"{self.__class__.__name__}: unknown parameter "
                    f"{key!r}. Valid parameters: {sorted(defaults)}"
                )
            setattr(self, key, value)
        return self

    def get_params(self) -> Dict[str, Any]:
        """
        Return current hyperparameter values (scikit-learn convention).

        Returns
        -------
        dict
            Snapshot of current hyperparameters.
        """
        return {k: getattr(self, k) for k in self.get_default_params()}

    # ── The actual work: subclasses implement this ───────────────────
    @abstractmethod
    def optimize(
        self,
        initial_design: np.ndarray,
        space:          'ParameterSpace',
        criterion:      'BaseCriterion',
        n_frozen:       int  = 0,
        crit_start:     int  = 0,
        seed:           int  = 44,
        verbose:        bool = True,
    ) -> OptimisationResult:
        """
        Run the optimisation.

        Parameters
        ----------
        initial_design : np.ndarray, shape (n, d)
            Starting design in the original parameter space. Built by
            :meth:`prepare_initial_design` (typically a balanced LHS
            plus the prescribed / focus anchor rows).
        space : ParameterSpace
            The parameter space — provides grid, constraints, parameter
            names, and value ranges.
        criterion : BaseCriterion
            The criterion to minimise. Algorithms call
            ``criterion.evaluate``, ``criterion.incremental``,
            ``criterion.begin_1d`` / ``criterion.try_1d`` as appropriate.
        n_frozen : int, optional
            The first ``n_frozen`` rows of ``initial_design`` are
            **fixed** and must not be altered. These are typically
            prescribed points the user requires in the final design.
            Default: 0.
        crit_start : int, optional
            The criterion is evaluated on
            ``initial_design[crit_start:]``. Rows ``[0, crit_start)``
            participate in optimisation moves (they may be swapped) but
            are excluded from the criterion. Used for prescribed-but-
            not-criterion-included points. Default: 0.
        seed : int, optional
            Random seed for reproducibility. Default: 44.
        verbose : bool, optional
            Print progress information. Default: True.

        Returns
        -------
        OptimisationResult
            The optimised design and metadata.

        Notes
        -----
        Subclasses must:

        - Treat the first ``n_frozen`` rows as immutable.
        - Evaluate the criterion only on rows ``[crit_start:]``.
        - Respect any constraints in ``space._constraints``.
        - Populate ``metadata`` with at least the algorithm name and
          hyperparameters used.
        """

    # ── Initial-design factory (overridable per algorithm) ─────────────
    def prepare_initial_design(
        self,
        anchors:    np.ndarray,
        budget:     int,
        space:      'ParameterSpace',
        reserved:   set,
        seed:       int  = 44,
    ) -> 'tuple[np.ndarray, set]':
        """
        Build the starting design for this optimiser.

        The default implementation returns a **balanced Latin Hypercube**
        seed on top of the supplied ``anchors``. This matches the
        literature convention for SA (Morris & Mitchell 1995), SCE
        (Kang 2019), and ESE (Jin et al. 2005), which all expect an
        LHS as input.

        Subclasses may override this method if their algorithm requires
        a different initial design (e.g. a greedy maximin seed, a Sobol
        sequence, or a user-supplied design).

        Parameters
        ----------
        anchors : np.ndarray, shape (k, d)
            Rows that must appear in the output unchanged (prescribed
            and focus points placed by the Sampler).
        budget : int
            Number of *additional* rows to add on top of the anchors.
            The returned design has ``len(anchors) + budget`` rows.
        space : ParameterSpace
            The parameter space.
        reserved : set
            Grid indices already in use; updated in place.
        seed : int, default 44
            RNG seed for the LHS shuffles.

        Returns
        -------
        design   : np.ndarray, shape (len(anchors) + budget, d)
        reserved : set (updated)
        """
        import numpy as np
        gs = space.grid_sampler()
        if budget <= 0:
            return anchors.copy(), reserved.copy()
        rng = np.random.default_rng(int(seed))
        return gs.balanced_lhs_seed(
            selected = anchors,
            budget   = budget,
            reserved = reserved,
            rng      = rng,
        )

    # ── Pretty printing ──────────────────────────────────────────────
    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.get_params().items()
        )
        return f"{self.__class__.__name__}({params})"
