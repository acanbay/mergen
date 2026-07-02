"""
mergen.algorithms
=================
Registry of optimisation algorithms for space-filling design.

This subpackage holds Mergen's optimisation engines (SA, SCE, ESE, …)
and a lightweight registry that lets users discover, look up, and
extend the set of algorithms without modifying Mergen's core.

Public functions
----------------
:func:`get_optimizer`
    Look up an optimiser class by its registered name.
:func:`list_optimizers`
    List the names of all registered optimisers.
:func:`register_optimizer`
    Register a new optimiser (typically called by third-party packages
    or by Mergen's own algorithm modules at import time).

Examples
--------
>>> from mergen.algorithms import list_optimizers, get_optimizer
>>> list_optimizers()
[]   # algorithms are added in subsequent phases (SA, SCE, ESE)
>>> # Once SA is implemented:
>>> # SAOptimizer = get_optimizer('sa')
>>> # sa = SAOptimizer(n_restarts=5, max_iter=10000)

Registering a custom optimiser
------------------------------
>>> from mergen.algorithms import BaseOptimizer, register_optimizer
>>> class MyOptimizer(BaseOptimizer):
...     name = 'my_algo'
...     @classmethod
...     def get_default_params(cls): return {'param': 1.0}
...     def optimize(self, *a, **kw): ...
>>> register_optimizer('my_algo', MyOptimizer)
>>> # Now usable via Sampler.set_optimizer('my_algo', ...)
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import BaseOptimizer, OptimisationResult

__all__ = [
    "BaseOptimizer",
    "OptimisationResult",
    "register_optimizer",
    "get_optimizer",
    "list_optimizers",
]


# ── Internal registry ────────────────────────────────────────────────────
_OPTIMIZER_REGISTRY: Dict[str, Type[BaseOptimizer]] = {}


# ── Public API ────────────────────────────────────────────────────────────
def register_optimizer(name: str, cls: Type[BaseOptimizer]) -> None:
    """
    Register an optimiser class under a short identifier.

    Once registered, the class becomes accessible via :func:`get_optimizer`
    and selectable via ``Sampler.set_optimizer(name, **kwargs)`` and
    ``Sampler.run(algorithm=name, ...)``.

    Parameters
    ----------
    name : str
        Short identifier for the algorithm (e.g. ``'sa'``, ``'sce'``,
        ``'ese'``). Convention: lowercase, underscore-separated.
    cls : type
        A subclass of :class:`BaseOptimizer`.

    Raises
    ------
    TypeError
        If ``cls`` is not a subclass of :class:`BaseOptimizer`.
    ValueError
        If ``name`` is empty.

    Examples
    --------
    >>> from mergen.algorithms import BaseOptimizer, register_optimizer
    >>> class MyOpt(BaseOptimizer):
    ...     name = 'my_algo'
    ...     @classmethod
    ...     def get_default_params(cls): return {}
    ...     def optimize(self, *a, **kw): ...
    >>> register_optimizer('my_algo', MyOpt)

    Notes
    -----
    Re-registering an existing name *overwrites* the previous entry. A
    warning is **not** issued because legitimate use cases include
    swapping implementations during testing.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"Optimizer name must be a non-empty string, "
                         f"got {name!r}.")
    if not isinstance(cls, type) or not issubclass(cls, BaseOptimizer):
        raise TypeError(
            f"Optimizer class must inherit from BaseOptimizer; "
            f"got {cls!r}."
        )
    _OPTIMIZER_REGISTRY[name] = cls


def get_optimizer(name: str) -> Type[BaseOptimizer]:
    """
    Look up a registered optimiser class by name.

    Parameters
    ----------
    name : str
        The identifier under which the optimiser was registered.

    Returns
    -------
    type
        The optimiser class. Instantiate it as
        ``get_optimizer('sa')(n_restarts=5)``.

    Raises
    ------
    KeyError
        If no optimiser is registered under ``name``.

    Examples
    --------
    >>> SAOptimizer = get_optimizer('sa')   # after SA is implemented
    >>> sa = SAOptimizer(n_restarts=5)
    """
    if name not in _OPTIMIZER_REGISTRY:
        available = list_optimizers()
        if available:
            raise KeyError(
                f"Unknown optimiser {name!r}. "
                f"Available optimisers: {available}."
            )
        raise KeyError(
            f"Unknown optimiser {name!r}. "
            f"No optimisers are registered yet "
            f"(they will be added in later phases: SA, SCE, ESE)."
        )
    return _OPTIMIZER_REGISTRY[name]


def list_optimizers() -> List[str]:
    """
    Return the sorted list of registered optimiser names.

    Returns
    -------
    list[str]
        Sorted list of identifiers (e.g. ``['ese', 'sa', 'sce']``).

    Examples
    --------
    >>> list_optimizers()
    []   # empty until SA / SCE / ESE are added
    """
    return sorted(_OPTIMIZER_REGISTRY.keys())


# ── Auto-register built-in optimisers ────────────────────────────────────
# Algorithms register themselves as their modules are imported here.

from .sa  import SAOptimizer   # noqa: E402
from .sce import SCEOptimizer  # noqa: E402
from .ese import ESEOptimizer  # noqa: E402
register_optimizer('sa',  SAOptimizer)
register_optimizer('sce', SCEOptimizer)
register_optimizer('ese', ESEOptimizer)
