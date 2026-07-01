"""
mergen.algorithms.sa
====================
Simulated Annealing for space-filling design optimisation.

The implementation follows the classical Morris & Mitchell (1995)
recipe of single-point coordinate exchange on a discrete grid, extended
with three modern refinements:

* **Auto-tuned initial temperature** via Ben-Ameur (2004): a probe pass
  empirically estimates the criterion's log-delta magnitudes so the
  user does not need to set ``T_start`` manually.
* **Hybrid moves**: each iteration picks one of two move types — a
  *random replacement* (the original Morris–Mitchell move, fast escape
  from local optima) or a *column swap* (preserves the per-axis level
  frequency and is essential for Latin-Hypercube structure). The mix is
  controlled by the ``hybrid_ratio`` hyperparameter.
* **Iterated Local Search restarts** (Lourenço, Martin & Stützle 2003):
  after each restart the best design so far is perturbed and reused as
  the starting point for the next restart, providing diversification
  without losing earlier gains.

References
----------
Morris, M. D. & Mitchell, T. J. (1995). Exploratory designs for
    computational experiments. *J. Statist. Plan. Infer.*, 43, 381–402.
Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P. (1983). Optimization by
    simulated annealing. *Science*, 220, 671–680.
Ben-Ameur, W. (2004). Computing the initial temperature of simulated
    annealing. *Computational Optimization and Applications*,
    29(3), 369–385.
Lourenço, H. R., Martin, O. C. & Stützle, T. (2003). Iterated local
    search. In *Handbook of Metaheuristics*, Springer, 320–353.
Joseph, V. R., Gul, E. & Ba, S. (2015). Maximum projection designs for
    computer experiments. *Biometrika*, 102(2), 371–380.
Vorechovsky, M. & Elias, J. (2026). uMaxPro: A coordinate-exchange
    algorithm for uniform space-filling designs on discrete grids.
    *Computers & Structures*.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from .base import BaseOptimizer, OptimisationResult

if TYPE_CHECKING:
    from mergen.space    import ParameterSpace
    from mergen.criteria import BaseCriterion


# Numerical floor below which we refuse to update best_design.
# criteria.py clamps scores at this value when two points coincide in
# projection — those are not valid optimisation states.
_CRIT_EPS = 1e-10
_EPS      = 1e-300


class SAOptimizer(BaseOptimizer):
    """
    Hybrid Simulated Annealing optimiser for space-filling designs.

    Parameters
    ----------
    n_restarts : int, default 5
        Number of independent restarts; the best design across restarts
        is returned. Restarts ≥ 2 use ILS-style perturbation kicks from
        the current best (Lourenço et al. 2003).
    max_iter : int or None, default None
        Maximum Metropolis iterations per restart. ``None`` triggers the
        auto rule ``max(2000, 100 × n)`` where *n* is the design size.
    T_start : float or None, default None
        Initial temperature. ``None`` activates the Ben-Ameur (2004)
        auto-tune, which probes random transitions to estimate a
        temperature giving acceptance rate ``target_accept``.
    cooling : float or None, default None
        Geometric cooling rate (``T_{k+1} = cooling × T_k``). ``None``
        sets it so that the temperature reaches ``T_end_ratio × T_start``
        at the last iteration.
    T_end_ratio : float, default 1e-4
        Final-to-initial temperature ratio, used only when ``cooling``
        is auto-computed.
    target_accept : float, default 0.80
        Target initial acceptance rate for Ben-Ameur auto-tune.
    hybrid_ratio : float, default 0.5
        Probability of choosing a *column-swap* move at each iteration
        (preserves LHS structure). The complement is the probability of
        a *random replacement* move (faster local-optimum escape but
        breaks LHS frequency balance). ``hybrid_ratio = 0.0`` → pure
        Morris–Mitchell random replacement; ``1.0`` → pure column-swap.
    perturbation_size : int or None, default None
        Number of rows to perturb between restarts (ILS kick size).
        ``None`` activates ``max(1, n_optimised // 4)``.
    seed_offset : int, default 0
        Added to the user-supplied seed to derive the SA-internal RNG
        seed. Useful when one wants two independent SA runs from the
        same Sampler.

    Notes
    -----
    The optimiser **respects the first ``n_frozen`` rows** of
    ``initial_design`` (they are never altered) and evaluates the
    criterion on rows ``[crit_start:]`` only. Constraints declared on
    ``space`` are honoured for every candidate move; infeasible moves
    are rejected before the criterion is evaluated.

    Examples
    --------
    >>> from mergen.algorithms.sa import SAOptimizer
    >>> sa = SAOptimizer(n_restarts=10, hybrid_ratio=0.7)
    >>> result = sa.optimize(initial_design, space, criterion)
    >>> result.score
    """

    name: str = "sa"

    # ────────────────────────────────────────────────────────────────
    # Defaults                                                          #
    # ────────────────────────────────────────────────────────────────
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'n_restarts'        : 5,
            'max_iter'          : None,
            'T_start'           : None,
            'cooling'           : None,
            'T_end_ratio'       : 1e-4,
            'target_accept'     : 0.80,
            'hybrid_ratio'      : 0.5,
            'perturbation_size' : None,
            'seed_offset'       : 0,
        }

    # ────────────────────────────────────────────────────────────────
    # Public: optimise                                                  #
    # ────────────────────────────────────────────────────────────────
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
        Run the SA optimisation.

        See :class:`BaseOptimizer.optimize` for parameter semantics.
        """
        t_start = time.perf_counter()

        # Derive a private seed for this SA run
        eff_seed = int(seed) + int(self.seed_offset)

        # ── Validate hyperparameters at run time (sklearn pattern) ──
        if not 0.0 <= self.hybrid_ratio <= 1.0:
            raise ValueError(
                f"hybrid_ratio must be in [0, 1]; got {self.hybrid_ratio}"
            )
        if self.n_restarts < 1:
            raise ValueError(
                f"n_restarts must be ≥ 1; got {self.n_restarts}"
            )

        # ── Set up grid + helpers ─────────────────────────────────
        gs            = space.grid_sampler()
        gmins         = space.gmins
        granges       = space.granges
        constraints   = space._constraints
        names         = space.names
        n_dims        = space.n_parameters

        # ── Initial reserved set from frozen + optimised rows ───────
        reserved = set()
        for row in initial_design:
            idx = gs.point_to_index(row)
            if idx >= 0:
                reserved.add(idx)

        # ── Multi-restart loop ─────────────────────────────────────
        best_design:    np.ndarray = initial_design.copy()
        best_reserved:  set        = reserved.copy()
        best_score:     float      = float('inf')
        restart_scores: List[float] = []
        total_iter:     int        = 0
        total_accepted: int        = 0

        for r_idx in range(self.n_restarts):
            # Per-restart starting point
            if r_idx == 0 or best_score == float('inf'):
                r_design, r_reserved = initial_design.copy(), reserved.copy()
            else:
                # ILS kick: perturb the best design so far
                r_design, r_reserved = self._perturb(
                    best_design, best_reserved, gs, n_frozen, constraints,
                    names, eff_seed, r_idx,
                )

            if verbose and self.n_restarts > 1:
                self._info(f"Restart {r_idx + 1}/{self.n_restarts}")

            # Run one restart
            out_design, out_reserved, out_score, n_iter, n_acc = self._run_one(
                r_design, r_reserved, gs, gmins, granges,
                space, criterion, n_frozen, crit_start,
                eff_seed, r_idx, verbose,
            )

            restart_scores.append(out_score)
            total_iter     += n_iter
            total_accepted += n_acc

            if out_score < best_score:
                best_score    = out_score
                best_design   = out_design
                best_reserved = out_reserved
                if verbose and self.n_restarts > 1:
                    self._info(
                        f"Restart {r_idx + 1}: new best  "
                        f"log(score)={np.log(max(out_score, _EPS)):.3f}"
                    )

        elapsed = time.perf_counter() - t_start

        # ── Build the result object ─────────────────────────────────
        return OptimisationResult(
            design     = best_design,
            score      = best_score,
            converged  = True,   # SA converges deterministically per schedule
            n_iter     = total_iter,
            elapsed    = elapsed,
            metadata   = {
                'algorithm'        : self.name,
                'params'           : self.get_params(),
                'n_restarts'       : self.n_restarts,
                'restart_scores'   : [float(s) for s in restart_scores],
                'best_restart'     : int(np.argmin(restart_scores)),
                'n_accepted'       : total_accepted,
                'acceptance_rate'  : (total_accepted / total_iter
                                      if total_iter else 0.0),
                'best_log_score'   : float(np.log(max(best_score, _EPS))),
            },
        )

    # ────────────────────────────────────────────────────────────────
    # Private: one SA restart                                           #
    # ────────────────────────────────────────────────────────────────
    def _run_one(
        self,
        design:      np.ndarray,
        reserved:    set,
        gs,
        gmins:       np.ndarray,
        granges:     np.ndarray,
        space,
        criterion:   'BaseCriterion',
        n_frozen:    int,
        crit_start:  int,
        seed:        int,
        r_idx:       int,
        verbose:     bool,
    ) -> Tuple[np.ndarray, set, float, int, int]:
        """
        Run a single SA restart with hybrid moves.

        Returns
        -------
        best_design : np.ndarray
        reserved    : set
        best_score  : float
        n_iter      : int
        n_accepted  : int
        """
        design   = design.copy()
        reserved = reserved.copy()
        n        = len(design)
        n_swap   = n - n_frozen

        if n_swap == 0:
            return design, reserved, float('inf'), 0, 0

        # Normalised view for criterion
        X_norm    = (design[crit_start:] - gmins) / granges
        raw_score = criterion.evaluate(X_norm, space)

        # Per-restart deterministic RNG (independent of global random)
        rng = np.random.default_rng(seed + r_idx)

        # ── Auto-tune temperature if not user-supplied ──────────────
        if verbose:
            self._info("Tuning temperature...")
        T_start, max_iter = self._auto_temp(
            design, gs, gmins, granges, crit_start, criterion, space,
            seed, r_idx,
        )

        if self.T_start is not None:
            T_start  = float(self.T_start)
        if self.max_iter is not None:
            max_iter = int(self.max_iter)

        if self.cooling is not None:
            cooling = float(self.cooling)
        else:
            T_end   = float(self.T_end_ratio) * T_start
            cooling = (T_end / T_start) ** (1.0 / max(1, max_iter))

        best_score  = raw_score
        best_design = design.copy()
        T           = T_start

        if verbose:
            self._info(
                f"Start  log(score)={np.log(max(raw_score, _EPS)):.3f}  "
                f"T={T_start:.4e}  iters={max_iter}  "
                f"swappable={n_swap}  hybrid={self.hybrid_ratio:.2f}"
            )

        log_interval = max(max_iter // 10, 500)
        n_accepted   = 0
        constraints  = space._constraints
        names        = space.names

        # ── Main Metropolis loop ─────────────────────────────────
        for it in range(max_iter):
            # Choose move type — column-swap or random replacement
            use_swap = (n_swap >= 2 and rng.random() < self.hybrid_ratio)

            if use_swap:
                accepted, raw_score, X_norm = self._try_swap(
                    design, X_norm, n_frozen, crit_start, raw_score,
                    space, criterion, constraints, names, T, rng,
                )
            else:
                accepted, raw_score, X_norm, reserved = self._try_replace(
                    design, X_norm, reserved, n_frozen, crit_start,
                    raw_score, space, gs, criterion, gmins, granges,
                    constraints, names, T, rng,
                )

            if accepted:
                n_accepted += 1
                if raw_score < best_score and raw_score > _CRIT_EPS:
                    best_score  = raw_score
                    best_design = design.copy()

            T *= cooling

            if verbose and (it + 1) % log_interval == 0:
                print(f"    iter {it+1:>{len(str(max_iter))}}/{max_iter}  "
                      f"T={T:.3e}  "
                      f"best log(score)={np.log(max(best_score, _EPS)):.3f}",
                      flush=True)

        if verbose:
            self._info(
                f"Done   log(score)={np.log(max(best_score, _EPS)):.3f}  "
                f"(accepted={n_accepted}/{max_iter}, "
                f"rate={n_accepted/max_iter:.1%})"
            )

        return best_design, reserved, best_score, max_iter, n_accepted

    # ────────────────────────────────────────────────────────────────
    # Private: one column-swap move                                     #
    # ────────────────────────────────────────────────────────────────
    def _try_swap(
        self,
        design,      X_norm,
        n_frozen,    crit_start,
        raw_score,
        space,       criterion,
        constraints, names,
        T,           rng,
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Attempt a column-swap between two non-frozen rows on a random axis.

        Returns
        -------
        accepted   : bool
        raw_score  : float (updated if accepted, else unchanged)
        X_norm     : np.ndarray (updated if accepted)
        """
        n      = len(design)
        n_dims = design.shape[1]
        # Pick two distinct rows (both must be ≥ n_frozen) and one axis
        i_abs = int(rng.integers(n_frozen, n))
        j_abs = int(rng.integers(n_frozen, n))
        if i_abs == j_abs:
            return False, raw_score, X_norm

        v = int(rng.integers(0, n_dims))
        if np.isclose(design[i_abs, v], design[j_abs, v],
                      rtol=1e-9, atol=1e-9):
            return False, raw_score, X_norm

        # Constraint check on BOTH resulting rows
        if constraints:
            row_i = design[i_abs].copy()
            row_j = design[j_abs].copy()
            row_i[v], row_j[v] = row_j[v], row_i[v]
            p_i = dict(zip(names, row_i))
            p_j = dict(zip(names, row_j))
            if not (all(c(p_i) for c in constraints) and
                    all(c(p_j) for c in constraints)):
                return False, raw_score, X_norm

        # Compute Δscore via two incremental updates (i then j on updated X)
        gmins, granges = space.gmins, space.granges
        n_dims_ = design.shape[1]
        new_i = (np.array([design[j_abs, v] if k == v else design[i_abs, k]
                           for k in range(n_dims_)]) - gmins) / granges
        new_j = (np.array([design[i_abs, v] if k == v else design[j_abs, k]
                           for k in range(n_dims_)]) - gmins) / granges

        i_rel = i_abs - crit_start
        j_rel = j_abs - crit_start
        i_in  = (0 <= i_rel < len(X_norm))
        j_in  = (0 <= j_rel < len(X_norm))
        if not (i_in or j_in):
            return False, raw_score, X_norm

        X_temp     = X_norm.copy()
        score_temp = raw_score
        if i_in:
            _, score_temp = criterion.incremental(
                X_temp, i_rel, new_i, space, score_temp)
            X_temp[i_rel] = new_i
        if j_in:
            _, score_temp = criterion.incremental(
                X_temp, j_rel, new_j, space, score_temp)
            X_temp[j_rel] = new_j

        log_delta = (np.log(max(score_temp, _EPS))
                     - np.log(max(raw_score, _EPS)))

        # Metropolis acceptance
        if (log_delta < 0
                or rng.random() < np.exp(
                    float(np.clip(-log_delta / T, -700, 0)))):
            # Apply swap to design
            design[i_abs, v], design[j_abs, v] = (
                design[j_abs, v], design[i_abs, v])
            return True, score_temp, X_temp

        return False, raw_score, X_norm

    # ────────────────────────────────────────────────────────────────
    # Private: one random-replacement move                              #
    # ────────────────────────────────────────────────────────────────
    def _try_replace(
        self,
        design,      X_norm,        reserved,
        n_frozen,    crit_start,
        raw_score,
        space,       gs,            criterion,
        gmins,       granges,
        constraints, names,
        T,           rng,
    ) -> Tuple[bool, float, np.ndarray, set]:
        """
        Attempt a Morris–Mitchell random replacement of a single row.

        Returns
        -------
        accepted  : bool
        raw_score : float
        X_norm    : np.ndarray
        reserved  : set
        """
        n = len(design)

        i = int(rng.integers(n_frozen, n))
        i_rel = i - crit_start
        if not (0 <= i_rel < len(X_norm)):
            return False, raw_score, X_norm, reserved

        # Draw a fresh grid point that is currently unused
        new_raw, new_idx = gs.random_point_excluding(reserved, rng=rng)
        if new_raw is None:
            return False, raw_score, X_norm, reserved

        # Constraint check
        if constraints:
            p_new = dict(zip(names, new_raw))
            if not all(c(p_new) for c in constraints):
                return False, raw_score, X_norm, reserved

        new_pt = (new_raw - gmins) / granges
        log_delta, new_score = criterion.incremental(
            X_norm, i_rel, new_pt, space, raw_score,
        )

        # Metropolis acceptance
        if (log_delta < 0
                or rng.random() < np.exp(
                    float(np.clip(-log_delta / T, -700, 0)))):
            old_idx = gs.point_to_index(design[i])
            reserved.discard(old_idx)
            reserved.add(new_idx)
            design[i]     = new_raw
            X_norm[i_rel] = new_pt
            return True, new_score, X_norm, reserved

        return False, raw_score, X_norm, reserved

    # ────────────────────────────────────────────────────────────────
    # Private: Ben-Ameur (2004) automatic T_start                       #
    # ────────────────────────────────────────────────────────────────
    def _auto_temp(
        self,
        design,
        gs,
        gmins, granges,
        crit_start,
        criterion,
        space,
        seed,
        r_idx,
        n_probe: int = 200,
    ) -> Tuple[float, int]:
        """
        Ben-Ameur (2004) automatic initial temperature.

        Estimates ``T_start`` such that the probability of accepting a
        positive-cost transition equals ``target_accept`` on average.
        See reference list in module docstring.

        Returns
        -------
        T_start  : float
        max_iter : int  (auto rule: ``max(2000, 100 × n)``)
        """
        n       = len(design)
        n_crit  = n - crit_start
        rng     = np.random.default_rng(seed + r_idx + 1000)

        # Use a random design for the probe (Damblin–Couplet–Iooss 2013):
        # a near-optimal design produces too-small Δ's and underestimates T.
        probe_pts = []
        probe_res: set = set()
        while len(probe_pts) < min(n_crit, gs.n_candidates):
            pt, idx = gs.random_point_excluding(probe_res, rng=rng)
            if pt is None:
                break
            probe_pts.append(pt)
            probe_res.add(idx)
        if not probe_pts:
            return 1.0, max(2000, 100 * n)

        probe_arr   = np.array(probe_pts)
        probe_norm  = (probe_arr - gmins) / granges
        probe_score = criterion.evaluate(probe_norm, space)

        # Collect positive transitions
        E_min, E_max, all_deltas = [], [], []
        for _ in range(n_probe):
            i_rel   = int(rng.integers(0, n_crit))
            new_idx = int(rng.integers(0, gs.n_candidates))
            if new_idx in probe_res:
                continue
            new_raw = gs.index_to_point(new_idx)
            new_pt  = (new_raw - gmins) / granges
            ld, new_score = criterion.incremental(
                probe_norm, i_rel, new_pt, space, probe_score)
            all_deltas.append(abs(ld))
            if ld > 0:
                E_min.append(probe_score)
                E_max.append(new_score)

        target = float(self.target_accept)

        if not E_min:
            # No positive transitions found — fall back on |Δ| average
            T_start = (np.mean(all_deltas) / -np.log(target)
                       if all_deltas else 1.0)
        else:
            E_min_arr = np.array(E_min)
            E_max_arr = np.array(E_max)
            # Johnson (1989) initial estimate
            deltas = np.log(E_max_arr) - np.log(np.maximum(E_min_arr, 1e-300))
            T_start = -np.mean(deltas) / np.log(target)

            # Ben-Ameur (2004) iterative refinement, Eq. (6) with p=1
            eps = 1e-3
            p   = 1
            for _ in range(50):
                lo = -E_max_arr / T_start
                hi = -E_min_arr / T_start
                lo -= lo.max()
                hi -= hi.max()
                chi_hat = np.sum(np.exp(lo)) / np.sum(np.exp(hi))
                chi_hat = float(np.clip(chi_hat, 1e-10, 1 - 1e-10))
                if abs(chi_hat - target) <= eps:
                    break
                ratio = np.log(chi_hat) / np.log(target)
                if ratio <= 0:
                    T_start *= 2.0
                    continue
                T_new = T_start * (ratio ** (1.0 / p))
                if T_new < 0 or np.isnan(T_new) or np.isinf(T_new):
                    p *= 2
                    continue
                T_start = float(T_new)

        max_iter = max(2000, 100 * n)
        return float(T_start), int(max_iter)

    # ────────────────────────────────────────────────────────────────
    # Private: ILS-style perturbation between restarts                  #
    # ────────────────────────────────────────────────────────────────
    def _perturb(
        self,
        design:      np.ndarray,
        reserved:    set,
        gs,
        n_frozen:    int,
        constraints,
        names,
        seed:        int,
        r_idx:       int,
    ) -> Tuple[np.ndarray, set]:
        """
        Iterated Local Search kick — perturb a fraction of rows.

        Returns a fresh (design, reserved) pair where ``perturbation_size``
        rows have been replaced with random feasible grid points.
        """
        design   = design.copy()
        reserved = reserved.copy()
        n        = len(design)
        n_swap   = n - n_frozen
        if n_swap <= 0:
            return design, reserved

        if self.perturbation_size is None:
            k = max(1, n_swap // 4)
        else:
            k = max(1, min(int(self.perturbation_size), n_swap))

        rng = np.random.default_rng(seed + r_idx + 500)
        kick_rows = rng.choice(np.arange(n_frozen, n), size=k, replace=False)

        for i in kick_rows:
            for _ in range(20):  # up to 20 attempts to find a feasible replacement
                new_pt, new_idx = gs.random_point_excluding(reserved, rng=rng)
                if new_pt is None:
                    break
                if constraints:
                    p_new = dict(zip(names, new_pt))
                    if not all(c(p_new) for c in constraints):
                        continue
                old_idx = gs.point_to_index(design[i])
                reserved.discard(old_idx)
                reserved.add(new_idx)
                design[i] = new_pt
                break

        return design, reserved

    # ────────────────────────────────────────────────────────────────
    # Internal logger                                                   #
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _info(msg: str) -> None:
        # ANSI green tag to match the Sampler's log style
        print(f"  \033[1;32m[SA]\033[0m       {msg}", flush=True)