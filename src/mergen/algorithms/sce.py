"""
mergen.algorithms.sce
=====================
Stochastic Coordinate Exchange (SCE) for space-filling design optimisation.

SCE is a *greedy* coordinate-descent style optimiser: each row of the
design is visited in deterministic round-robin order, and for every
parameter axis a small set of candidate replacement values is scored
using the criterion's :meth:`begin_1d` / :meth:`try_1d` interface. The
best feasible 1D move (strictly improving) is applied; if none improve,
the algorithm moves on. After a complete sweep with no improvement, an
Iterated Local Search kick perturbs a fraction of the rows and the
sweep restarts. Termination occurs when several consecutive kicks fail
to yield any further improvement.

Compared with SA, SCE has *low seed variance* because the acceptance
rule is greedy (no Metropolis randomness); the seed only affects the
small set of candidates probed at each axis visit. This makes SCE a
strong choice when reproducibility across seeds is important and the
problem does not require a stochastic escape from local optima.

References
----------
Kang, L. (2019). Stochastic coordinate-exchange optimal designs with
    complex constraints. *Quality Engineering*, 31(3), 401–416.
You, Y. et al. (2021). Modified coordinate-exchange algorithm for
    space-filling Latin hypercube designs. *Mathematics*, 9(24), 3314.
Meyer, R. K. & Nachtsheim, C. J. (1995). The coordinate-exchange
    algorithm for constructing exact optimal experimental designs.
    *Technometrics*, 37(1), 60–69.
Lourenço, H. R., Martin, O. C. & Stützle, T. (2003). Iterated local
    search. In *Handbook of Metaheuristics*, Springer, 320–353.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer, OptimisationResult

if TYPE_CHECKING:
    from mergen.space    import ParameterSpace
    from mergen.criteria import BaseCriterion


# Numerical floor — criterion clamps at this when two points coincide
# in projection; we never update the best design with a clamped score.
_CRIT_EPS = 1e-10
_EPS      = 1e-300


class SCEOptimizer(BaseOptimizer):
    """
    Stochastic Coordinate Exchange optimiser for space-filling designs.

    Parameters
    ----------
    n_restarts : int, default 5
        Number of independent restarts. Each restart starts from the
        previous best after an ILS perturbation (Lourenço et al. 2003).
    max_iter : int or None, default None
        Total iteration budget per restart (1 iteration = 1 axis visit
        on 1 row). ``None`` activates ``max(2000, 100 × n)``.
    K_axis : int, default 30
        Number of candidate replacement values probed per axis per
        visit. Small axes are scanned exhaustively; large ones are
        sampled without replacement.
    max_ils_kicks : int, default 4
        Maximum number of consecutive ILS kicks with no improvement
        before the restart terminates.
    perturbation_size : int or None, default None
        Number of rows perturbed per kick. ``None`` → ``max(1, n // 4)``.
    seed_offset : int, default 0
        Added to the user-supplied seed to derive the internal RNG seed.

    Notes
    -----
    SCE relies on the criterion's optional :meth:`begin_1d` /
    :meth:`try_1d` fast-update interface (see :mod:`mergen.criteria`).
    The base implementation in :class:`BaseCriterion` provides a
    correct O(n·d) fallback; criteria with a separable structure may
    override these for O(n) inner loops.

    Examples
    --------
    >>> from mergen.algorithms.sce import SCEOptimizer
    >>> sce = SCEOptimizer(n_restarts=5, K_axis=20)
    >>> result = sce.optimize(initial_design, space, criterion)
    """

    name: str = "sce"

    # ────────────────────────────────────────────────────────────────
    # Defaults                                                          #
    # ────────────────────────────────────────────────────────────────
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'n_restarts'        : 5,
            'max_iter'          : None,
            'K_axis'            : 30,
            'max_ils_kicks'     : 4,
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
        banned:         Optional[set] = None,
    ) -> OptimisationResult:
        """
        Run the SCE optimisation.

        See :class:`BaseOptimizer.optimize` for parameter semantics.
        """
        t_start  = time.perf_counter()
        eff_seed = int(seed) + int(self.seed_offset)

        # ── Run-time validation ─────────────────────────────────
        if self.n_restarts < 1:
            raise ValueError(f"n_restarts must be ≥ 1; got {self.n_restarts}")
        if self.K_axis < 1:
            raise ValueError(f"K_axis must be ≥ 1; got {self.K_axis}")
        if self.max_ils_kicks < 1:
            raise ValueError(
                f"max_ils_kicks must be ≥ 1; got {self.max_ils_kicks}"
            )

        # ── Set-up ───────────────────────────────────────────────
        gs       = space.grid_sampler()
        gmins    = space.gmins
        granges  = space.granges

        # Initial reserved set from the input design, plus externally
        # banned grid nodes that a swap may never select.
        reserved: set = set(banned or ())
        for row in initial_design:
            idx = gs.point_to_index(row)
            if idx >= 0:
                reserved.add(idx)

        # ── Multi-restart loop ─────────────────────────────────────
        best_design:   np.ndarray = initial_design.copy()
        best_reserved: set        = reserved.copy()
        best_score:    float      = float('inf')
        restart_scores: List[float] = []
        total_iter:    int        = 0
        total_accepted: int       = 0

        for r_idx in range(self.n_restarts):
            if r_idx == 0 or best_score == float('inf'):
                r_design, r_reserved = initial_design.copy(), reserved.copy()
            else:
                r_design, r_reserved = self._perturb(
                    best_design, best_reserved, gs, n_frozen,
                    space, eff_seed, r_idx,
                )

            if verbose and self.n_restarts > 1:
                self._info(f"Restart {r_idx + 1}/{self.n_restarts}")

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

        return OptimisationResult(
            design     = best_design,
            score      = best_score,
            converged  = True,
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
    # Private: one SCE restart                                          #
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
        """One SCE restart with round-robin coordinate descent + ILS kicks."""
        design   = design.copy()
        reserved = reserved.copy()
        n_total  = len(design)
        n_swap   = n_total - n_frozen
        n_dims   = design.shape[1]

        if n_swap == 0:
            X_norm = (design[crit_start:] - gmins) / granges
            score  = criterion.evaluate(X_norm, space)
            return design, reserved, float(score), 0, 0

        base_seed = int(seed) + r_idx
        rng       = np.random.default_rng(base_seed)

        # Iteration budget
        max_iter = (self.max_iter
                    if self.max_iter is not None
                    else max(2000, 100 * n_total))

        # Normalised view + initial score
        X_norm     = (design[crit_start:] - gmins) / granges
        raw_score  = criterion.evaluate(X_norm, space)
        best_score = raw_score
        best_design  = design.copy()
        best_reserv  = reserved.copy()

        if verbose:
            self._info(
                f"Start  log(score)={np.log(max(raw_score, _EPS)):.3f}  "
                f"max_iter={max_iter}  swappable={n_swap}  "
                f"K_axis={self.K_axis}"
            )

        constraints = space._constraints if space._constraints else None
        param_names = space.names

        K_axis_default     = int(self.K_axis)
        max_ils_kicks      = int(self.max_ils_kicks)

        n_accepted         = 0
        log_interval       = max(max_iter // 10, 500)
        n_local_opt        = 0
        score_at_last_kick = best_score

        iter_count = 0
        sweep      = 0

        while iter_count < max_iter:
            sweep += 1
            improved_in_sweep = False

            # Round-robin through optimisable rows
            for i in range(n_frozen, n_total):
                if iter_count >= max_iter:
                    break

                i_rel = i - crit_start
                if not (0 <= i_rel < len(X_norm)):
                    continue

                cache = criterion.begin_1d(X_norm, i_rel, raw_score)

                # Each axis once
                for axis in range(n_dims):
                    if iter_count >= max_iter:
                        break

                    axis_levels    = gs.values[axis]
                    n_levels       = len(axis_levels)

                    # Build candidate levels for this axis (excluding the
                    # current value)
                    K_axis = min(K_axis_default, max(1, n_levels - 1))
                    if n_levels - 1 <= K_axis_default:
                        # Small axis — scan all other levels
                        cur_raw = design[i, axis]
                        cand_levels = [v for v in axis_levels
                                       if abs(v - cur_raw) > 1e-12]
                    else:
                        # Large axis — random sample
                        cand_idx = rng.choice(n_levels, size=K_axis,
                                              replace=False)
                        cand_levels = [axis_levels[j] for j in cand_idx
                                       if abs(axis_levels[j]
                                              - design[i, axis]) > 1e-12]

                    # Greedy 1D search
                    best_value_raw   = None
                    best_value_norm  = None
                    best_value_score = raw_score
                    best_log_delta   = 0.0

                    cur_point_raw = design[i].copy()
                    cur_idx       = gs.point_to_index(design[i])

                    for v_raw in cand_levels:
                        v_norm = (v_raw - gmins[axis]) / granges[axis]

                        # Feasibility: constraints + uniqueness
                        if constraints is not None:
                            trial = cur_point_raw.copy()
                            trial[axis] = v_raw
                            p = dict(zip(param_names, trial))
                            try:
                                if not all(c(p) for c in constraints):
                                    continue
                            except (TypeError, KeyError):
                                continue

                        trial = cur_point_raw.copy()
                        trial[axis] = v_raw
                        trial_idx   = gs.point_to_index(trial)
                        if (trial_idx >= 0
                                and trial_idx != cur_idx
                                and trial_idx in reserved):
                            continue

                        # Score this candidate via fast 1D criterion update
                        log_delta, new_score, _ = criterion.try_1d(
                            cache, axis, v_norm, space,
                        )

                        if log_delta < best_log_delta:
                            best_log_delta   = log_delta
                            best_value_raw   = v_raw
                            best_value_norm  = v_norm
                            best_value_score = new_score

                    # Accept the best (strictly improving) move
                    if (best_value_raw is not None
                            and best_log_delta < -1e-12):
                        old_idx = gs.point_to_index(design[i])
                        new_pt_raw         = design[i].copy()
                        new_pt_raw[axis]   = best_value_raw
                        new_idx = gs.point_to_index(new_pt_raw)

                        if old_idx >= 0:
                            reserved.discard(old_idx)
                        if new_idx >= 0:
                            reserved.add(new_idx)

                        design[i]            = new_pt_raw
                        X_norm[i_rel, axis]  = best_value_norm
                        raw_score            = best_value_score
                        n_accepted          += 1
                        improved_in_sweep    = True

                        # Refresh cache for subsequent axes on this row
                        cache = criterion.begin_1d(X_norm, i_rel, raw_score)

                        # Verify and track best — with full evaluate to
                        # guard against incremental drift
                        if raw_score < best_score and raw_score > _CRIT_EPS:
                            true_score = criterion.evaluate(X_norm, space)
                            raw_score  = true_score
                            if true_score < best_score and true_score > _CRIT_EPS:
                                best_score        = true_score
                                best_design       = design.copy()
                                best_reserv       = reserved.copy()
                            cache = criterion.begin_1d(X_norm, i_rel, raw_score)

                    iter_count += 1

                    if verbose and iter_count % log_interval == 0:
                        print(
                            f"    iter {iter_count:>{len(str(max_iter))}}"
                            f"/{max_iter}  "
                            f"best log(score)="
                            f"{np.log(max(best_score, _EPS)):.3f}  "
                            f"accepted={n_accepted}",
                            flush=True,
                        )

            # End of sweep — handle local optima with ILS kick
            if not improved_in_sweep:
                n_local_opt += 1
                if (n_local_opt >= max_ils_kicks
                        or (n_local_opt > 1
                            and best_score >= score_at_last_kick - _EPS)):
                    if verbose:
                        self._info(
                            f"Local optimum reached after sweep {sweep} "
                            f"(iter {iter_count}/{max_iter}, "
                            f"kicks={n_local_opt - 1})."
                        )
                    break

                # ── ILS kick on current best ─────────────────────
                if self.perturbation_size is not None:
                    n_kick = max(1, int(self.perturbation_size))
                else:
                    n_kick = max(1, n_swap // 4)
                n_kick = min(n_kick, n_swap)

                kick_rows = rng.choice(
                    np.arange(n_frozen, n_total),
                    size=n_kick, replace=False,
                )

                # Draw replacement points from the local numpy Generator
                # so that reproducibility does not depend on the global
                # Python random state (which is not synchronised between
                # a joblib parent and its worker processes).
                for ki in kick_rows:
                    new_raw, new_idx = gs.random_point_excluding(
                        reserved, rng=rng,
                    )
                    if new_raw is None:
                        continue
                    if constraints is not None:
                        p = dict(zip(param_names, new_raw))
                        try:
                            if not all(c(p) for c in constraints):
                                continue
                        except (TypeError, KeyError):
                            continue
                    old_idx = gs.point_to_index(design[ki])
                    if old_idx >= 0:
                        reserved.discard(old_idx)
                    reserved.add(new_idx)
                    design[ki] = new_raw
                    X_norm[ki - crit_start] = (
                        (new_raw - gmins) / granges
                    )

                raw_score = criterion.evaluate(X_norm, space)
                score_at_last_kick = best_score
                if verbose:
                    self._info(
                        f"Kick {n_local_opt}: perturbed {len(kick_rows)} "
                        f"rows, log(score)="
                        f"{np.log(max(raw_score, _EPS)):.3f}"
                    )

        if verbose:
            self._info(
                f"Done   log(score)={np.log(max(best_score, _EPS)):.3f}  "
                f"(accepted={n_accepted}/{iter_count}, sweeps={sweep})"
            )

        return best_design, best_reserv, float(best_score), iter_count, n_accepted

    # ────────────────────────────────────────────────────────────────
    # Private: ILS perturbation between restarts                        #
    # ────────────────────────────────────────────────────────────────
    def _perturb(
        self,
        design:    np.ndarray,
        reserved:  set,
        gs,
        n_frozen:  int,
        space,
        seed:      int,
        r_idx:     int,
    ) -> Tuple[np.ndarray, set]:
        """ILS kick — replace a fraction of rows with random feasible points."""
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

        rng         = np.random.default_rng(seed + r_idx + 500)
        kick_rows   = rng.choice(np.arange(n_frozen, n), size=k, replace=False)
        constraints = space._constraints if space._constraints else None
        names       = space.names

        # Belt-and-braces against a stale ``reserved``: also exclude the
        # current design rows' indices so a kick replacement can never
        # coincide with an existing design point.
        occupied = set(reserved)
        for row in design:
            ridx = gs.point_to_index(row)
            if ridx >= 0:
                occupied.add(ridx)

        for i in kick_rows:
            for _ in range(20):
                new_pt, new_idx = gs.random_point_excluding(occupied)
                if new_pt is None:
                    break
                if constraints:
                    p = dict(zip(names, new_pt))
                    if not all(c(p) for c in constraints):
                        continue
                old_idx = gs.point_to_index(design[i])
                if old_idx >= 0:
                    reserved.discard(old_idx)
                    occupied.discard(old_idx)
                reserved.add(new_idx)
                occupied.add(new_idx)
                design[i] = new_pt
                break

        return design, reserved

    # ────────────────────────────────────────────────────────────────
    # Internal logger                                                   #
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _info(msg: str) -> None:
        # ANSI green tag, matches the Sampler's log style
        print(f"  \033[1;32m[SCE]\033[0m      {msg}", flush=True)
