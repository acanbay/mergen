"""
mergen.algorithms.ese
=====================
Enhanced Stochastic Evolutionary (ESE) algorithm for space-filling design
optimisation.

ESE is the de-facto reference algorithm in the DoE community for
optimising Latin Hypercube Samples. Compared with classical Simulated
Annealing it has two distinguishing features:

1. **Best-of-J inner loop.** At every inner iteration, ``J`` candidate
   elementary perturbations of the current design are generated and the
   *best* (lowest-scoring) is the one passed to the Metropolis test.
   This is a greedy filter that focuses the stochastic acceptance on
   high-quality moves and dramatically reduces variance across seeds.

2. **Adaptive temperature.** After every outer iteration the temperature
   is updated according to the *acceptance ratio* and the
   *improvement ratio* over the just-completed inner loop. Unlike SA
   the temperature can *increase* (re-heat) when the algorithm is
   stuck — this is the "warming" phase of Jin et al. (2005) and is
   especially effective on constrained problems where classical SA
   gets trapped in column-swap dead-ends.

By construction ESE uses *column-swap* perturbations only, so every
accepted move preserves the per-axis level frequencies of the input
design. This makes ESE the optimiser of choice when LHS structure must
be guaranteed in the output (e.g. when ``initial_design`` is a balanced
LHS produced by :meth:`mergen.space.GridSampler.balanced_lhs_seed`).

References
----------
Jin, R., Chen, W. & Sudjianto, A. (2005). An efficient algorithm for
    constructing optimal design of computer experiments.
    *Journal of Statistical Planning and Inference*, 134(1), 268–287.
Damblin, G., Couplet, M. & Iooss, B. (2013). Numerical studies of
    space filling designs: optimization of Latin Hypercube Samples
    and subprojection properties.
    *Journal of Simulation*, 7, 276–289.
Dupuy, D., Helbert, C. & Franco, J. (2015). DiceDesign and DiceEval:
    Two R-Packages for Design and Analysis of Computer Experiments.
    *Journal of Statistical Software*, 65(11), 1–38.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from .base import BaseOptimizer, OptimisationResult

if TYPE_CHECKING:
    from mergen.space    import ParameterSpace
    from mergen.criteria import BaseCriterion


_CRIT_EPS = 1e-10
_EPS      = 1e-300


class ESEOptimizer(BaseOptimizer):
    """
    Enhanced Stochastic Evolutionary optimiser for space-filling designs.

    Parameters
    ----------
    n_restarts : int, default 1
        Number of independent ESE runs. Each restart starts from a
        perturbation of the running best design. The DiceDesign default
        is 1 because ESE's adaptive temperature already provides robust
        diversification within a single run.
    M : int, default 100
        Number of inner-loop iterations per outer iteration.
        ("inner_it" in Jin et al. 2005 and DiceDesign.)
    J : int, default 50
        Number of candidate column-swap perturbations generated at each
        inner iteration; the best (lowest-scoring) candidate is the one
        Metropolis-tested. Higher ``J`` ⇒ greedier search.
    Q : int, default 1
        Number of outer-loop iterations per restart. Total elementary
        perturbations evaluated ≈ ``M × J × Q``.
        ("it" in DiceDesign.)
    T_start : float or None, default None
        Initial temperature. ``None`` activates the Jin–Sudjianto
        default: ``0.005 × initial criterion value``.
    alpha_cool : float, default 0.8
        Geometric cooling factor applied when the algorithm enters the
        "improvement / exploitation" branch of the adaptive schedule.
        ``T ← alpha_cool × T``.
    alpha_warm : float, default 0.8
        Geometric warming factor applied when the algorithm enters the
        "no-improvement / exploration" branch. ``T ← T / alpha_warm``.
        (Equivalent to ``T × (1 / alpha_warm)`` ≈ ``T × 1.25``.)
    accept_low : float, default 0.1
        Lower acceptance-ratio threshold separating the exploitation
        and exploration branches.
    accept_high : float, default 0.8
        Upper acceptance-ratio threshold. Above this the algorithm is
        considered to be over-accepting and the cooling rate is
        increased.
    seed_offset : int, default 0
        Added to the user-supplied seed for the internal RNG.

    Notes
    -----
    Because ESE perturbations are pure column-swaps, the per-axis level
    histogram of the *final* design equals that of the *initial* design.
    To make full use of ESE, the initial design should already be a
    balanced LHS — Mergen's Sampler does this automatically via
    :meth:`GridSampler.greedy_maximin_seed` or
    :meth:`GridSampler.balanced_lhs_seed`.

    Examples
    --------
    >>> from mergen.algorithms.ese import ESEOptimizer
    >>> ese = ESEOptimizer(M=100, J=50, Q=2)
    >>> result = ese.optimize(initial_design, space, criterion)
    """

    name: str = "ese"

    # ────────────────────────────────────────────────────────────────
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'n_restarts'  : 1,
            'M'           : 100,
            'J'           : 50,
            'Q'           : 1,
            'T_start'     : None,
            'alpha_cool'  : 0.8,
            'alpha_warm'  : 0.8,
            'accept_low'  : 0.1,
            'accept_high' : 0.8,
            'seed_offset' : 0,
        }

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
        Run the ESE optimisation.

        See :class:`BaseOptimizer.optimize` for parameter semantics.
        """
        t_start  = time.perf_counter()
        eff_seed = int(seed) + int(self.seed_offset)

        # ── Validate ──
        for p_name in ['M', 'J', 'Q', 'n_restarts']:
            if getattr(self, p_name) < 1:
                raise ValueError(f"{p_name} must be ≥ 1; "
                                 f"got {getattr(self, p_name)}")
        if not 0.0 < self.alpha_cool < 1.0:
            raise ValueError(f"alpha_cool must be in (0, 1); "
                             f"got {self.alpha_cool}")
        if not 0.0 < self.alpha_warm < 1.0:
            raise ValueError(f"alpha_warm must be in (0, 1); "
                             f"got {self.alpha_warm}")

        gs      = space.grid_sampler()
        gmins   = space.gmins
        granges = space.granges

        # Initial reserved set
        reserved: set = set()
        for row in initial_design:
            idx = gs.point_to_index(row)
            if idx >= 0:
                reserved.add(idx)

        # ── Multi-restart loop ──
        best_design:   np.ndarray = initial_design.copy()
        best_reserved: set        = reserved.copy()
        best_score:    float      = float('inf')
        restart_scores: List[float] = []
        total_iter:     int = 0
        total_accepted: int = 0
        total_improved: int = 0

        for r_idx in range(self.n_restarts):
            if r_idx == 0 or best_score == float('inf'):
                r_design, r_reserved = initial_design.copy(), reserved.copy()
            else:
                r_design, r_reserved = self._perturb_restart(
                    best_design, best_reserved, gs, n_frozen, space,
                    eff_seed, r_idx,
                )

            if verbose and self.n_restarts > 1:
                self._info(f"Restart {r_idx + 1}/{self.n_restarts}")

            (out_design, out_reserved, out_score,
             n_iter, n_acc, n_imp) = self._run_one(
                r_design, r_reserved, gs, gmins, granges,
                space, criterion, n_frozen, crit_start,
                eff_seed, r_idx, verbose,
            )

            restart_scores.append(out_score)
            total_iter     += n_iter
            total_accepted += n_acc
            total_improved += n_imp

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
                'n_improved'       : total_improved,
                'acceptance_rate'  : (total_accepted / total_iter
                                      if total_iter else 0.0),
                'improvement_rate' : (total_improved / total_iter
                                      if total_iter else 0.0),
                'best_log_score'   : float(np.log(max(best_score, _EPS))),
            },
        )

    # ────────────────────────────────────────────────────────────────
    # Private: one ESE restart                                          #
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
    ) -> Tuple[np.ndarray, set, float, int, int, int]:
        """One ESE restart with outer/inner loop and adaptive temperature."""
        design   = design.copy()
        reserved = reserved.copy()
        n        = len(design)
        n_swap   = n - n_frozen
        n_dims   = design.shape[1]

        if n_swap < 2:
            X_norm = (design[crit_start:] - gmins) / granges
            score  = criterion.evaluate(X_norm, space)
            return design, reserved, float(score), 0, 0, 0

        rng = np.random.default_rng(int(seed) + r_idx)

        # ── Initial state ──
        X_norm    = (design[crit_start:] - gmins) / granges
        raw_score = criterion.evaluate(X_norm, space)
        best_score   = raw_score
        best_design  = design.copy()
        best_reserv  = reserved.copy()

        # T_start = 0.005 × initial score (Jin–Sudjianto default)
        T = (float(self.T_start) if self.T_start is not None
             else 0.005 * max(raw_score, _EPS))
        T0 = T

        if verbose:
            self._info(
                f"Start  log(score)={np.log(max(raw_score, _EPS)):.3f}  "
                f"T0={T0:.3e}  M={self.M}  J={self.J}  Q={self.Q}  "
                f"swappable={n_swap}"
            )

        constraints = space._constraints if space._constraints else None
        names       = space.names

        n_total_iter     = 0
        n_total_accepted = 0
        n_total_improved = 0

        # ── Outer loop ──
        for q in range(self.Q):
            for m in range(self.M):
                # ── Inner loop: produce J column-swap candidates and pick best ──
                best_cand_score   = None
                best_cand_log_del = None
                best_cand_swap    = None      # (i_abs, j_abs, axis)
                best_cand_X_new   = None

                for _ in range(self.J):
                    # Sample a column-swap (i, j, v)
                    i_abs = int(rng.integers(n_frozen, n))
                    j_abs = int(rng.integers(n_frozen, n))
                    if i_abs == j_abs:
                        continue
                    v = int(rng.integers(0, n_dims))
                    if np.isclose(design[i_abs, v], design[j_abs, v],
                                  rtol=1e-9, atol=1e-9):
                        continue

                    # Constraint check on both resulting rows
                    if constraints:
                        row_i = design[i_abs].copy()
                        row_j = design[j_abs].copy()
                        row_i[v], row_j[v] = row_j[v], row_i[v]
                        p_i = dict(zip(names, row_i))
                        p_j = dict(zip(names, row_j))
                        if not (all(c(p_i) for c in constraints) and
                                all(c(p_j) for c in constraints)):
                            continue

                    # Δscore via two incremental updates
                    new_i = (
                        np.array([design[j_abs, v] if k == v
                                  else design[i_abs, k]
                                  for k in range(n_dims)]) - gmins
                    ) / granges
                    new_j = (
                        np.array([design[i_abs, v] if k == v
                                  else design[j_abs, k]
                                  for k in range(n_dims)]) - gmins
                    ) / granges

                    i_rel = i_abs - crit_start
                    j_rel = j_abs - crit_start
                    i_in  = (0 <= i_rel < len(X_norm))
                    j_in  = (0 <= j_rel < len(X_norm))
                    if not (i_in or j_in):
                        continue

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

                    # Greedy: keep the *best* among J
                    if (best_cand_score is None
                            or score_temp < best_cand_score):
                        log_delta = (np.log(max(score_temp, _EPS))
                                     - np.log(max(raw_score, _EPS)))
                        best_cand_score   = score_temp
                        best_cand_log_del = log_delta
                        best_cand_swap    = (i_abs, j_abs, v)
                        best_cand_X_new   = X_temp

                n_total_iter += 1

                # ── Metropolis acceptance on the best candidate ──
                if best_cand_swap is not None:
                    log_delta = best_cand_log_del
                    if (log_delta < 0
                            or rng.random() < np.exp(
                                float(np.clip(-log_delta / max(T, _EPS),
                                              -700, 0)))):
                        i_abs, j_abs, v = best_cand_swap
                        design[i_abs, v], design[j_abs, v] = (
                            design[j_abs, v], design[i_abs, v])
                        X_norm    = best_cand_X_new
                        raw_score = best_cand_score
                        n_total_accepted += 1
                        if log_delta < 0:
                            n_total_improved += 1

                        # Track best with drift safeguard
                        if (raw_score < best_score
                                and raw_score > _CRIT_EPS):
                            true_score = criterion.evaluate(X_norm, space)
                            raw_score  = true_score
                            if (true_score < best_score
                                    and true_score > _CRIT_EPS):
                                best_score   = true_score
                                best_design  = design.copy()
                                best_reserv  = reserved.copy()

            # ── Outer step: adapt T (Jin et al. 2005) ──
            # We adapt based on the acceptance and improvement counts
            # *over the last outer iteration*. To keep state minimal we
            # use the running counts since the last adaptation.
            acc_ratio  = (n_total_accepted / max(1, n_total_iter))
            imp_ratio  = (n_total_improved / max(1, n_total_iter))

            T_old = T
            if imp_ratio > 0.10:
                # Lots of improvement → we are in exploitation.
                # Cool slowly to keep exploiting.
                T = T * self.alpha_cool
            elif acc_ratio > self.accept_high:
                # Many acceptances but no improvement → T too high;
                # cool aggressively to focus.
                T = T * self.alpha_cool * self.alpha_cool
            elif acc_ratio < self.accept_low:
                # Almost no acceptances → stuck; re-heat.
                T = T / self.alpha_warm
            else:
                # Mild progress → modest cooling.
                T = T * self.alpha_cool

            if verbose:
                self._info(
                    f"  outer {q+1}.{1}/{self.Q}  T={T_old:.3e}→{T:.3e}  "
                    f"acc_ratio={acc_ratio:.2f}  "
                    f"imp_ratio={imp_ratio:.2f}  "
                    f"best log(score)={np.log(max(best_score, _EPS)):.3f}"
                )

        if verbose:
            self._info(
                f"Done   log(score)={np.log(max(best_score, _EPS)):.3f}  "
                f"(accepted={n_total_accepted}/{n_total_iter}, "
                f"improved={n_total_improved})"
            )

        return (best_design, best_reserv, float(best_score),
                n_total_iter, n_total_accepted, n_total_improved)

    # ────────────────────────────────────────────────────────────────
    # Private: ILS-style perturbation between restarts                  #
    # ────────────────────────────────────────────────────────────────
    def _perturb_restart(
        self,
        design:    np.ndarray,
        reserved:  set,
        gs,
        n_frozen:  int,
        space,
        seed:      int,
        r_idx:     int,
    ) -> Tuple[np.ndarray, set]:
        """
        Restart kick: perform a few column-swaps so the new restart
        starts from a perturbed copy of the running best (preserves LHS
        structure, unlike a random replacement).
        """
        design   = design.copy()
        reserved = reserved.copy()
        n        = len(design)
        n_swap   = n - n_frozen
        if n_swap < 2:
            return design, reserved

        rng         = np.random.default_rng(seed + r_idx + 500)
        n_dims      = design.shape[1]
        n_kick      = max(2, n_swap // 4)
        if n_kick % 2 == 1:
            n_kick += 1
        constraints = space._constraints if space._constraints else None
        names       = space.names

        for _ in range(n_kick // 2):
            for _attempt in range(20):
                i_abs = int(rng.integers(n_frozen, n))
                j_abs = int(rng.integers(n_frozen, n))
                if i_abs == j_abs:
                    continue
                v = int(rng.integers(0, n_dims))
                if np.isclose(design[i_abs, v], design[j_abs, v],
                              rtol=1e-9, atol=1e-9):
                    continue
                if constraints:
                    row_i = design[i_abs].copy()
                    row_j = design[j_abs].copy()
                    row_i[v], row_j[v] = row_j[v], row_i[v]
                    p_i = dict(zip(names, row_i))
                    p_j = dict(zip(names, row_j))
                    if not (all(c(p_i) for c in constraints) and
                            all(c(p_j) for c in constraints)):
                        continue
                design[i_abs, v], design[j_abs, v] = (
                    design[j_abs, v], design[i_abs, v])
                break

        return design, reserved

    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _info(msg: str) -> None:
        # ANSI green tag, matches the rest of Mergen's log style
        print(f"  \033[1;32m[ESE]\033[0m      {msg}", flush=True)
