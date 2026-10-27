"""
mergen.algorithms.ese
=====================
Enhanced Stochastic Evolutionary (ESE) algorithm for space-filling design
optimisation.

ESE is the de-facto reference algorithm in the DoE community for
optimising Latin Hypercube Samples. Compared with classical Simulated
Annealing it has two distinguishing features:

1. **Best-of-J inner loop.** At inner iteration ``i``, ``J`` distinct
   element exchanges within column ``i mod d`` of the current design
   are generated and the *best* (lowest-scoring) one is the candidate.
   It is accepted when its score change Δ satisfies
   ``Δ <= Th × U(0, 1)`` for a threshold ``Th``.

2. **Adaptive threshold.** After every outer iteration ``Th`` is updated
   from the *acceptance ratio* and the *improvement ratio* of the
   just-completed inner loop. When the best design improved
   (improving process) ``Th`` is lowered while accepted moves outnumber
   improvements, kept when every accepted move improved, and raised
   otherwise. When it did not (exploration process) ``Th`` is raised
   quickly until more than ``accept_high`` of the moves are accepted,
   then lowered slowly until fewer than ``accept_low`` are, and so on.

The rules, constants and defaults follow Jin, Chen & Sudjianto (2005)
as restated by Liu & Wang's EESE paper and Ba, Myers & Brenneman (2015);
the original paper could not be consulted directly.

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
Ba, S., Myers, W. R. & Brenneman, W. A. (2015). Optimal sliced Latin
    hypercube designs. *Technometrics*, 57(4), 479–487.
Dupuy, D., Helbert, C. & Franco, J. (2015). DiceDesign and DiceEval:
    Two R-Packages for Design and Analysis of Computer Experiments.
    *Journal of Statistical Software*, 65(11), 1–38.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer, OptimisationResult, banned_points, hits

if TYPE_CHECKING:
    from mergen.space    import ParameterSpace
    from mergen.criteria import BaseCriterion


_CRIT_EPS = 1e-10
# Candidate exchanges per restart when Q is not given.
_DEFAULT_EXCHANGES = 10_000
_EPS      = 1e-300


class ESEOptimizer(BaseOptimizer):
    """
    Enhanced Stochastic Evolutionary optimiser for space-filling designs.

    Parameters
    ----------
    n_restarts : int, default 1
        Number of ESE runs. Each run after the first starts from a
        perturbation of the running best design. The DiceDesign default
        is 1 because ESE's adaptive temperature already provides robust
        diversification within a single run.
    M : int or None, default None
        Number of inner-loop iterations per outer iteration. ``None``
        uses Jin et al.'s ``min(2 × n_e × d / J, 100)``, where ``n_e`` is
        the number of distinct element exchanges in a column,
        ``n (n - 1) / 2`` for ``n`` movable rows.
    J : int or None, default None
        Number of distinct element exchanges tried at each inner
        iteration; the best one is the candidate. ``None`` uses
        ``min(n_e / 5, 50)``.
    Q : int or None, default None
        Number of outer-loop iterations per restart; the threshold is
        adapted after each one. ``None`` runs enough outer iterations
        for about 10,000 candidate exchanges (``M × J × Q``, the effort
        of DiceDesign's defaults), and never fewer than 2.
    T_start : float or None, default None
        Initial threshold ``Th0`` on the change of the criterion score.
        ``None`` uses ``0.005 ×`` the score of the starting design.
    alpha_cool : float, default 0.8
        Improving process: ``Th ← alpha_cool × Th`` while accepted moves
        outnumber improvements of the best design, ``Th`` unchanged
        when every accepted move improved it, ``Th ← Th / alpha_cool``
        when at most ``accept_low`` of the moves were accepted.
    alpha_warm : float, default 0.7
        Exploration process: ``Th ← Th / alpha_warm`` while raising the
        threshold.
    alpha_explore : float, default 0.9
        Exploration process: ``Th ← alpha_explore × Th`` while lowering
        it.
    accept_low : float, default 0.1
        Acceptance ratio below which exploration starts raising the
        threshold.
    accept_high : float, default 0.8
        Acceptance ratio above which exploration starts lowering it.
    seed_offset : int, default 0
        Added to the user-supplied seed for the internal RNG.

    Notes
    -----
    Because ESE perturbations are pure column-swaps, the per-axis level
    histogram of the *final* design equals that of the *initial* design.
    To make full use of ESE, the initial design should already be a
    balanced LHS — Mergen's Sampler does this automatically via
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
            'M'           : None,
            'J'           : None,
            'Q'           : None,
            'T_start'     : None,
            'alpha_cool'  : 0.8,
            'alpha_warm'  : 0.7,
            'alpha_explore': 0.9,
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
        banned:         Optional[set] = None,
    ) -> OptimisationResult:
        """
        Run the ESE optimisation.

        See :class:`BaseOptimizer.optimize` for parameter semantics.
        """
        t_start  = time.perf_counter()
        eff_seed = int(seed) + int(self.seed_offset)

        # ── Validate ──
        for p_name in ['M', 'J', 'Q', 'n_restarts']:
            if getattr(self, p_name) is not None and getattr(self, p_name) < 1:
                raise ValueError(f"{p_name} must be ≥ 1; "
                                 f"got {getattr(self, p_name)}")
        if not 0.0 < self.alpha_cool < 1.0:
            raise ValueError(f"alpha_cool must be in (0, 1); "
                             f"got {self.alpha_cool}")
        if not 0.0 < self.alpha_warm < 1.0:
            raise ValueError(f"alpha_warm must be in (0, 1); "
                             f"got {self.alpha_warm}")
        if not 0.0 < self.alpha_explore < 1.0:
            raise ValueError(f"alpha_explore must be in (0, 1); "
                             f"got {self.alpha_explore}")

        gs      = space.grid_sampler()
        gmins   = space.gmins
        granges = space.granges

        # Initial reserved set, plus externally banned grid nodes
        # that a swap may never select.
        reserved: set = set(banned or ())
        for row in initial_design:
            idx = gs.point_to_index(row)
            if idx >= 0:
                reserved.add(idx)
        banned_pts = banned_points(gs, banned, initial_design)

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
                    eff_seed, r_idx, banned_pts,
                )

            if verbose and self.n_restarts > 1:
                self._info(f"Restart {r_idx + 1}/{self.n_restarts}")

            (out_design, out_reserved, out_score,
             n_iter, n_acc, n_imp) = self._run_one(
                r_design, r_reserved, gs, gmins, granges,
                space, criterion, n_frozen, crit_start,
                eff_seed, r_idx, verbose, banned_pts,
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
        banned_pts:  Optional[np.ndarray] = None,
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

        # Jin, Chen & Sudjianto (2005): threshold Th0 = 0.005 x the
        # starting score; J and M from the number of distinct element
        # exchanges in a column.
        T  = (float(self.T_start) if self.T_start is not None
              else 0.005 * float(raw_score))
        T0 = T
        n_e = n_swap * (n_swap - 1) // 2
        J = self.J if self.J is not None else max(1, min(n_e // 5, 50))
        M = (self.M if self.M is not None
             else max(1, min(2 * n_e * n_dims // J, 100)))
        Q = (self.Q if self.Q is not None
             else max(2, -(-_DEFAULT_EXCHANGES // (M * J))))
        # n_repeats='auto' lengthens a default search through this factor.
        Q *= int(getattr(self, 'budget_factor', 1))

        if verbose:
            self._info(
                f"Start  log(score)={np.log(max(raw_score, _EPS)):.3f}  "
                f"Th0={T0:.3e}  M={M}  J={J}  Q={Q}  "
                f"swappable={n_swap}"
            )

        constraints = space._constraints if space._constraints else None
        names       = space.names

        n_total_iter     = 0
        n_total_accepted = 0
        n_total_improved = 0
        # Exploration raises Th until enough moves are accepted, then
        # lowers it until too few are; this flag holds the direction.
        raising = False

        # ── Outer loop ──
        for q in range(Q):
            # Jin et al. adapt Th on the counts of the last outer
            # iteration only.
            n_acc_q, n_imp_q = 0, 0
            best_at_start    = best_score
            for m in range(M):
                # ── Inner loop: J distinct exchanges in column m mod d ──
                v = m % n_dims
                best_cand_score = None
                best_cand_swap  = None      # (i_abs, j_abs)
                best_cand_X_new = None
                tried: set = set()
                n_cand = 0

                # J distinct exchanges that change the design; draws
                # that cannot (same row, same level, a repeated pair, an
                # infeasible or duplicate result) do not count.
                while n_cand < J and len(tried) < n_e:
                    i_abs = int(rng.integers(n_frozen, n))
                    j_abs = int(rng.integers(n_frozen, n))
                    if i_abs == j_abs:
                        continue
                    pair = (min(i_abs, j_abs), max(i_abs, j_abs))
                    if pair in tried:
                        continue
                    tried.add(pair)
                    if np.isclose(design[i_abs, v], design[j_abs, v],
                                  rtol=1e-9, atol=0.0):
                        continue
                    # Rows that differ only on axis v swap into each
                    # other, so the design does not change. Scoring such
                    # a swap as two single-row updates passes through a
                    # duplicate point, and the cancellation that follows
                    # reports a large false improvement.
                    other = np.arange(n_dims) != v
                    if np.array_equal(design[i_abs, other],
                                      design[j_abs, other]):
                        continue

                    # Constraint + duplicate check on both resulting rows
                    row_i = design[i_abs].copy()
                    row_j = design[j_abs].copy()
                    row_i[v], row_j[v] = row_j[v], row_i[v]
                    if constraints:
                        p_i = dict(zip(names, row_i))
                        p_j = dict(zip(names, row_j))
                        if not (all(c(p_i) for c in constraints) and
                                all(c(p_j) for c in constraints)):
                            continue
                    # A column swap between rows i and j can make either
                    # new row coincide with a *third* design row, which
                    # distance-free criteria (e.g. CD2) do not punish;
                    # designs must never contain duplicate points.
                    dup_i = np.all(np.isclose(design, row_i,
                                              rtol=1e-9, atol=0.0), axis=1)
                    dup_j = np.all(np.isclose(design, row_j,
                                              rtol=1e-9, atol=0.0), axis=1)
                    dup_i[i_abs] = dup_i[j_abs] = False
                    dup_j[i_abs] = dup_j[j_abs] = False
                    if dup_i.any() or dup_j.any():
                        continue
                    if banned_pts is not None and (
                            hits(banned_pts, row_i)
                            or hits(banned_pts, row_j)):
                        continue

                    # Score change via two incremental updates
                    new_i = (row_i - gmins) / granges
                    new_j = (row_j - gmins) / granges
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

                    n_cand += 1
                    if (best_cand_score is None
                            or score_temp < best_cand_score):
                        best_cand_score = score_temp
                        best_cand_swap  = (i_abs, j_abs)
                        best_cand_X_new = X_temp

                n_total_iter += 1

                # ── Threshold acceptance of the best candidate ──
                if best_cand_swap is not None and (
                        best_cand_score - raw_score
                        <= T * rng.random()):
                    i_abs, j_abs = best_cand_swap
                    design[i_abs, v], design[j_abs, v] = (
                        design[j_abs, v], design[i_abs, v])
                    X_norm    = best_cand_X_new
                    raw_score = best_cand_score
                    n_total_accepted += 1
                    n_acc_q          += 1

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
                            n_total_improved += 1
                            n_imp_q          += 1

            # ── Outer step: adapt Th (Jin, Chen & Sudjianto 2005) ──
            acc_ratio = n_acc_q / M
            imp_ratio = n_imp_q / M

            T_old = T
            if best_score < best_at_start:
                # Improving process.
                if acc_ratio > self.accept_low and imp_ratio < acc_ratio:
                    T = T * self.alpha_cool
                elif acc_ratio > self.accept_low and imp_ratio == acc_ratio:
                    pass
                else:
                    T = T / self.alpha_cool
            else:
                # Exploration process: raise Th quickly until more than
                # accept_high of the moves are accepted, then lower it
                # slowly until fewer than accept_low are.
                if raising and acc_ratio > self.accept_high:
                    raising = False
                elif not raising and acc_ratio < self.accept_low:
                    raising = True
                T = T / self.alpha_warm if raising else T * self.alpha_explore

            if verbose:
                self._info(
                    f"  outer {q+1}/{Q}  Th={T_old:.3e}→{T:.3e}  "
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
        banned_pts: Optional[np.ndarray] = None,
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
                              rtol=1e-9, atol=0.0):
                    continue
                row_i = design[i_abs].copy()
                row_j = design[j_abs].copy()
                row_i[v], row_j[v] = row_j[v], row_i[v]
                if constraints:
                    p_i = dict(zip(names, row_i))
                    p_j = dict(zip(names, row_j))
                    if not (all(c(p_i) for c in constraints) and
                            all(c(p_j) for c in constraints)):
                        continue
                dup_i = np.all(np.isclose(design, row_i,
                                          rtol=1e-9, atol=0.0), axis=1)
                dup_j = np.all(np.isclose(design, row_j,
                                          rtol=1e-9, atol=0.0), axis=1)
                dup_i[i_abs] = dup_i[j_abs] = False
                dup_j[i_abs] = dup_j[j_abs] = False
                if dup_i.any() or dup_j.any():
                    continue
                if banned_pts is not None and (hits(banned_pts, row_i)
                                               or hits(banned_pts, row_j)):
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
