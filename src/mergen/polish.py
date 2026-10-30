"""
mergen.polish
=============
Point-exchange local search on the priority metrics, used by
:meth:`Sampler.compare` (``polish=True``) to bring the compared designs
closer to the best designs for the user's priority.

Each step moves one movable row of a design to the free feasible
candidate that gives the best key, as long as the key improves; a run
of steps stops after a sweep over all rows without improvement.
Optionally, a fixed number of iterated-local-search kicks follows (a
few rows moved at random, then exchange again); a kicked result is kept
only if it is better. Point exchange is the classical local search for
discrete design problems (Fedorov 1972); the kicks follow Lourenço,
Martin & Stützle (2003).

With ``min_distance`` among the priority metrics, the exchange
alternates with sweeps that lower the Morris-Mitchell phi_p (p = 50) of
the design without making the key worse (Morris & Mitchell 1995). phi_p
counts every small distance, so these moves spread the closest pairs
on a plateau where the exchange alone stalls. They do not reliably
reach the best ``min_distance``: on test spaces the polished winner lay
a median 3 % (up to 12 %) below the best known value under
``'lexicographic'`` and 10 % (up to 38 %) under ``'pareto'``, where the
other priority metric may not get worse and the sweeps changed no
winner in the tests.

The key, for ``selection='lexicographic'``, starts with the amount by
which the first metric is worse than at the start (beyond ``tol``), so
moves that each lose a little within the tolerance cannot add up to a
loss on it; then the priority metrics follow in order. After a metric
whose best value is usually shared by many designs (``min_distance``,
``minimax``), the number of pairs or points that attain that value is
added as a tie-breaker, so the search can move along such a plateau.
For ``selection='pareto'`` the key starts with the total amount by
which the metrics are worse than at the start, so a polished design is
never worse on any priority metric than the design it starts from;
then the first metric (with its tie-breaker) and the others in order.
Two values within the user's ``tol``/``rtol`` count as equal.

Every candidate move of a row is scored at once: the metric changes are
computed from cached sums and distances, and ``minimax`` uses a pruned,
exact search over the evaluation points. The metric values are those of
:mod:`mergen.metrics` (the results are re-evaluated there).

References
----------
Fedorov, V. V. (1972). *Theory of Optimal Experiments*. Academic Press.
Lourenço, H. R., Martin, O. C. & Stützle, T. (2003). Iterated local
    search. In *Handbook of Metaheuristics*, Springer, 320-353.
Morris, M. D. & Mitchell, T. J. (1995). Exploratory designs for
    computational experiments. *Journal of Statistical Planning and
    Inference*, 43(3), 381-402.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .criteria.base import _EPS
from .distances import _embed_heom

# Work limits: counts, not times, so a seed gives the same design on
# every machine.
_MAX_SWEEPS = 50
_N_KICKS    = 0
_KICK_ROWS  = 3
#: Exponent of the Morris-Mitchell phi_p that guides moves along a
#: min_distance plateau, and the candidates tried per row.
_PHI_P      = 50
_PHI_TRIES  = 10
#: Metrics whose best value is typically attained by many designs.
_PLATEAU = ('min_distance', 'minimax')
#: Rows of the minimax evaluation set scored per block.
_BLOCK = 256


def _dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distances between the rows of A and of B."""
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2)


class _Metrics:
    """
    Values and candidate scores of the priority metrics for one design.

    ``rows`` are pool positions of the movable rows; ``fixed`` are the
    normalised fixed rows (frozen design rows and delivered extra rows),
    which every metric includes, as in quality_report().
    """

    def __init__(self, space, pool_n, pool_e, eval_e, fixed_n, fixed_e,
                 priority):
        self.space    = space
        self.pool_n   = pool_n
        self.pool_e   = pool_e
        self.eval_e   = eval_e
        self.fixed_n  = fixed_n
        self.fixed_e  = fixed_e
        self.priority = list(priority)
        is_mask = space.is_mask if space.has_nominal else \
            np.zeros(space.n_parameters, dtype=bool)
        self.corr_cols = np.flatnonzero(~is_mask)
        self.cd_cols   = np.flatnonzero(space.active_mask & ~is_mask)

    # ── state ─────────────────────────────────────────────────────────
    def set_rows(self, rows: np.ndarray) -> None:
        self.rows = np.asarray(rows, dtype=int).copy()
        m = len(self.rows)
        self.Xn = np.vstack([self.pool_n[self.rows], self.fixed_n])
        self.Xe = np.vstack([self.pool_e[self.rows], self.fixed_e])
        self.n  = len(self.Xn)
        self.m  = m
        need = set(self.priority)
        if 'min_distance' in need:
            self.D = _dist(self.Xe, self.Xe)
            np.fill_diagonal(self.D, np.inf)
            self.Dpool = _dist(self.pool_e, self.Xe)
        if 'minimax' in need:
            self.DE = _dist(self.eval_e, self.Xe)

    def move(self, slot: int, cand: int) -> None:
        rows = self.rows.copy()
        rows[slot] = cand
        self.set_rows(rows)

    # ── exact values ──────────────────────────────────────────────────
    def values(self) -> dict:
        out = {}
        for m in self.priority:
            out[m] = getattr(self, '_val_' + m)()
        return out

    def counts(self, vals: dict, tol: dict) -> dict:
        out = {}
        if 'min_distance' in vals:
            v, t = vals['min_distance'], tol['min_distance']
            iu = np.triu_indices(self.n, k=1)
            out['min_distance'] = int((self.D[iu] <= v + t).sum())
        if 'minimax' in vals:
            v, t = vals['minimax'], tol['minimax']
            out['minimax'] = int((self.DE.min(axis=1) >= v - t).sum())
        return out

    def _val_min_distance(self):
        return float(self.D.min()) if self.n > 1 else 0.0

    def _val_minimax(self):
        return float(self.DE.min(axis=1).max())

    def _val_max_abs_correlation(self):
        Z = self.Xn[:, self.corr_cols]
        if len(Z) < 2 or Z.shape[1] < 2:
            return 0.0
        Zc = Z - Z.mean(axis=0)
        nr = np.sqrt((Zc ** 2).sum(axis=0))
        nr[nr < _EPS] = 1.0
        C = (Zc / nr).T @ (Zc / nr)
        np.fill_diagonal(C, 0.0)
        return float(np.abs(C).max())

    def _val_cd2(self):
        W = self.Xn[:, self.cd_cols]
        if W.shape[1] == 0:
            return float('nan')
        return float(np.sqrt(max(_cd2_sq(W), 0.0)))

    def _val_projection_cd2(self):
        W = self.Xn[:, self.cd_cols]
        r = W.shape[1]
        if r < 2:
            return 0.0
        return float(np.mean([_cd2_sq(W[:, [a, b]])
                              for a, b in combinations(range(r), 2)]))

    # ── candidate scores: row `slot` moved to each pool position in C ──
    def cand_values(self, metric: str, slot: int, C: np.ndarray):
        return getattr(self, '_cand_' + metric)(slot, C)

    def _others(self, slot):
        keep = np.ones(self.n, dtype=bool)
        keep[slot] = False
        return keep

    def _cand_min_distance(self, slot, C):
        keep = self._others(slot)
        Dsub = self.D[np.ix_(keep, keep)]
        m_i  = Dsub.min() if Dsub.size else np.inf
        Dc   = self.Dpool[np.ix_(C, keep)]
        cmin = Dc.min(axis=1) if Dc.shape[1] else np.full(len(C), np.inf)
        return np.minimum(m_i, cmin)

    def cand_count_min_distance(self, slot, C, v, t):
        keep = self._others(slot)
        Dsub = self.D[np.ix_(keep, keep)]
        iu   = np.triu_indices(len(Dsub), k=1)
        base = int((Dsub[iu] <= v + t).sum())
        Dc   = self.Dpool[np.ix_(C, keep)]
        return base + (Dc <= v + t).sum(axis=1)

    def _cand_minimax(self, slot, C):
        keep = self._others(slot)
        cap  = self.DE[:, keep].min(axis=1) if keep.any() else \
            np.full(len(self.eval_e), np.inf)
        return _max_min_pruned(cap, self.eval_e, self.pool_e[C])

    def cand_count_minimax(self, slot, C, v, t):
        keep = self._others(slot)
        cap  = self.DE[:, keep].min(axis=1)
        crit = np.flatnonzero(cap >= v - t)
        if not len(crit):
            return np.zeros(len(C), dtype=int)
        d = _dist(self.eval_e[crit], self.pool_e[C])
        return (np.minimum(cap[crit, None], d) >= v - t).sum(axis=0)

    def _cand_max_abs_correlation(self, slot, C):
        Z = self.Xn[:, self.corr_cols]
        q = Z.shape[1]
        if self.n < 2 or q < 2:
            return np.zeros(len(C))
        n  = self.n
        S  = Z.sum(axis=0) - Z[slot]
        Q  = Z.T @ Z - np.outer(Z[slot], Z[slot])
        Y  = self.pool_n[C][:, self.corr_cols]
        S2 = S[None, :] + Y
        Q2 = Q[None, :, :] + Y[:, :, None] * Y[:, None, :]
        # Centred cross products n·cov; norms as in max_abs_correlation.
        M  = Q2 - S2[:, :, None] * S2[:, None, :] / n
        sd = np.sqrt(np.clip(np.einsum('ckk->ck', M), 0.0, None))
        sd[sd < _EPS] = 1.0
        R  = M / (sd[:, :, None] * sd[:, None, :])
        iu = np.triu_indices(q, k=1)
        return np.abs(R[:, iu[0], iu[1]]).max(axis=1)

    def _cand_cd2(self, slot, C):
        W = self.Xn[:, self.cd_cols]
        if W.shape[1] == 0:
            return np.full(len(C), np.nan)
        sq = _cd2_sq(W) + _cd2_sq_delta(W, slot, self.pool_n[C][:, self.cd_cols],
                                       [np.arange(W.shape[1])])[:, 0]
        return np.sqrt(np.clip(sq, 0.0, None))

    def _cand_projection_cd2(self, slot, C):
        W = self.Xn[:, self.cd_cols]
        r = W.shape[1]
        if r < 2:
            return np.zeros(len(C))
        pairs = [np.array(p) for p in combinations(range(r), 2)]
        base  = np.array([_cd2_sq(W[:, p]) for p in pairs])
        delta = _cd2_sq_delta(W, slot, self.pool_n[C][:, self.cd_cols], pairs)
        return (base[None, :] + delta).mean(axis=1)


def _cd2_sq(W: np.ndarray) -> float:
    """Squared centred L2 discrepancy (Hickernell 1998), as criteria.CD2."""
    n, d = W.shape
    a  = np.abs(W - 0.5)
    t1 = (13.0 / 12.0) ** d
    t2 = 2.0 / n * np.prod(1.0 + 0.5 * a - 0.5 * a ** 2, axis=1).sum()
    P  = np.prod(1.0 + 0.5 * a[:, None, :] + 0.5 * a[None, :, :]
                 - 0.5 * np.abs(W[:, None, :] - W[None, :, :]), axis=2)
    return float(t1 - t2 + P.sum() / n ** 2)


def _cd2_sq_delta(W, slot, Y, col_sets) -> np.ndarray:
    """
    Change of CD2² over each column set when row ``slot`` of W becomes
    each row of Y: an array (len(Y), len(col_sets)).
    """
    n   = len(W)
    z   = W[slot]
    keep = np.ones(n, dtype=bool)
    keep[slot] = False
    Wo  = W[keep]
    az, ay, ao = np.abs(z - 0.5), np.abs(Y - 0.5), np.abs(Wo - 0.5)
    # Per-column factors of the single and pair terms.
    s_z = 1.0 + 0.5 * az - 0.5 * az ** 2                       # (d,)
    s_y = 1.0 + 0.5 * ay - 0.5 * ay ** 2                       # (c, d)
    p_z = 1.0 + 0.5 * az[None, :] + 0.5 * ao - 0.5 * np.abs(z[None, :] - Wo)
    p_y = (1.0 + 0.5 * ay[:, None, :] + 0.5 * ao[None, :, :]
           - 0.5 * np.abs(Y[:, None, :] - Wo[None, :, :]))     # (c, n-1, d)
    self_z = 1.0 + az
    self_y = 1.0 + ay
    out = np.empty((len(Y), len(col_sets)))
    for k, cols in enumerate(col_sets):
        single = s_y[:, cols].prod(axis=1) - s_z[cols].prod()
        pair   = (p_y[:, :, cols].prod(axis=2).sum(axis=1)
                  - p_z[:, cols].prod(axis=1).sum())
        own    = self_y[:, cols].prod(axis=1) - self_z[cols].prod()
        out[:, k] = -2.0 / n * single + (2.0 * pair + own) / n ** 2
    return out


def _max_min_pruned(cap: np.ndarray, E: np.ndarray, Y: np.ndarray):
    """
    ``max_p min(cap[p], |E[p] - y|)`` for every row y of Y, exactly.

    Evaluation points are taken in blocks by decreasing cap; a
    candidate is final once its value reaches the largest cap left,
    since no later point can raise it.
    """
    order = np.argsort(-cap, kind='stable')
    capo  = cap[order]
    out   = np.full(len(Y), -np.inf)
    live  = np.arange(len(Y))
    for start in range(0, len(order), _BLOCK):
        blk = order[start:start + _BLOCK]
        d   = _dist(E[blk], Y[live])
        out[live] = np.maximum(
            out[live], np.minimum(cap[blk, None], d).max(axis=0))
        nxt = capo[start + _BLOCK] if start + _BLOCK < len(order) else -np.inf
        live = live[out[live] < nxt]
        if not len(live):
            break
    return out


class _Key:
    """The polishing key: stages compared in order with tolerances."""

    def __init__(self, priority, higher_better, selection, tol, rtol,
                 start_vals):
        self.priority = list(priority)
        self.hb       = dict(zip(priority, higher_better))
        self.selection = selection
        self.tol  = dict(zip(priority, np.asarray(tol, float).tolist()))
        self.rtol = dict(zip(priority, np.asarray(rtol, float).tolist()))
        self.start = dict(start_vals)
        # Metrics never allowed to end worse than at the start.
        self.guarded = (self.priority if selection == 'pareto'
                        else self.priority[:1])

    def t(self, m, ref):
        return self.tol[m] + self.rtol[m] * abs(ref)

    def oriented(self, m, v):
        return -v if self.hb[m] else v

    def stages(self):
        """Stage names, in order: ('viol',), (m,), ('count', m)."""
        st = [('viol',)]
        ms = self.priority
        for k, m in enumerate(ms):
            st.append(('val', m))
            if m in _PLATEAU and (self.selection == 'lexicographic' or k == 0):
                st.append(('count', m))
        return st

    def violation(self, vals: dict) -> np.ndarray:
        tot = 0.0
        for m in self.guarded:
            o  = self.oriented(m, np.asarray(vals[m], float))
            o0 = self.oriented(m, self.start[m])
            tot = tot + np.clip(o - o0 - self.t(m, self.start[m]), 0.0, None)
        return tot


def _better(a: Sequence[float], b: Sequence[float], tols: Sequence[float]):
    """a better than b, stage by stage, with per-stage tolerances."""
    for x, y, t in zip(a, b, tols):
        if x < y - t:
            return True
        if x > y + t:
            return False
    return False


def polish(design: np.ndarray, n_kept: int, extra: Optional[np.ndarray],
           space, priority: Sequence[str], higher_better: Sequence[bool],
           selection: str, tol, rtol, banned: List[Sequence[float]],
           rng: np.random.Generator, max_sweeps: Optional[int] = None,
           n_kicks: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Polish a design on the priority metrics.

    Parameters
    ----------
    design : ndarray (n, d)
        Raw design rows; the first ``n_kept`` (prescribed and focus
        rows) stay fixed, the others are moved.
    extra : ndarray or None
        Delivered rows outside the budget; fixed, and included in every
        metric as in quality_report().
    banned : list of points
        Grid points no row may move to (exclusions, user sets, ...).
    rng : Generator
        Sweep order and kicks.

    Returns
    -------
    design : ndarray (n, d)
        Same layout; only movable rows differ.
    info : dict
        ``improved`` (bool), ``n_moves``, ``n_sweeps``.
    """
    max_sweeps = _MAX_SWEEPS if max_sweeps is None else max_sweeps
    n_kicks    = _N_KICKS if n_kicks is None else n_kicks
    design = np.asarray(design, dtype=float)
    gs     = space.grid_sampler()
    pool   = np.asarray(space.candidate_pool, dtype=float)
    gidx   = gs.pool_indices()
    pos_of = {int(g): k for k, g in enumerate(gidx)}
    gm, gr = space.gmins, space.granges
    pool_n = (pool - gm) / gr

    levels = None
    if space.has_nominal:
        mask   = space.is_mask
        levels = [np.unique(pool_n[:, j]) if mask[j] else None
                  for j in range(space.n_parameters)]
    emb = (lambda Z: _embed_heom(Z, levels)) if levels else (lambda Z: Z)
    pool_e = emb(pool_n)
    from .metrics import _minimax_eval_set
    eval_e = emb(_minimax_eval_set(space)) if 'minimax' in priority \
        else np.empty((0, pool_e.shape[1]))

    movable = design[n_kept:]
    rows: list = []
    for r in movable:
        p = pos_of.get(int(gs.point_to_index(r)), -1)
        if p < 0:                       # not a feasible pool node
            return design.copy(), dict(improved=False, n_moves=0, n_sweeps=0)
        rows.append(p)
    rows = np.array(rows, dtype=int)
    start_rows = rows.copy()
    if not len(rows):
        return design.copy(), dict(improved=False, n_moves=0, n_sweeps=0)
    fixed = design[:n_kept]
    if extra is not None and len(extra):
        fixed = np.vstack([fixed, np.asarray(extra, dtype=float)])
    fixed_n = (fixed - gm) / gr
    fixed_e = emb(fixed_n) if len(fixed_n) else np.empty((0, pool_e.shape[1]))

    blocked = np.zeros(len(pool), dtype=bool)
    for p in list(banned) + list(fixed):
        k = pos_of.get(int(gs.point_to_index(np.asarray(p, float))), -1)
        if k >= 0:
            blocked[k] = True

    met = _Metrics(space, pool_n, pool_e, eval_e, fixed_n, fixed_e, priority)
    met.set_rows(rows)
    start = met.values()
    if not all(np.isfinite(v) for v in start.values()):
        return design.copy(), dict(improved=False, n_moves=0, n_sweeps=0)
    key = _Key(priority, higher_better, selection, tol, rtol, start)
    stages = key.stages()

    def state_key():
        vals = met.values()
        cnts = met.counts(vals, {m: key.t(m, vals[m]) for m in vals})
        out, tols = [], []
        for st in stages:
            if st[0] == 'viol':
                out.append(float(key.violation(vals)))
                tols.append(0.0)
            elif st[0] == 'val':
                m = st[1]
                out.append(key.oriented(m, vals[m]))
                tols.append(key.t(m, vals[m]))
            else:
                out.append(float(cnts[st[1]]))
                tols.append(0.0)
        return out, tols

    def best_move(slot):
        """Best candidate for one row: (key, tols, pool position)."""
        used = blocked.copy()
        used[met.rows] = True
        C = np.flatnonzero(~used)
        if not len(C):
            return None
        vals = {}
        for m in key.guarded:
            vals[m] = met.cand_values(m, slot, C)
        surv = np.arange(len(C))
        kv, kt = [], []
        for st in stages:
            if st[0] == 'viol':
                v = key.violation({m: vals[m] for m in key.guarded})
                t = 0.0
            elif st[0] == 'val':
                m = st[1]
                if m not in vals:
                    vals[m] = np.full(len(C), np.nan)
                    vals[m][surv] = met.cand_values(m, slot, C[surv])
                v = key.oriented(m, vals[m])
                if not np.isfinite(v[surv]).any():
                    return None
                t = key.t(m, np.nanmin(np.abs(vals[m][surv])))
            else:
                m = st[1]
                b_val = vals[m][surv]
                ref = b_val[np.argmin(key.oriented(m, b_val))]
                cnt = np.zeros(len(C))
                f = getattr(met, 'cand_count_' + m)
                cnt[surv] = f(slot, C[surv], ref, key.t(m, ref))
                v, t = cnt, 0.0
            vs = v[surv]
            b  = np.nanmin(vs)
            surv = surv[vs <= b + t]
            kv.append(float(b))
            kt.append(t)
        return kv, kt, int(C[surv.min()])

    def phi_sweep(cur, cur_t):
        """
        One pass over the rows that lowers the Morris-Mitchell phi_p
        (p = _PHI_P) of the design without making the key worse. The
        exchange alone stalls on a min_distance plateau: moving one row
        of a closest pair leaves the minimum unchanged while another
        pair still attains it. phi_p counts every small distance, so
        its moves spread the closest pairs and let the exchange improve
        the minimum later. A move is kept only if the exact key is not
        worse; rows in closest pairs are tried first.
        """
        n_acc = 0
        Dm = met.D[:met.m]
        md = float(met.D.min())
        crit = np.flatnonzero((Dm <= md + key.t('min_distance', md))
                              .any(axis=1))
        rest = np.setdiff1d(np.arange(met.m), crit)
        order = np.r_[crit[rng.permutation(len(crit))],
                      rest[rng.permutation(len(rest))]]
        for slot in order:
            slot = int(slot)
            used = blocked.copy()
            used[met.rows] = True
            C = np.flatnonzero(~used)
            if not len(C):
                continue
            cols = np.r_[0:slot, slot + 1:met.n]
            md = float(met.D.min())
            if not md > 0:
                # Coinciding fixed rows: phi_p is infinite for every move.
                break
            with np.errstate(divide='ignore', over='ignore',
                             invalid='ignore'):
                old = float(np.sum((met.D[slot, cols] / md) ** -_PHI_P))
                tot = float(np.sum(
                    (met.D[np.triu_indices(met.n, 1)] / md) ** -_PHI_P))
                Dc = met.Dpool[C][:, cols]
                new = np.sum((Dc / md) ** -_PHI_P, axis=1)
            delta = new - old
            ok = np.flatnonzero(
                (delta < -1e-12 * tot)
                & (Dc.min(axis=1) >= md - key.t('min_distance', md)))
            for k in ok[np.argsort(delta[ok], kind='stable')][:_PHI_TRIES]:
                prev = met.rows.copy()
                met.move(slot, int(C[k]))
                new_key, new_t = state_key()
                if not _better(cur, new_key,
                               [max(a, b) for a, b in zip(cur_t, new_t)]):
                    cur, cur_t = new_key, new_t
                    n_acc += 1
                    break
                met.set_rows(prev)
        return cur, cur_t, n_acc

    def search():
        """Exchange; with min_distance in the priority, alternated
        with phi_p sweeps until neither changes the design."""
        cur, cur_t, n_moves, n_sweeps = local_search()
        if 'min_distance' not in priority:
            return cur, cur_t, n_moves, n_sweeps
        for _ in range(max_sweeps):
            cur, cur_t, n_acc = phi_sweep(cur, cur_t)
            n_moves += n_acc
            n_sweeps += 1
            if not n_acc:
                break
            cur, cur_t, nm, ns = local_search()
            n_moves += nm
            n_sweeps += ns
        return cur, cur_t, n_moves, n_sweeps

    def local_search():
        n_moves = n_sweeps = 0
        cur, cur_t = state_key()
        for _ in range(max_sweeps):
            n_sweeps += 1
            improved = False
            for slot in rng.permutation(len(met.rows)):
                bm = best_move(int(slot))
                if bm is None:
                    continue
                kv, kt, c = bm
                if not _better(kv, cur, [max(a, b)
                                         for a, b in zip(kt, cur_t)]):
                    continue
                # Confirm on the exact values: rounding in the fast
                # scores must not let the search cycle.
                old = met.rows.copy()
                met.move(int(slot), c)
                new, new_t = state_key()
                if _better(new, cur, [max(a, b) for a, b in zip(new_t, cur_t)]):
                    cur, cur_t = new, new_t
                    n_moves += 1
                    improved = True
                else:
                    met.set_rows(old)
            if not improved:
                break
        return cur, cur_t, n_moves, n_sweeps

    best_key, best_t, n_moves, n_sweeps = search()
    best_rows = met.rows.copy()
    for _ in range(n_kicks):
        free = np.flatnonzero(~blocked)
        free = np.setdiff1d(free, met.rows)
        k = min(_KICK_ROWS, len(met.rows), len(free))
        if k == 0:
            break
        slots = rng.choice(len(met.rows), size=k, replace=False)
        rows = met.rows.copy()
        rows[slots] = rng.choice(free, size=k, replace=False)
        met.set_rows(rows)
        kv, kt, nm, ns = search()
        n_moves += nm
        n_sweeps += ns
        if _better(kv, best_key, [max(a, b) for a, b in zip(kt, best_t)]):
            best_key, best_t, best_rows = kv, kt, met.rows.copy()
        else:
            met.set_rows(best_rows)

    out = design.copy()
    out[n_kept:] = pool[best_rows]
    improved = not np.array_equal(np.sort(best_rows), np.sort(start_rows))
    return out, dict(improved=bool(improved), n_moves=n_moves,
                     n_sweeps=n_sweeps)
