"""
Criteria and metrics against their published definitions (audit items
A1-01, A1-02, A1-03, A1-04, A1-06, A1-08, A1-10).

Each test computes the reference value independently, from the
formula in the cited paper, and compares it with Mergen's value.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

import mergen
from mergen.criteria import CD2, MaxPro, MaxProQQ, PhiP, QQD, StratifiedL2, \
    UMaxPro


def _norm(space, D):
    return (np.asarray(D, float) - space.gmins) / space.granges


def _lhd(n, seed):
    """An n-run Latin hypercube on two n-level axes (level indices)."""
    rng = np.random.default_rng(seed)
    return np.column_stack([np.arange(n), rng.permutation(n)])


# ── References ───────────────────────────────────────────────────────

def _ref_maxpro(levels, periodic):
    """Sum over pairs of 1 / prod_v term_v, Joseph, Gul & Ba (2020)
    Eq. 9 for an axis with m < n levels, the published form otherwise;
    uMaxPro places level k of m at (k + 1/2) / m on the circle
    (Vorechovsky & Masek)."""
    L = np.asarray(levels, float)
    n, d = L.shape
    m = L.max(axis=0) + 1
    total = 0.0
    for i, j in combinations(range(n), 2):
        prod = 1.0
        for v in range(d):
            if periodic:
                t_i, t_j = (L[i, v] + .5) / m[v], (L[j, v] + .5) / m[v]
                dd = abs(t_i - t_j)
                dd = min(dd, 1 - dd)
            else:
                dd = abs(L[i, v] - L[j, v]) / max(m[v] - 1, 1)
            c = 1.0 / m[v] if m[v] < n else 0.0
            prod *= (dd + c) ** 2
        total += 1.0 / prod
    return total


def _space_levels(*ms):
    return mergen.ParameterSpace({f'x{k}': list(range(m))
                                  for k, m in enumerate(ms)})


# ── A1-01: uMaxPro on stratum midpoints ──────────────────────────────

@pytest.mark.parametrize('n', [5, 7, 9])
def test_umaxpro_is_invariant_under_cyclic_shifts(n):
    space = _space_levels(n, n)
    crit = UMaxPro()
    D = _lhd(n, n)
    ref = crit.evaluate(_norm(space, D), space)
    for s in range(1, n):
        shifted = np.column_stack([(D[:, 0] + s) % n, D[:, 1]])
        assert crit.evaluate(_norm(space, shifted), space) == \
            pytest.approx(ref, rel=1e-12)
    assert ref == pytest.approx(_ref_maxpro(D, periodic=True), rel=1e-12)


# ── A1-02: few-level axes (Joseph, Gul & Ba 2020, Eq. 9) ─────────────

@pytest.mark.parametrize('crit, periodic', [(MaxPro(), False),
                                            (UMaxPro(), True)])
def test_few_level_axis_uses_the_published_term(crit, periodic):
    # 12 runs, axis 0 has 4 levels (< n), axis 1 has 12 (>= n).
    space = _space_levels(4, 12)
    rng = np.random.default_rng(3)
    D = np.column_stack([rng.integers(0, 4, 12), rng.permutation(12)])
    got = crit.evaluate(_norm(space, D), space)
    assert got == pytest.approx(_ref_maxpro(D, periodic), rel=1e-12)


def test_ties_no_longer_dominate_maxpro():
    space = _space_levels(3, 12)
    D = np.column_stack([np.arange(12) % 3, np.arange(12)])
    score = MaxPro().evaluate(_norm(space, D), space)
    assert score < 1e6          # a clamped tie alone contributed 1e10


# ── A1-03: MaxProQQ continuous column with fewer than n levels ───────

def test_maxproqq_scores_a_coarse_continuous_column_as_discrete():
    space = mergen.ParameterSpace({
        'a': ('continuous', 0.0, 1.0, {'resolution': 5}),
        'b': ('continuous', 0.0, 1.0, {'resolution': 20})})
    rng = np.random.default_rng(1)
    D = np.column_stack([rng.integers(0, 5, 12) / 4,
                         rng.permutation(20)[:12] / 19])
    X = _norm(space, D)
    ref = 0.0
    for i, j in combinations(range(12), 2):
        a = (abs(X[i, 0] - X[j, 0]) + 1 / 5) ** 2     # 5 levels < 12
        b = (X[i, 1] - X[j, 1]) ** 2                  # 20 levels >= 12
        ref += 1 / (a * b)
    assert MaxProQQ().evaluate(X, space) == pytest.approx(ref, rel=1e-12)


# ── A1-04: phi_p updates without cancellation ────────────────────────

def test_phi_p_update_survives_a_dominant_pair():
    space = _space_levels(101, 101)
    X = np.array([[0, 0], [1, 0], [50, 50], [100, 0], [0, 100]]) / 100
    crit = PhiP()
    score = crit.evaluate(X, space)
    new_pt = np.array([0.3, 0.7])
    _, inc = crit.incremental(X, 1, new_pt, space, score)
    X2 = X.copy()
    X2[1] = new_pt
    assert inc == pytest.approx(crit.evaluate(X2, space), rel=1e-9)
    cache = crit.begin_1d(X, 1, score)
    _, inc1, _ = crit.try_1d(cache, 1, 0.7, space)
    X3 = X.copy()
    X3[1, 1] = 0.7
    assert inc1 == pytest.approx(crit.evaluate(X3, space), rel=1e-9)


def test_phi_p_with_a_large_p_stays_finite_on_a_duplicate():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    assert np.isfinite(PhiP(p=50).evaluate(X, None))


# ── A1-06: QQD level map (Zhang, Yang & Zhou 2021, p. 6) ─────────────

def _ref_wd2(levels, s):
    """QQD^2 with no qualitative column: C + mean of the product of
    3/2 - |t-z| + |t-z|^2 over all pairs, levels at (2x+1)/(2s)."""
    T = (2 * np.asarray(levels, float) + 1) / (2 * np.asarray(s, float))
    n, q = T.shape
    tot = 0.0
    for i in range(n):
        for j in range(n):
            a = np.abs(T[i] - T[j])
            tot += np.prod(1.5 - a + a * a)
    return -(4 / 3) ** q + tot / n ** 2


@pytest.mark.parametrize('n', [5, 7])
def test_qqd_uses_midpoint_levels_and_is_shift_invariant(n):
    space = _space_levels(n, n)
    D = _lhd(n, 2 * n)
    crit = QQD()
    ref = crit.evaluate(_norm(space, D), space)
    assert ref == pytest.approx(_ref_wd2(D, [n, n]), rel=1e-12)
    shifted = np.column_stack([(D[:, 0] + 2) % n, D[:, 1]])
    assert crit.evaluate(_norm(space, shifted), space) == \
        pytest.approx(ref, rel=1e-12)


# ── A1-08: projection CD2 is the mean of CD2^2 (Sun, Wang & Xu 2019) ─

def test_projection_cd2_is_the_mean_squared_discrepancy():
    from scipy.stats import qmc
    from mergen.metrics import projection_cd2
    rng = np.random.default_rng(5)
    X = rng.random((10, 4))
    ref = np.mean([qmc.discrepancy(X[:, [a, b]], method='CD')
                   for a, b in combinations(range(4), 2)])
    assert projection_cd2(X) == pytest.approx(ref, rel=1e-10)


# ── A1-10: a one-level axis does not take part ───────────────────────

@pytest.mark.parametrize('crit', [CD2(), StratifiedL2(), QQD(), MaxPro(),
                                  UMaxPro(), PhiP(), MaxProQQ()])
def test_a_one_level_axis_changes_no_criterion(crit):
    full = mergen.ParameterSpace({'a': list(range(6)), 'c': [3.0],
                                  'b': list(range(8))})
    part = mergen.ParameterSpace({'a': list(range(6)),
                                  'b': list(range(8))})
    rng = np.random.default_rng(2)
    D = np.column_stack([rng.integers(0, 6, 7), rng.permutation(8)[:7]])
    Dfull = np.column_stack([D[:, 0], np.full(7, 3.0), D[:, 1]])
    assert crit.evaluate(_norm(full, Dfull), full) == pytest.approx(
        crit.evaluate(_norm(part, D), part), rel=1e-12)


def test_a_one_level_axis_changes_no_metric():
    from mergen.metrics import _compute_metric, _METRIC_FN
    full = mergen.ParameterSpace({'a': list(range(6)), 'c': [3.0],
                                  'b': list(range(8))})
    part = mergen.ParameterSpace({'a': list(range(6)),
                                  'b': list(range(8))})
    rng = np.random.default_rng(2)
    D = np.column_stack([rng.integers(0, 6, 7), rng.permutation(8)[:7]])
    Dfull = np.column_stack([D[:, 0], np.full(7, 3.0), D[:, 1]])
    for m in _METRIC_FN:
        assert _compute_metric(m, _norm(full, Dfull), full) == pytest.approx(
            _compute_metric(m, _norm(part, D), part), rel=1e-12), m


# ── Every update path agrees with a full evaluation ──────────────────

@pytest.mark.parametrize('crit', [CD2(), StratifiedL2(), QQD(), MaxPro(),
                                  UMaxPro(), PhiP(), MaxProQQ()])
def test_updates_match_full_evaluation_on_awkward_spaces(crit):
    space = mergen.ParameterSpace({'a': list(range(3)), 'c': [1.0],
                                   'b': list(range(15)),
                                   'g': ('ordinal', ['lo', 'mid', 'hi'])})
    rng = np.random.default_rng(9)
    pool = _norm(space, space.candidate_pool)
    X = pool[rng.choice(len(pool), 10, replace=False)].copy()
    score = crit.evaluate(X, space)
    for _ in range(25):
        i = int(rng.integers(0, 10))
        new = pool[int(rng.integers(0, len(pool)))]
        _, inc = crit.incremental(X, i, new, space, score)
        X2 = X.copy()
        X2[i] = new
        assert inc == pytest.approx(crit.evaluate(X2, space), rel=1e-8)
        axis = int(rng.integers(0, 4))
        cache = crit.begin_1d(X, i, score, space)
        _, inc1, _ = crit.try_1d(cache, axis, new[axis], space)
        X3 = X.copy()
        X3[i, axis] = new[axis]
        assert inc1 == pytest.approx(crit.evaluate(X3, space), rel=1e-8)
        X, score = X2, crit.evaluate(X2, space)


def test_sce_hands_the_space_to_the_fast_update():
    from mergen.algorithms.sce import _begin_1d
    space = _space_levels(3, 12)
    X = _norm(space, np.column_stack([np.arange(6) % 3, np.arange(6)]))
    crit = MaxPro()
    cache = _begin_1d(crit, X, 0, crit.evaluate(X, space), space)
    assert cache['c'][0] == pytest.approx(1 / 3)

    class Old(CD2):                     # a criterion with the old signature
        def begin_1d(self, X, i, current_score):
            return {'X': X, 'i': i, 'score': current_score}
    assert _begin_1d(Old(), X, 0, 1.0, space)['i'] == 0


def test_constrained_levels_are_the_feasible_ones():
    # A constraint leaves the nominal factor two of its three levels:
    # MaxProQQ's 1/L and QQD's constant C use L = 2.
    space = mergen.ParameterSpace({'x': list(range(6)),
                                   'g': ('nominal', ['a', 'b', 'c'])})
    space.add_constraint(lambda p: p['g'] != 'c')
    assert space.feasible_levels.tolist() == [6, 2]
    rng = np.random.default_rng(4)
    pool = _norm(space, space.candidate_pool)
    X = pool[rng.choice(len(pool), 8, replace=False)]
    ref = 0.0
    for i, j in combinations(range(8), 2):
        a = (abs(X[i, 0] - X[j, 0]) + 1 / 6) ** 2     # 6 levels < 8 rows
        g = (float(X[i, 1] != X[j, 1]) + 1 / 2) ** 2
        ref += 1 / (a * g)
    assert MaxProQQ().evaluate(X, space) == pytest.approx(ref, rel=1e-12)
    T = (X[:, 0] * 5 + 0.5) / 6
    tot = sum((5 / 4) * (6 / 5) ** (X[i, 1] == X[j, 1])
              * (1.5 - abs(T[i] - T[j]) + (T[i] - T[j]) ** 2)
              for i in range(8) for j in range(8))
    C = -((5 * 2 + 1) / (4 * 2)) * (4 / 3)
    assert QQD().evaluate(X, space) == pytest.approx(C + tot / 64, rel=1e-12)


@pytest.mark.parametrize('crit', [MaxPro(), UMaxPro(), MaxProQQ()])
def test_removing_a_dominant_tie_keeps_the_update_exact(crit):
    # Rows 0 and 1 almost coincide on both axes, so their pair holds
    # nearly all of the score; moving row 1 away must not leave the
    # update with only rounding error.
    space = mergen.ParameterSpace({
        k: ('continuous', 0.0, 1.0, {'resolution': 201}) for k in 'ab'})
    X = np.array([[0.5, 0.5], [0.5 + 1e-5, 0.5 + 1e-5], [0.1, 0.9],
                  [0.9, 0.1]])
    score = crit.evaluate(X, space)
    new = np.array([0.2, 0.3])
    _, inc = crit.incremental(X, 1, new, space, score)
    X2 = X.copy()
    X2[1] = new
    assert inc == pytest.approx(crit.evaluate(X2, space), rel=1e-9)
    if hasattr(crit, '_terms'):
        cache = crit.begin_1d(X, 1, score, space)
        _, inc1, cache = crit.try_1d(cache, 0, 0.2, space)
        _, inc2, _ = crit.try_1d(cache, 1, 0.3, space)
        assert inc2 == pytest.approx(crit.evaluate(X2, space), rel=1e-9)


def test_a_cache_built_without_a_space_stays_consistent():
    # begin_1d without a space fixes the plain terms; try_1d must not
    # mix in the space's scale or offsets afterwards.
    # 12 rows, so row 2 holds a small share and the O(n) path is used.
    space = _space_levels(3, 24)
    X = _norm(space, np.column_stack([np.arange(12) % 3,
                                      2 * np.arange(12)]))
    for crit in (MaxPro(), UMaxPro()):
        score = crit.evaluate(X, None)
        cache = crit.begin_1d(X, 2, score)
        _, got, _ = crit.try_1d(cache, 1, 0.95, space)
        X3 = X.copy()
        X3[2, 1] = 0.95
        assert got == pytest.approx(crit.evaluate(X3, None), rel=1e-9)


@pytest.mark.parametrize('crit', [MaxPro(), UMaxPro(), MaxProQQ(),
                                  PhiP(), PhiP(p=50)])
def test_chains_of_fast_updates_stay_accurate(crit):
    # Swaps that pass through ties on one or two axes, every result
    # adopted: the error must not build up between full evaluations.
    import copy
    space = _space_levels(4, 4, 30)
    rng = np.random.default_rng(11)
    pool = _norm(space, space.candidate_pool)
    X = pool[rng.choice(len(pool), 12, replace=False)].copy()
    ref = copy.deepcopy(crit)       # full evaluations on its own instance
    score = crit.evaluate(X, space)
    worst = 0.0
    for _ in range(200):
        i = int(rng.integers(0, 12))
        new = pool[int(rng.integers(0, len(pool)))]
        _, score = crit.incremental(X, i, new, space, score)
        X[i] = new
        exact = ref.evaluate(X.copy(), space)
        worst = max(worst, abs(score - exact) / exact)
    assert worst < 1e-6


def test_phi_p_chained_update_prices_a_duplicate_exactly():
    # Two chained one-axis moves turn row 1 into a copy of row 0; the
    # copy must meet the distance floor, not a rounding residue.
    import copy
    crit = PhiP()
    rng = np.random.default_rng(1)
    for _ in range(30):
        X = rng.integers(0, 4, (8, 3)) / 3.0
        X[1] = X[0] + np.array([1, 2, 0]) / 3.0 * (X[0] < 0.5) \
            - np.array([1, 2, 0]) / 3.0 * (X[0] >= 0.5)
        ref = copy.deepcopy(crit)
        score = crit.evaluate(X, None)
        cache = crit.begin_1d(X, 1, score)
        _, _, cache = crit.try_1d(cache, 0, X[0, 0], None)
        _, got, _ = crit.try_1d(cache, 1, X[0, 1], None)
        X2 = X.copy()
        X2[1, :2] = X[0, :2]
        assert got == pytest.approx(ref.evaluate(X2, None), rel=1e-9)


@pytest.mark.parametrize('crit', [MaxPro(), UMaxPro()])
def test_one_instance_follows_changes_to_the_space(crit):
    # The cached scale and offsets belong to one pool; a constraint, a
    # new resolution or another space of the same shape must not reuse
    # them.
    space = mergen.ParameterSpace({'a': ('continuous', 0, 1),
                                   'b': list(range(12))})
    X = np.column_stack([np.linspace(0, 1, 6), np.arange(6) / 11])
    crit.evaluate(X, space)
    space.add_constraint(lambda p: p['b'] < 4)
    other = type(crit)()
    assert crit.evaluate(X, space) == pytest.approx(other.evaluate(X, space))
    space.set_resolution(3)
    assert crit.evaluate(X, space) == pytest.approx(
        type(crit)().evaluate(X, space))
    alt = _space_levels(5, 5)
    assert crit.evaluate(X, alt) == pytest.approx(
        type(crit)().evaluate(X, alt))


def test_a_recomputed_fast_update_uses_the_cache_space():
    # A cache built without a space recomputes without one too.
    import copy
    space = mergen.ParameterSpace({
        k: ('continuous', 0.0, 1.0, {'resolution': 201}) for k in 'ab'})
    X = np.array([[0.5, 0.5], [0.5 + 1e-5, 0.5 + 1e-5], [0.1, 0.9],
                  [0.9, 0.1]])
    for crit in (MaxPro(), UMaxPro()):
        ref = copy.deepcopy(crit)
        cache = crit.begin_1d(X, 1, crit.evaluate(X, None))
        _, got, _ = crit.try_1d(cache, 0, 0.2, space)
        X2 = X.copy()
        X2[1, 0] = 0.2
        assert got == pytest.approx(ref.evaluate(X2, None), rel=1e-9)
