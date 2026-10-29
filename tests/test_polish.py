"""Tests for mergen.polish and compare(polish=True)."""
from __future__ import annotations

import numpy as np
import pytest

import mergen
from mergen import polish as P
from mergen.distances import _embed_heom
from mergen.metrics import _compute_metric, _minimax_eval_set

_ALL = ['min_distance', 'minimax', 'max_abs_correlation', 'cd2',
        'projection_cd2']


def _spaces():
    return {
        'numeric': mergen.ParameterSpace({'x': list(range(9)),
                                          'y': list(range(16)),
                                          'z': list(range(5))}),
        'nominal': mergen.ParameterSpace({'a': list(range(11)),
                                          'b': list(range(11)),
                                          'k': ('nominal', ['A', 'B', 'C'])}),
        'one_level': mergen.ParameterSpace({'a': list(range(7)),
                                            'b': [3.0],
                                            'c': list(range(6))}),
    }


@pytest.mark.parametrize('name', ['numeric', 'nominal', 'one_level'])
def test_fast_scores_equal_the_metrics(name):
    # Every candidate score must be the value mergen.metrics gives for
    # the design with that move, fixed rows included.
    sp = _spaces()[name]
    pool_n = (sp.candidate_pool - sp.gmins) / sp.granges
    levels = ([np.unique(pool_n[:, j]) if sp.is_mask[j] else None
               for j in range(sp.n_parameters)] if sp.has_nominal else None)
    emb = (lambda Z: _embed_heom(Z, levels)) if levels else (lambda Z: Z)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pool_n), 12, replace=False)
    rows, fixed = idx[:9], pool_n[idx[9:]]
    met = P._Metrics(sp, pool_n, emb(pool_n), emb(_minimax_eval_set(sp)),
                     fixed, emb(fixed), _ALL)
    met.set_rows(rows)
    X = np.vstack([pool_n[rows], fixed])
    for m, v in met.values().items():
        assert v == pytest.approx(_compute_metric(m, X, space=sp),
                                  rel=1e-10, nan_ok=True)
    C = np.setdiff1d(np.arange(len(pool_n)), idx)[:12]
    for m in _ALL:
        got = met.cand_values(m, 3, C)
        for k, c in enumerate(C):
            r2 = rows.copy()
            r2[3] = c
            want = _compute_metric(m, np.vstack([pool_n[r2], fixed]),
                                   space=sp)
            assert got[k] == pytest.approx(want, rel=1e-10, abs=1e-14,
                                           nan_ok=True), m


def _start(space, n, seed=0, n_kept=0):
    pool = space.candidate_pool
    idx = np.random.default_rng(seed).choice(len(pool), n, replace=False)
    return pool[idx].copy()


@pytest.mark.parametrize('selection,priority', [
    ('lexicographic', ('minimax', 'cd2')),
    ('lexicographic', ('min_distance',)),
    ('pareto', ('min_distance', 'max_abs_correlation')),
    ('pareto', ('projection_cd2', 'minimax')),
])
def test_polish_never_worse_and_keeps_the_rules(selection, priority):
    space = mergen.ParameterSpace({'x': list(range(11)),
                                   'y': list(range(11))})
    space.add_constraint(lambda p: p['x'] + p['y'] <= 16)
    D0 = _start(space, 10, seed=3)
    banned = [space.candidate_pool[0].tolist()]
    if any(np.array_equal(r, banned[0]) for r in D0):
        D0 = D0[1:]
    hb = [m in ('min_distance',) for m in priority]
    tol = np.full(len(priority), 1e-9)
    rtol = np.zeros(len(priority))
    D, info = P.polish(D0, 2, None, space, priority, hb, selection, tol,
                       rtol, banned, np.random.default_rng(1), n_kicks=3)
    # Frozen rows untouched; feasible, distinct, not banned.
    np.testing.assert_array_equal(D[:2], D0[:2])
    assert len({tuple(r) for r in D}) == len(D)
    assert all(r[0] + r[1] <= 16 for r in D)
    assert not any(np.array_equal(r, banned[0]) for r in D)
    before = [_compute_metric(m, (D0 - space.gmins) / space.granges, space)
              for m in priority]
    after = [_compute_metric(m, (D - space.gmins) / space.granges, space)
             for m in priority]
    sign = [-1 if h else 1 for h in hb]
    o_b = [s * v for s, v in zip(sign, before)]
    o_a = [s * v for s, v in zip(sign, after)]
    if selection == 'pareto':
        assert all(a <= b + 1e-9 for a, b in zip(o_a, o_b))
    else:
        assert tuple(np.round(o_a, 9)) <= tuple(np.round(o_b, 9))
    assert info['improved'] == (not np.array_equal(
        np.sort(D, axis=0), np.sort(D0, axis=0)))


def test_polish_is_deterministic():
    space = mergen.ParameterSpace({'x': list(range(9)),
                                   'y': list(range(16))})
    D0 = _start(space, 12, seed=5)
    args = (0, None, space, ('minimax', 'cd2'), [False, False],
            'lexicographic', np.full(2, 1e-9), np.zeros(2), [])
    a, _ = P.polish(D0, *args, rng=np.random.default_rng(7), n_kicks=4)
    b, _ = P.polish(D0, *args, rng=np.random.default_rng(7), n_kicks=4)
    np.testing.assert_array_equal(a, b)


def _grid916():
    s = mergen.Sampler(mergen.ParameterSpace({'x': list(range(9)),
                                              'y': list(range(16))}))
    s.set_design(n_samples=20, n_validation=4)
    return s


def test_compare_reaches_the_proven_minimax_optimum():
    # 0.141667 is the minimax optimum of 20 points on the 9 x 16 grid
    # (set-cover ILP in the 0.1.1 audit); unpolished winners stopped at
    # 0.18 to 0.24.
    cmp = _grid916().compare(priority=('minimax', 'cd2'),
                             selection='lexicographic', seed=44,
                             n_repeats=2, mc_samples=0, verbose=False)
    w = cmp.metrics[(cmp.metrics.criterion == cmp.best[0])
                    & (cmp.metrics.algorithm == cmp.best[1])].iloc[0]
    assert w['minimax'] == pytest.approx(0.141667, abs=1e-6)
    assert cmp.best[1].endswith('+polish')
    # best_result delivers the compared design, with its origin noted.
    D = cmp.best_result.samples[['x', 'y']].to_numpy(dtype=float)
    E = np.asarray(cmp.results[cmp.best].design, dtype=float)
    assert sorted(map(tuple, D)) == sorted(map(tuple, E))
    assert cmp.best_result._meta['polished_from'] == (
        cmp.best[0], cmp.best[1][:-len('+polish')])
    assert len(cmp.best_result.validation) == 4


def test_polish_false_keeps_the_plain_selection():
    cmp = _grid916().compare(criteria=['cd2', 'phi_p'], seed=44,
                             n_repeats=2, mc_samples=0, verbose=False,
                             polish=False)
    assert not any(a.endswith('+polish') for a in cmp.metrics.algorithm)


def test_reselect_polishes_for_the_new_priority():
    # With kicks, reselect() reaches the proven minimax optimum that the
    # sweep under the default priority never targeted.
    cmp = _grid916().compare(criteria=['cd2', 'phi_p'], seed=44,
                             n_repeats=2, mc_samples=0, verbose=False,
                             polish_kicks=10)
    r = cmp.reselect(priority=('minimax',), selection='lexicographic',
                     verbose=False)
    assert r.metrics['minimax'].min() == pytest.approx(0.141667, abs=1e-6)
    again = cmp.reselect(priority=('minimax',), selection='lexicographic',
                         verbose=False)
    assert again.best == r.best


def test_prescribed_points_stay_and_exclusions_stay_free():
    s = _grid916()
    s.add_prescribed([[0, 0]])
    s.add_exclusion([4, 8])
    cmp = s.compare(criteria=['cd2'], priority=('minimax',),
                    selection='lexicographic', seed=1, n_repeats=2,
                    mc_samples=0, verbose=False)
    D = cmp.best_result.samples
    assert ((D.x == 0) & (D.y == 0)).sum() == 1
    assert not ((D.x == 4) & (D.y == 8)).any()
    assert len(D) == 20


def test_summary_names_the_start_of_a_polished_winner(capsys):
    cmp = _grid916().compare(criteria=['cd2'], priority=('minimax',),
                             selection='lexicographic', seed=44,
                             n_repeats=2, mc_samples=0, verbose=False)
    assert cmp.best == ('cd2', 'sa+polish')
    capsys.readouterr()
    cmp.best_result.summary()
    assert 'Polished        : from cd2 + sa (compare)' in capsys.readouterr().out


@pytest.mark.parametrize('bad', [-1, 1.5, True, '3'])
def test_polish_kicks_must_be_a_non_negative_integer(bad):
    with pytest.raises(Exception, match='polish_kicks'):
        _grid916().compare(criteria=['cd2'], polish_kicks=bad,
                           verbose=False)


def test_kicks_follow_the_seed_and_never_lose_to_no_kicks():
    space = mergen.ParameterSpace({'x': list(range(9)),
                                   'y': list(range(16))})
    D0 = _start(space, 12, seed=5)
    args = (0, None, space, ('min_distance',), [True], 'lexicographic',
            np.full(1, 1e-9), np.zeros(1), [])
    a, _ = P.polish(D0, *args, rng=np.random.default_rng(3), n_kicks=0)
    b, _ = P.polish(D0, *args, rng=np.random.default_rng(3), n_kicks=5)
    c, _ = P.polish(D0, *args, rng=np.random.default_rng(3), n_kicks=5)
    np.testing.assert_array_equal(b, c)
    md = [_compute_metric('min_distance', (X - space.gmins) / space.granges,
                          space) for X in (a, b)]
    assert md[1] >= md[0] - 1e-12


def test_reselect_keeps_the_kicks_of_compare(monkeypatch):
    seen = []
    real = P.polish

    def spy(*a, **k):
        seen.append(k.get('n_kicks'))
        return real(*a, **k)
    monkeypatch.setattr(P, 'polish', spy)
    cmp = _grid916().compare(criteria=['cd2'], seed=1, n_repeats=1,
                             mc_samples=0, verbose=False, polish_kicks=2)
    cmp.reselect(priority=('minimax',), selection='lexicographic',
                 verbose=False)
    assert seen and set(seen) == {2}


def test_lexicographic_first_metric_does_not_drift_beyond_tol():
    # Each move may lose up to tol on the first metric; the losses must
    # not add up (found in review: 0.031 lost at tol 0.02).
    sp = mergen.ParameterSpace({'x': list(range(15)), 'y': list(range(15)),
                                'z': list(range(6))})
    pr, tol = ('max_abs_correlation', 'cd2'), np.array([0.02, 1e-9])
    for s in range(8):
        D0 = _start(sp, 14, seed=s)
        D0, _ = P.polish(D0, 0, None, sp, pr, [False, False],
                         'lexicographic', np.full(2, 1e-9), np.zeros(2), [],
                         np.random.default_rng(s))
        D, _ = P.polish(D0, 0, None, sp, pr, [False, False],
                        'lexicographic', tol, np.zeros(2), [],
                        np.random.default_rng(s))
        f = [_compute_metric(pr[0], (X - sp.gmins) / sp.granges, sp)
             for X in (D0, D)]
        assert f[1] <= f[0] + tol[0] + 1e-12
