"""Tests for mergen.selection and its use in Sampler.compare / run."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import mergen
from mergen.selection import resolve_tolerance, select


# ─────────────────────────────────────────────────────────────────────
# select(): lexicographic
# ─────────────────────────────────────────────────────────────────────
class TestLexicographic:
    def test_priority_order_decides(self):
        v = np.array([[1.0, 5.0],    # A: best on metric 0
                      [2.0, 1.0]])   # B: best on metric 1
        assert select(v, [False, False]).winner == 0
        assert select(v[:, ::-1], [False, False]).winner == 1

    def test_tie_within_tol_falls_through(self):
        v = np.array([[1.0, 5.0],
                      [1.0 + 5e-10, 1.0]])
        assert select(v, [False, False]).winner == 1
        zero = np.zeros(2)
        assert select(v, [False, False], tol=zero).winner == 0

    def test_tolerance_does_not_chain(self):
        t = 1e-3
        v = np.array([[0.0,     3.0],
                      [0.9 * t, 2.0],
                      [1.8 * t, 1.0]])   # within t of row 1, not of row 0
        sel = select(v, [False, False], tol=np.array([t, 0.0]))
        assert sel.winner == 1

    def test_relative_tolerance(self):
        v = np.array([[100.0, 5.0],
                      [100.5, 1.0]])
        assert select(v, [False, False]).winner == 0
        sel = select(v, [False, False], rtol=np.array([0.01, 0.0]))
        assert sel.winner == 1

    def test_higher_is_better(self):
        v = np.array([[0.2, 0.0], [0.3, 0.0]])
        assert select(v, [True, False]).winner == 1

    def test_exact_tie_reported_and_tiebreak(self):
        v = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 0.0]])
        sel = select(v, [False, False])
        assert sel.ties == (0, 1)
        assert sel.winner == 0
        assert select(v, [False, False], tiebreak=[5.0, 1.0, 0.0]).winner == 1


# ─────────────────────────────────────────────────────────────────────
# select(): pareto
# ─────────────────────────────────────────────────────────────────────
class TestPareto:
    def test_winner_is_never_dominated(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            v = rng.random((6, 3))
            w = select(v, [False] * 3, rule='pareto').winner
            for j in range(len(v)):
                assert not (np.all(v[j] <= v[w]) and np.any(v[j] < v[w]))

    def test_saturated_metric_does_not_decide(self):
        # Metric 0 differs only by float noise; metric 1 must decide.
        v = np.array([[1.0, 0.5],
                      [1.0 + 1e-15, 0.2]])
        assert select(v, [False, False], rule='pareto').winner == 1

    def test_large_tolerance_never_crashes(self):
        rng = np.random.default_rng(1)
        tol = np.full(3, 0.25)
        for _ in range(2000):
            v = rng.random((5, 3))
            sel = select(v, [False] * 3, rule='pareto', tol=tol)
            assert sel.front and sel.winner in range(5)


@pytest.mark.parametrize("rule", ['lexicographic', 'pareto'])
def test_candidate_order_does_not_matter(rule):
    rng = np.random.default_rng(2)
    for _ in range(100):
        v = rng.random((5, 2))
        w = select(v, [False, True], rule=rule).winner
        for perm in itertools.islice(itertools.permutations(range(5)), 10):
            p = list(perm)
            assert p[select(v[p], [False, True], rule=rule).winner] == w


def test_nan_candidates_excluded():
    v = np.array([[np.nan, 0.0], [1.0, 1.0], [2.0, 0.0]])
    sel = select(v, [False, False])
    assert sel.excluded == (0,)
    assert sel.winner == 1
    with pytest.raises(ValueError):
        select(np.array([[np.nan], [np.nan]]), [False])


def test_unknown_rule():
    with pytest.raises(ValueError):
        select(np.zeros((2, 1)), [False], rule='weighted')


class TestResolveTolerance:
    def test_scalar_and_mapping(self):
        assert resolve_tolerance(0.1, ['a', 'b'], 'tol').tolist() == [0.1, 0.1]
        out = resolve_tolerance({'b': 0.5}, ['a', 'b'], 'tol')
        assert out.tolist() == [1e-9, 0.5]
        assert resolve_tolerance({}, ['a'], 'rtol').tolist() == [0.0]

    @pytest.mark.parametrize("bad", [-1.0, float('nan'), 'x', {'c': 0.1}])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            resolve_tolerance(bad, ['a', 'b'], 'tol')


# ─────────────────────────────────────────────────────────────────────
# Sampler integration
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def sampler():
    space = mergen.ParameterSpace({
        'kappa': [round(v, 1) for v in np.arange(0.2, 1.01, 0.1)],
        'mass':  list(range(1000, 2501, 100)),
    })
    s = mergen.Sampler(space)
    s.set_design(n_samples=10, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=150)
    s.set_optimizer('sce', n_restarts=1, max_iter=150)
    return s


_KW = dict(n_repeats=2, mc_samples=20, verbose=False)


def _lexi_best(metrics, priority):
    from mergen.metrics import _HIGHER_BETTER
    v = metrics[list(priority)].to_numpy(dtype=float)
    return select(v, [m in _HIGHER_BETTER for m in priority]).winner


class TestCompareSelection:
    def test_winner_is_lexicographic_best_on_raw_values(self, sampler):
        prio = ('min_distance', 'max_abs_correlation')
        cmp = sampler.compare(criteria=['cd2', 'phi_p', 'maxpro'],
                              priority=prio, **_KW)
        i = _lexi_best(cmp.metrics, prio)
        assert cmp.best == (cmp.metrics.criterion[i],
                            cmp.metrics.algorithm[i])

    def test_listing_order_does_not_decide(self, sampler):
        a = sampler.compare(criteria=['phi_p', 'cd2'], **_KW)
        b = sampler.compare(criteria=['cd2', 'phi_p'], **_KW)
        if len(a.ties) == 1:
            assert a.best == b.best

    def test_priority_order_changes_the_rule(self, sampler):
        kw = dict(criteria=['cd2', 'phi_p', 'maxpro'],
                  algorithms=['sa', 'sce'], **_KW)
        cmp = sampler.compare(priority=('max_abs_correlation',
                                        'min_distance'), **kw)
        i = _lexi_best(cmp.metrics, ('max_abs_correlation', 'min_distance'))
        assert cmp.best == tuple(cmp.metrics.loc[i, ['criterion',
                                                     'algorithm']])

    def test_pareto_winner_not_dominated(self, sampler):
        prio = ('min_distance', 'max_abs_correlation')
        cmp = sampler.compare(criteria=['cd2', 'phi_p', 'maxpro'],
                              algorithms=['sa', 'sce'], priority=prio,
                              selection='pareto', **_KW)
        v = cmp.metrics[list(prio)].to_numpy(dtype=float) * [-1, 1]
        keys = list(zip(cmp.metrics.criterion, cmp.metrics.algorithm))
        w = v[keys.index(cmp.best)]
        assert not any(np.all(x <= w) and np.any(x < w - 1e-9) for x in v)
        assert cmp.best in cmp.front

    def test_result_is_consistent(self, sampler, capsys):
        cmp = sampler.compare(criteria=['cd2', 'phi_p'], **_KW)
        # verbose=False: no run summary, only warnings.
        out = capsys.readouterr().out.splitlines()
        assert all('[WARNING]' in line for line in out if line.strip())
        starred = cmp.table[cmp.table.best.str.strip() == '*']
        assert len(starred) == 1
        assert tuple(starred.iloc[0][['criterion', 'algorithm']]) == cmp.best
        bc, ba = cmp.best
        np.testing.assert_array_equal(
            cmp.best_result.algorithm_results[ba].design,
            cmp.results[cmp.best].design)
        with pytest.raises(AttributeError):
            cmp.best = ('cd2', 'sa')

    def test_summary_names_the_rule(self, sampler, capsys):
        lex = sampler.compare(criteria=['cd2', 'phi_p'], **_KW)
        par = lex.reselect(selection='pareto', verbose=False)
        capsys.readouterr()
        lex.summary()
        out = capsys.readouterr().out
        assert 'lexicographic' in out and ' > ' in out
        par.summary()
        out = capsys.readouterr().out
        assert 'Pareto' in out and ' > ' not in out

    def test_reselect(self, sampler):
        cmp = sampler.compare(criteria=['cd2', 'phi_p', 'maxpro'], **_KW)
        before = cmp.best
        prio = ('projection_cd2',)
        new = cmp.reselect(priority=prio, verbose=False)
        assert cmp.best == before and cmp.priority != new.priority
        assert new.priority == prio
        i = _lexi_best(new.metrics, prio)
        assert new.best == (new.metrics.criterion[i],
                            new.metrics.algorithm[i])

    def test_mc_samples_zero(self, sampler):
        cmp = sampler.compare(criteria=['cd2', 'phi_p'], n_repeats=1,
                              mc_samples=0, verbose=False)
        assert cmp.table['min_distance'].isna().all()
        assert cmp.best in cmp.results

    def test_argument_handling(self, sampler):
        cmp = sampler.compare(criteria=['cd2'], algorithms=['SA'],
                              priority='min_distance', n_repeats=1,
                              mc_samples=0, verbose=False)
        assert cmp.best == ('cd2', 'sa') and cmp.priority == ('min_distance',)
        bad = [dict(priority=()),
               dict(priority=('min_distance', 'min_distance')),
               dict(priority=('no_such_metric',)),
               dict(selection='weighted'),
               dict(tol={'minimax': 0.1}),
               dict(tol=-1.0),
               dict(mc_samples=-1),
               dict(criteria=['cd2', 3])]
        for kw in bad:
            kw = {'criteria': ['cd2'], **kw}
            with pytest.raises(ValueError):
                sampler.compare(n_repeats=1, verbose=False, **kw)

    def test_aliases_run_once(self, sampler, capsys):
        cmp = sampler.compare(criteria=['phi_p', 'phip'], n_repeats=1,
                              mc_samples=0, verbose=False)
        assert list(cmp.metrics.criterion) == ['phi_p']
        assert 'more than once' in capsys.readouterr().out

    def test_tolerance_changes_the_winner(self, sampler):
        prio = ('min_distance', 'max_abs_correlation')
        crit = ['cd2', 'phi_p', 'maxpro']
        exact = sampler.compare(criteria=crit, priority=prio, **_KW)
        # A tolerance wider than any spread ties the first metric, so
        # the second one alone decides.
        loose = exact.reselect(tol={'min_distance': 10.0}, verbose=False)
        assert loose.best == tuple(exact.metrics.iloc[
            int(exact.metrics['max_abs_correlation'].argmin())][
            ['criterion', 'algorithm']])
        assert loose.best != exact.best

    def test_run_warnings_reach_the_user(self, sampler, capsys):
        sampler.set_design(n_samples=5, n_validation=0)
        sampler.compare(criteria=['cd2'], n_repeats=1, mc_samples=0,
                        verbose=False)
        assert 'n_samples (5) < recommended' in capsys.readouterr().out


class TestRunRepeatSelection:
    def test_lexicographic_repeat(self, sampler):
        prio = ('min_distance', 'max_abs_correlation')
        r = sampler.run(criteria='cd2', seed=3, n_repeats=4,
                        priority=prio, verbose=False)
        from mergen.sampler import _priority_values
        reps = r._repeats_by_alg['sa']
        vals = np.array([_priority_values(x.design, prio, sampler.space)
                         for x in reps])
        w = select(vals, [True, False],
                   tiebreak=[x.score for x in reps]).winner
        np.testing.assert_array_equal(r.algorithm_results['sa'].design,
                                      reps[w].design)

    def test_without_priority_lowest_score(self, sampler):
        r = sampler.run(criteria='cd2', seed=3, n_repeats=3, verbose=False)
        reps = r._repeats_by_alg['sa']
        assert r.best_score == min(x.score for x in reps)


class TestReviewRegressions:
    def test_pareto_two_member_front_reports_priority_order(self):
        v = np.array([[0.30, 0.20], [0.20, 0.05]])
        sel = select(v, [True, False], rule='pareto')
        assert sel.front == (0, 1) and sel.ties == (0, 1)
        assert sel.decided_by == 'priority order'
        assert select(np.array([[1.0, 1.0], [2.0, 2.0]]), [False, False],
                      rule='pareto').decided_by == 'metrics'

    def test_non_finite_excluded(self):
        v = np.array([[np.inf, 1.0], [1.0, 0.0]])
        for rule in ('lexicographic', 'pareto'):
            sel = select(v, [True, False], rule=rule)
            assert sel.excluded == (0,) and sel.winner == 1

    @pytest.mark.parametrize("bad", [True, {'a': 'x'}, {'a': True}, '0.1',
                                     np.array([0.1])])
    def test_tolerance_rejects_bool_and_text(self, bad):
        with pytest.raises(ValueError, match="non-negative number"):
            resolve_tolerance(bad, ['a'], 'tol')

    def test_chained_reselect_keeps_tolerances(self, sampler):
        c = sampler.compare(criteria=['cd2', 'phi_p'],
                            tol={'min_distance': 0.05}, **_KW)
        r1 = c.reselect(priority='max_abs_correlation', verbose=False)
        r2 = r1.reselect(priority=('min_distance', 'max_abs_correlation'),
                         verbose=False)
        assert r2.tol['min_distance'] == 0.05
        with pytest.raises(ValueError):
            c.reselect(priority='minimax', tol={'min_distance': 0.1},
                       verbose=False)

    def test_pareto_front_of_two_is_not_always_a_tie(self):
        # Scaling uses every valid candidate, so a two-member front can
        # still be decided on Utopia distance.
        sel = select(np.array([[0, 1], [1, 0], [4, 2]]), [False, False],
                     rule='pareto')
        assert sel.front == (0, 1) and sel.decided_by == 'metrics'

    def test_pareto_tie_text_does_not_claim_tolerance(self):
        from mergen.sampler import ComparisonResult
        c = ComparisonResult(table=None, metrics=None, results={},
                             best=('cd2', 'sa'), priority=('min_distance',),
                             selection='pareto',
                             ties=(('cd2', 'sa'), ('phi_p', 'sa')),
                             decided_by='tiebreak')
        assert 'Pareto' in c._tie_text()
        assert 'within the tolerance' not in c._tie_text()

    def test_run_without_priority_ignores_selection_args(self, sampler,
                                                          capsys):
        kw = dict(criteria='cd2', algorithm='sa', seed=3, verbose=False)
        ref = sampler.run(**kw).samples
        capsys.readouterr()
        r = sampler.run(tol={'minimax': 0.1}, selection='pareto', **kw)
        assert 'apply only when a priority' in capsys.readouterr().out
        assert r.samples.equals(ref)

    def test_multi_algorithm_run_honours_priority(self, sampler):
        prio = ('max_abs_correlation', 'min_distance')
        r = sampler.run(criteria='cd2', algorithm=['sa', 'sce'], seed=1,
                        n_repeats=2, priority=prio, verbose=False)
        # The scenario must separate the rules: the lowest criterion
        # score (the 0.1.0 choice) is not the priority winner here.
        scores = {a: res.score for a, res in r.algorithm_results.items()}
        assert min(scores, key=scores.get) != r.best_algorithm
        from mergen.sampler import _priority_values
        names = list(r.algorithm_results)
        vals = np.array([_priority_values(r.algorithm_results[a].design,
                                          prio, sampler.space)
                         for a in names])
        w = select(vals, [False, True],
                   tiebreak=[r.algorithm_results[a].score
                             for a in names]).winner
        assert r.best_algorithm == names[w]
