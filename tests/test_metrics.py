"""Tests for mergen.metrics."""
from __future__ import annotations

import numpy as np
import pytest

import mergen
from mergen.metrics import (
    min_distance, max_abs_correlation, projection_cd2,
)


@pytest.fixture
def num_result(num_space):
    s = mergen.Sampler(num_space)
    s.set_design(n_samples=15, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=1500)
    return s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=44, verbose=False)


@pytest.fixture
def mixed_result(mixed_space):
    s = mergen.Sampler(mixed_space)
    s.set_design(n_samples=12, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=500)
    return s.run(n_repeats=1, criteria='maxproqq', algorithm='sa',
                 seed=1, verbose=False)


class TestMetricFunctionsBackwardCompat:
    def test_min_distance_no_space(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
        assert np.isclose(min_distance(X), np.sqrt(2))

    def test_locked_values_numerical(self, num_result):
        rp = num_result.quality_report(verbose=False)
        # locked pre-categorical values
        assert np.isclose(rp['min_distance'], 0.141421, atol=1e-4)
        # One pair of axes: the mean of CD2² is CD2² = 0.059619².
        assert np.isclose(rp['projection_cd2'], 0.059619 ** 2, rtol=1e-4)


class TestMetricFunctionsWithSpace:
    def test_heom_path_active(self, mixed_space):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        d = min_distance(X, space=mixed_space)
        # HEOM: sqrt(1² + 1²) = sqrt(2)
        assert np.isclose(d, np.sqrt(2))

    def test_projection_cd2_skips_nominal(self, mixed_space):
        X = np.random.default_rng(0).random((10, 2))
        # Only 1 non-nominal column → returns 0
        assert projection_cd2(X, space=mixed_space) == 0.0

    def test_correlation_skips_nominal(self, mixed_space):
        X = np.random.default_rng(0).random((10, 2))
        assert max_abs_correlation(X, space=mixed_space) == 0.0


class TestQualityReport:
    def test_returns_dict(self, num_result):
        rp = num_result.quality_report(verbose=False)
        assert isinstance(rp, dict)
        for m in ['min_distance', 'mean_distance', 'cv_distances', 'minimax',
                  'max_abs_correlation', 'projection_cd2']:
            assert m in rp

    def test_criteria_metrics(self, num_result):
        rp = num_result.quality_report(
            criteria_metrics=['maxpro', 'phi_p'], verbose=False,
        )
        # keys are prefixed 'criterion_'
        assert 'criterion_maxpro' in rp
        assert 'criterion_phi_p' in rp
        assert np.isfinite(rp['criterion_maxpro'])

    def test_mc_baseline(self, num_result):
        rp = num_result.quality_report(mc_samples=30, verbose=False)
        assert 'mc_samples' in rp
        # baseline_median columns should be present for each metric
        assert 'min_distance_baseline_median' in rp

    def test_mixed_space_report(self, mixed_result):
        rp = mixed_result.quality_report(verbose=False)
        assert rp['projection_cd2'] == 0.0
        assert rp['max_abs_correlation'] == 0.0
        assert np.isfinite(rp['min_distance'])


# ─────────────────────────────────────────────────────────────────────
# Feasible baseline and exact minimax
# ─────────────────────────────────────────────────────────────────────
def _triangle():
    s = mergen.ParameterSpace({'a': list(range(10)), 'b': list(range(10))})
    s.add_constraint(lambda p: p['a'] + p['b'] <= 9)
    return s


def _norm(space, pts):
    return (np.asarray(pts, float) - space.gmins) / space.granges


def _brute_minimax(X, space):
    from mergen.distances import heom_squared
    E = _norm(space, space.candidate_pool)
    d2 = heom_squared(E[:, None, :], X[None, :, :], space=space)
    return float(np.sqrt(d2.min(axis=1)).max())


class TestMinimaxExact:
    @pytest.mark.parametrize("make", [
        _triangle,
        lambda: mergen.ParameterSpace({
            'x': ('continuous', 0, 1, {'resolution': 10}),
            'm': ('nominal', ['a', 'b', 'c']),
            'o': ('ordinal', ['lo', 'mid', 'hi'])}),
    ])
    def test_matches_brute_force(self, make):
        from mergen.metrics import minimax
        space = make()
        E = _norm(space, space.candidate_pool)
        rng = np.random.default_rng(0)
        for n in (1, 4, 12):
            X = E[rng.choice(len(E), n, replace=False)]
            assert np.isclose(minimax(X, space=space),
                              _brute_minimax(X, space), atol=1e-12)

    def test_design_holding_every_feasible_point_scores_zero(self):
        # 0.1.0 covered the unconstrained square and reported 0.71.
        from mergen.metrics import minimax
        space = _triangle()
        assert minimax(_norm(space, space.candidate_pool), space=space) == 0

    def test_nominal_levels_count(self):
        # A design that never visits a nominal level covers less.
        from mergen.metrics import minimax
        space = mergen.ParameterSpace({
            'x': ('continuous', 0, 1, {'resolution': 5}),
            'm': ('nominal', ['a', 'b', 'c'])})
        E = _norm(space, space.candidate_pool)
        one_level = E[E[:, 1] == E[0, 1]]
        assert minimax(one_level, space=space) > minimax(E, space=space)

    def test_cache_follows_new_constraint(self):
        from mergen.metrics import minimax
        space = mergen.ParameterSpace({'a': list(range(10)),
                                       'b': list(range(10))})
        X = _norm(space, [[0, 0], [0, 9], [9, 0]])
        before = minimax(X, space=space)
        space.add_constraint(lambda p: p['a'] + p['b'] <= 9)
        after = minimax(X, space=space)
        assert after < before
        assert np.isclose(after, _brute_minimax(X, space), atol=1e-12)

    def test_large_pool_uses_fixed_subset(self, monkeypatch):
        from mergen import metrics as M
        monkeypatch.setattr(M, '_MINIMAX_EXACT_MAX', 20)
        space = _triangle()
        E = _norm(space, space.candidate_pool)
        X = E[[0, len(E) - 1]]
        value = M.minimax(X, space=space)
        assert len(space._minimax_eval[1]) == 20
        assert value <= _brute_minimax(X, space)
        space._minimax_eval = None
        assert M.minimax(X, space=space) == value


class TestFeasibleBaseline:
    def test_draws_only_feasible_points_and_skips_exclusions(self, monkeypatch):
        from mergen import metrics as M
        space, drawn = _triangle(), []
        real = M._compute_metric

        def spy(name, X, space=None):
            drawn.append(X * space.granges + space.gmins)
            return real(name, X, space=space)

        monkeypatch.setattr(M, '_compute_metric', spy)
        M._mc_baseline(space.grid_sampler(), space.gmins, space.granges,
                       10, ['min_distance'], [], 100, seed=1, space=space,
                       exclude=[[0, 0], [9, 0]])
        P = np.vstack(drawn)
        assert np.all(P.sum(axis=1) <= 9)
        assert not np.any(np.all(P == [0, 0], axis=1))
        assert not np.any(np.all(P == [9, 0], axis=1))

    def test_unconstrained_pool_order_matches_grid_index(self):
        # Keeps the baseline of unconstrained spaces identical to 0.1.0:
        # drawing pool rows equals drawing grid indices.
        space = mergen.ParameterSpace({'a': [1, 2, 3], 'b': [10, 20],
                                       'c': [0.5, 1.5, 2.5, 3.5]})
        gs = space.grid_sampler()
        expected = np.array([gs.index_to_point(i)
                             for i in range(gs.n_candidates)])
        assert np.array_equal(space.candidate_pool, expected)

    def test_report_and_compare_pass_exclusions(self, monkeypatch):
        from mergen import metrics as M
        seen, real = [], M._mc_baseline

        def spy(*args, **kw):
            seen.append(kw.get('exclude'))
            return real(*args, **kw)

        monkeypatch.setattr(M, '_mc_baseline', spy)
        s = mergen.Sampler(mergen.ParameterSpace(
            {'a': list(range(8)), 'b': list(range(8))}))
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=100)
        s.add_exclusion([0, 0])
        r = s.run(n_repeats=1, criteria='cd2', seed=1, verbose=False)
        assert r._meta['exclusions'] == [[0.0, 0.0]]
        r.quality_report(mc_samples=5, verbose=False)
        s.compare(criteria=['cd2'], n_repeats=1, mc_samples=5,
                  verbose=False)
        assert len(seen) == 2
        assert all(np.allclose(np.asarray(e, float), [[0, 0]])
                   for e in seen)

    def test_comparison_passes_space(self, monkeypatch, capsys):
        from mergen import metrics as M
        spaces, real = [], M._compute_metric

        def spy(name, X, space=None):
            spaces.append(space)
            return real(name, X, space=space)

        monkeypatch.setattr(M, '_compute_metric', spy)
        s = mergen.Sampler(mergen.ParameterSpace(
            {'a': list(range(5)), 'b': list(range(5))}))
        s.set_design(n_samples=5, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=50)
        r = s.run(n_repeats=1, criteria='cd2', seed=1, verbose=False)
        M.comparison(r)
        assert spaces and all(sp is r.space for sp in spaces)


class TestPercentileRank:
    def test_counts_the_design_among_b_plus_one(self):
        from mergen.metrics import _percentile_rank
        base = np.array([1.0, 2.0, 3.0])
        # Beats all three: (3 + 0.5) / 4.
        assert _percentile_rank(4.0, base, True) == pytest.approx(87.5)
        # Beaten by all three: 0.5 / 4.
        assert _percentile_rank(0.0, base, True) == pytest.approx(12.5)
        # Ties one, beats one: (1 + 0.5 * 2) / 4.
        assert _percentile_rank(2.0, base, True) == pytest.approx(50.0)

    def test_nan_design_value_gives_nan(self):
        # 0.1.0 printed a NaN metric as the 0th percentile.
        from mergen.metrics import _percentile_rank
        assert np.isnan(_percentile_rank(np.nan, np.array([1.0]), True))


class TestReportUsesRunSettings:
    def _run(self, criterion, seed):
        space = mergen.ParameterSpace({'x': list(range(9)),
                                       'y': list(range(9))})
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        return s.run(n_repeats=1, criteria=criterion, algorithm='sa', seed=seed,
                     verbose=False)

    def test_criterion_instance_scores_the_report(self):
        # 0.1.0 looked the criterion up by name and lost its parameters;
        # an instance gave NaN.
        from mergen.criteria import PhiP
        r = self._run(PhiP(p=5), seed=1)
        rp = r.quality_report(mc_samples=20, verbose=False)
        key = next(k for k in rp if k.startswith('criterion_')
                   and not k.endswith(('_rank', '_median')))
        X = _norm(r.space, r.samples[r.space.names].to_numpy(float))
        assert rp[key] == pytest.approx(PhiP(p=5).evaluate(X, r.space))
        assert np.isfinite(rp[key + '_percentile_rank'])

    def test_baseline_seed_follows_the_run_seed(self, monkeypatch):
        from mergen import metrics as M
        seeds = []
        real = M._mc_baseline

        def spy(*a, **k):
            seeds.append(k.get('seed'))
            return real(*a, **k)

        monkeypatch.setattr(M, '_mc_baseline', spy)
        self._run('cd2', seed=7).quality_report(mc_samples=5, verbose=False)
        assert seeds == [7]


def test_mean_and_cv_distances_cannot_be_priority(num_space):
    s = mergen.Sampler(num_space)
    s.set_design(n_samples=6, n_validation=0)
    for m in ('mean_distance', 'cv_distances'):
        with pytest.raises(ValueError, match="cannot be a priority"):
            s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False,
                  n_repeats=2, priority=(m,))


class TestBaselineEdgeCases:
    def _space(self):
        return mergen.ParameterSpace({'x': [0.0, 1.0, 2.0]})

    def test_warns_when_the_pool_is_smaller_than_the_design(self):
        from mergen import metrics as M
        space = self._space()
        with pytest.warns(UserWarning, match="instead of 3"):
            M._mc_baseline(space.grid_sampler(), space.gmins, space.granges,
                           3, ['min_distance'], [], 4, space=space,
                           exclude=[[0.0]])

    def test_empty_pool_gives_nan(self):
        from mergen import metrics as M
        space = self._space()
        out = M._mc_baseline(space.grid_sampler(), space.gmins,
                             space.granges, 2, ['min_distance'], [], 4,
                             space=space, exclude=[[0.0], [1.0], [2.0]])
        assert np.isnan(out['min_distance']).all()


def test_baseline_skips_exclusions_at_tiny_scale():
    # An absolute 1e-9 tolerance matched every level of this factor
    # to the exclusion, leaving an empty pool.
    from mergen import metrics as M
    space = mergen.ParameterSpace({'x': [1e-12, 2e-12, 3e-12, 4e-12]})
    drawn = []
    real = M._compute_metric

    def spy(name, X, space=None):
        drawn.append(X)
        return real(name, X, space=space)

    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(M, '_compute_metric', spy)
        M._mc_baseline(space.grid_sampler(), space.gmins, space.granges, 2,
                       ['min_distance'], [], 10, space=space,
                       exclude=[[1e-12]])
    P = np.vstack(drawn)
    assert len(P) == 20 and not np.any(np.isclose(P, 0.0))


def test_cd2_metric_equals_the_criterion_and_can_be_a_priority():
    from mergen.metrics import cd2
    space = mergen.ParameterSpace({'x': [0, 1, 2, 3], 'y': [0, 1, 2]})
    X = np.array([[0.0, 0.0], [1 / 3, 1.0], [1.0, 0.5]])
    crit = mergen.criteria.get_criterion('cd2')
    assert cd2(X, space) == pytest.approx(crit.evaluate(X, space))
    s = mergen.Sampler(space)
    s.set_design(n_samples=4, n_validation=0)
    cmp = s.compare(criteria=['phi_p', 'cd2'], algorithms=['sce'],
                    priority=('minimax', 'cd2'), n_repeats=2,
                    mc_samples=20, verbose=False)
    assert 'cd2' in cmp.table.columns
    assert cmp.reselect(priority=('cd2',), verbose=False).best in cmp.results


@pytest.mark.parametrize('p, text', [(1, '1st'), (2, '2nd'), (3, '3rd'),
                                     (11, '11th'), (13, '13th'),
                                     (22, '22nd'), (72, '72nd'),
                                     (99.8, '100th'), (111, '111th')])
def test_percentile_ordinal_suffix(p, text):
    from mergen.metrics import _ordinal
    assert _ordinal(p) == text


@pytest.mark.parametrize('p, text', [(99.83, '99th'), (0.17, '1st'),
                                     (50.4, '50th')])
def test_percentile_rank_is_shown_within_1_and_99(p, text):
    from mergen.metrics import _pct_ordinal
    assert _pct_ordinal(p) == text


def test_report_percentile_column_is_within_1_and_99():
    from mergen.metrics import _fmt_pct
    assert '99th pct' in _fmt_pct(99.83) and '1st pct' in _fmt_pct(0.17)


def _fixed_sampler():
    space = mergen.ParameterSpace({'x': list(range(11)),
                                   'y': list(range(11))})
    s = mergen.Sampler(space)
    s.add_prescribed([[0, 0], [0, 1], [1, 0]])
    s.set_design(n_samples=12, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=200)
    return space, s


def test_baseline_without_fixed_points_is_unchanged():
    from mergen.metrics import _mc_baseline
    space = mergen.ParameterSpace({'x': list(range(11)),
                                   'y': list(range(11))})
    args = (space.grid_sampler(), space.gmins, space.granges, 8,
            ['min_distance'], [], 30)
    a = _mc_baseline(*args, seed=3, space=space)
    b = _mc_baseline(*args, seed=3, space=space, fixed=[])
    assert np.array_equal(a['min_distance'], b['min_distance'])


def test_baseline_completes_the_fixed_points():
    # The fixed points plus a random fill is a draw from the baseline
    # itself, so its mean rank is about 50.
    from mergen.metrics import _mc_baseline, _percentile_rank
    space, s = _fixed_sampler()
    r = s.run(criteria='phi_p', algorithm='sa', seed=1, n_repeats=1,
              verbose=False)
    F = np.asarray(r._meta['fixed_points'])
    assert len(F) == 3
    base = _mc_baseline(space.grid_sampler(), space.gmins, space.granges,
                        12, ['mean_distance'], [], 300, seed=5,
                        space=space, fixed=F)
    null = _mc_baseline(space.grid_sampler(), space.gmins, space.granges,
                        12, ['mean_distance'], [], 200, seed=6,
                        space=space, fixed=F)
    ranks = [_percentile_rank(v, base['mean_distance'], True)
             for v in null['mean_distance']]
    assert 40 < np.mean(ranks) < 60
    rep = r.quality_report(mc_samples=50, verbose=False)
    assert rep['baseline_fixed_points'] == 3


def test_compare_percentiles_match_the_report_with_fixed_points():
    space, s = _fixed_sampler()
    c = s.compare(criteria=['phi_p'], algorithms=['sa'], n_repeats=1,
                  mc_samples=50, verbose=False)
    row = c.table.set_index(['criterion', 'algorithm']).loc[c.best]
    rep = c.best_result.quality_report(mc_samples=50, verbose=False)
    assert rep['baseline_fixed_points'] == 3
    for m in ('min_distance', 'minimax', 'max_abs_correlation'):
        assert row[m] == pytest.approx(rep[f'{m}_percentile_rank'])


def test_extend_conditions_the_baseline_on_the_existing_rows():
    space, s = _fixed_sampler()
    pilot = s.run(criteria='phi_p', algorithm='sa', seed=1, n_repeats=1,
                  verbose=False)
    ext = mergen.sequential.extend(s, pilot.samples, n_new=4, seed=2,
                                   verbose=False, n_repeats=1)
    assert len(ext._meta['fixed_points']) == 12


def test_baseline_never_draws_a_fixed_point_again():
    from mergen.metrics import _mc_baseline
    space = mergen.ParameterSpace({'x': [0, 1], 'y': [0, 1, 2]})
    base = _mc_baseline(space.grid_sampler(), space.gmins, space.granges,
                        6, ['min_distance'], [], 20, seed=1, space=space,
                        fixed=[[0, 0], [1, 2], [0, 1]])
    assert np.allclose(base['min_distance'], 0.5)


def test_report_of_a_loaded_design_has_no_criterion_row():
    space = mergen.ParameterSpace({'x': list(range(5)), 'y': list(range(5))})
    s = mergen.Sampler(space)
    s.load_design([[0, 0], [4, 4], [2, 2], [0, 4]])
    rep = s.run(verbose=False).quality_report(mc_samples=10, verbose=False)
    assert not any(k.startswith('criterion_') for k in rep)
    assert rep['baseline_fixed_points'] == 0


def test_compare_tie_on_every_table_metric_goes_to_cd2():
    from mergen.metrics import cd2
    space = mergen.ParameterSpace({'x': list(range(9)), 'y': list(range(9))})
    s = mergen.Sampler(space)
    s.set_design(n_samples=8, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=100)
    prio = ('min_distance', 'minimax', 'max_abs_correlation',
            'projection_cd2')
    c = s.compare(criteria=['phi_p', 'stratified', 'umaxpro'],
                  algorithms=['sa'],
                  priority=prio, selection='lexicographic',
                  tol={m: 10.0 for m in prio}, n_repeats=1, mc_samples=0,
                  verbose=False)
    val = {k: cd2((np.asarray(r.design, float) - space.gmins)
                  / space.granges, space) for k, r in c.results.items()}
    assert c.best == min(val, key=val.get)


def test_compare_without_seed_matches_the_report():
    space, s = _fixed_sampler()
    c = s.compare(criteria=['phi_p'], algorithms=['sa'], n_repeats=1,
                  mc_samples=50, seed=None, verbose=False)
    row = c.table.set_index(['criterion', 'algorithm']).loc[c.best]
    rep = c.best_result.quality_report(mc_samples=50, verbose=False)
    for m in ('min_distance', 'minimax', 'mean_distance'):
        assert row[m] == pytest.approx(rep[f'{m}_percentile_rank'])


def test_baseline_leaves_out_points_the_optimiser_cannot_use():
    space = mergen.ParameterSpace({'x': [0, 1], 'y': [0, 1, 2]})
    s = mergen.Sampler(space)
    s.add_set('hand', [[0, 0], [1, 2]])
    s.set_design(n_samples=4, n_validation=0)
    r = s.run(criteria='cd2', algorithm='sa', seed=1, n_repeats=1,
              verbose=False)
    assert sorted(r._meta['baseline_reserved']) == [[0.0, 0.0], [1.0, 2.0]]
    # Four free nodes for four points: every baseline design is the
    # optimised design itself, so the baseline median equals its value.
    rep = r.quality_report(mc_samples=10, verbose=False)
    assert rep['minimax_baseline_median'] == pytest.approx(rep['minimax'])


def test_drop_rows_matches_a_row_by_row_scan():
    from mergen.metrics import _drop_rows
    lv = np.linspace(0, 1, 7)
    pool = np.array(np.meshgrid(lv, lv[:4], indexing='ij')).reshape(2, -1).T
    rows = np.vstack([pool[[3, 9, 9]], [[0.9, 0.4]], [[lv[2] + 1e-12, 0]]])
    keep = np.ones(len(pool), dtype=bool)
    for r in rows:
        keep &= ~np.all(np.abs(pool - r) <= 1e-9, axis=1)
    assert np.array_equal(_drop_rows(pool, rows, 1e-9), pool[keep])


@pytest.mark.parametrize('kind', ['set', 'focus'])
def test_compare_matches_the_report_with_reserved_points(kind):
    space = mergen.ParameterSpace({'x': list(range(7)), 'y': list(range(7))})
    s = mergen.Sampler(space)
    if kind == 'set':
        s.add_set('hand', [[0, 0], [6, 6], [3, 3]])
    else:
        s.add_focus([3, 3], spread=1.0, n_samples=2, include_center=False,
                    in_design=False)
    s.set_design(n_samples=6, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=100)
    c = s.compare(criteria=['phi_p'], algorithms=['sa'], n_repeats=1,
                  mc_samples=50, verbose=False)
    assert c.best_result._meta['baseline_reserved']
    row = c.table.set_index(['criterion', 'algorithm']).loc[c.best]
    rep = c.best_result.quality_report(mc_samples=50, verbose=False)
    for m in ('min_distance', 'minimax', 'mean_distance'):
        assert row[m] == pytest.approx(rep[f'{m}_percentile_rank'])


def test_quality_report_plain_text_off_terminal_and_checks_mc_samples(capsys):
    s = mergen.Sampler(mergen.ParameterSpace({'x': list(range(6)),
                                              'y': list(range(6))}))
    s.set_design(n_samples=5, n_validation=0)
    r = s.run(criteria='phi_p', n_repeats=1, seed=1, verbose=False)
    capsys.readouterr()
    r.quality_report(criteria_metrics=['cd2'], mc_samples=20)
    out = capsys.readouterr().out
    assert '\033[' not in out
    assert 'CD2 +' in out and '+ evaluated after the run' in out
    with pytest.raises(ValueError, match='mc_samples'):
        r.quality_report(mc_samples=-1, verbose=False)
