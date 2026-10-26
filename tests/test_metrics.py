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
    return s.run(criteria='cd2', algorithm='sa', seed=44, verbose=False)


@pytest.fixture
def mixed_result(mixed_space):
    s = mergen.Sampler(mixed_space)
    s.set_design(n_samples=12, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=500)
    return s.run(criteria='maxproqq', algorithm='sa', seed=1, verbose=False)


class TestMetricFunctionsBackwardCompat:
    def test_min_distance_no_space(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
        assert np.isclose(min_distance(X), np.sqrt(2))

    def test_locked_values_numerical(self, num_result):
        rp = num_result.quality_report(verbose=False)
        # locked pre-categorical values
        assert np.isclose(rp['min_distance'], 0.141421, atol=1e-4)
        assert np.isclose(rp['projection_cd2'], 0.059917, atol=1e-4)


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
        r = s.run(criteria='cd2', seed=1, verbose=False)
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
        r = s.run(criteria='cd2', seed=1, verbose=False)
        M.comparison(r)
        assert spaces and all(sp is r.space for sp in spaces)
