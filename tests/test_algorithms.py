"""Tests for optimisation algorithms (via Sampler)."""
from __future__ import annotations

import numpy as np
import pytest

import mergen


def _configure(sampler, alg, n_restarts=1):
    """ESE uses J/M/Q, SA/SCE use max_iter."""
    if alg == 'ese':
        sampler.set_optimizer(alg, n_restarts=n_restarts)
    else:
        sampler.set_optimizer(alg, n_restarts=n_restarts, max_iter=300)


class TestAlgorithmSmoke:
    @pytest.mark.parametrize("alg", ['sa', 'sce', 'ese'])
    @pytest.mark.parametrize("crit",
                             ['cd2', 'maxpro', 'umaxpro', 'phi_p', 'stratified'])
    def test_alg_crit_numerical(self, alg, crit, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        _configure(s, alg)
        r = s.run(n_repeats=1, criteria=crit, algorithm=alg, seed=1, verbose=False)
        assert np.isfinite(r.best_score)

    @pytest.mark.parametrize("alg", ['sa', 'sce', 'ese'])
    @pytest.mark.parametrize("crit", ['maxproqq', 'qqd'])
    def test_alg_crit_mixed(self, alg, crit, mixed_space):
        s = mergen.Sampler(mixed_space)
        s.set_design(n_samples=12, n_validation=0)
        _configure(s, alg)
        r = s.run(n_repeats=1, criteria=crit, algorithm=alg, seed=1, verbose=False)
        assert np.isfinite(r.best_score)


class TestSeedDeterminism:
    def test_sa_deterministic(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r1 = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=7, verbose=False)
        r2 = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=7, verbose=False)
        assert np.isclose(r1.best_score, r2.best_score)

    @pytest.mark.skipif(
        __import__('os').cpu_count() < 2,
        reason="requires >= 2 CPUs",
    )
    def test_sce_parallel_matches_sequential(self, num_space):
        """SCE with joblib -> same score as single-thread."""
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        seq = s.run(n_repeats=1, criteria='cd2', algorithm=['sa', 'sce'],
                    seed=7, verbose=False, n_jobs=None)
        par = s.run(n_repeats=1, criteria='cd2', algorithm=['sa', 'sce'],
                    seed=7, verbose=False, n_jobs=2)
        assert np.isclose(seq.best_score, par.best_score)


class TestESE1DGuard:
    """ESE must reject one-dimensional spaces with an actionable error.

    ESE's within-column swap (Jin, Chen & Sudjianto 2005) permutes
    rows without changing a 1D design as a point set, so running it
    would silently return the initial design. Sampler.run therefore
    raises immediately and points the user to 'sa' or 'sce'.
    """

    def test_ese_rejects_1d_space(self):
        space = mergen.ParameterSpace(
            {'x': np.arange(0.0, 1.01, 0.1).tolist()})
        s = mergen.Sampler(space)
        s.set_design(n_samples=3)
        with pytest.raises(ValueError, match="1-parameter"):
            s.run(n_repeats=1, criteria='phi_p', algorithm='ese', verbose=False)

    def test_sa_and_sce_still_accept_1d(self):
        space = mergen.ParameterSpace(
            {'x': np.arange(0.0, 1.01, 0.1).tolist()})
        for alg in ('sa', 'sce'):
            s = mergen.Sampler(space)
            s.set_design(n_samples=3)
            res = s.run(n_repeats=1, criteria='phi_p', algorithm=alg,
                        seed=44, verbose=False)
            assert len(res.samples) == 3


class TestDeliveredPointsAreAllowed:
    """0.1.0 could deliver an infeasible seed corner or an exclusion centre."""

    def _sampler(self, alg, exclusions):
        space = mergen.ParameterSpace({'a': list(range(8)),
                                       'b': list(range(8))})
        space.add_constraint(lambda p: p['a'] >= p['b'])
        s = mergen.Sampler(space)
        s.set_design(n_samples=10, n_validation=0)
        for e in exclusions:
            s.add_exclusion(e)
        _configure(s, alg)
        return s

    @pytest.mark.parametrize("alg", ['sa', 'sce', 'ese'])
    @pytest.mark.parametrize("seed", [1, 44])
    def test_no_infeasible_point_or_exclusion_centre(self, alg, seed):
        excl = [[4, 0], [2, 2]]
        s = self._sampler(alg, excl)
        r = s.run(n_repeats=1, criteria='cd2', algorithm=alg, seed=seed, verbose=False)
        P = r.samples[['a', 'b']].to_numpy(float)
        assert np.all(P[:, 0] >= P[:, 1])
        for e in excl:
            assert not np.any(np.all(P == e, axis=1))

    def test_swap_rejects_banned_point(self):
        from mergen.algorithms.sa import SAOptimizer
        space = mergen.ParameterSpace({'a': [0, 1], 'b': [0, 1]})
        crit  = mergen.criteria.get_criterion('cd2')
        design = np.array([[0.0, 0.0], [1.0, 1.0]])
        X = design.copy()
        score = crit.evaluate(X, space)
        # Every swap of these two rows creates [0, 1] or [1, 0].
        banned = np.array([[0.0, 1.0]])
        rng = np.random.default_rng(0)
        for _ in range(50):
            ok, _, _ = SAOptimizer()._try_swap(
                design, X, 0, 0, score, space, crit, None, space.names,
                1e9, rng, banned)
            assert not ok
        assert np.array_equal(design, [[0.0, 0.0], [1.0, 1.0]])


class TestRandomDraw:
    def _space(self):
        space = mergen.ParameterSpace({'a': list(range(5)),
                                       'b': list(range(5))})
        space.add_constraint(lambda p: p['a'] >= p['b'])
        return space

    def test_returns_none_quickly_when_nothing_is_free(self):
        import time
        space = self._space()
        gs = space.grid_sampler()
        t = time.perf_counter()
        for _ in range(20):
            pt, idx = gs.random_point_excluding(
                set(gs.pool_indices().tolist()),
                rng=np.random.default_rng(0))
            assert pt is None and idx is None
        # 0.1.0 spent max_tries (100,000) draws on every call.
        assert time.perf_counter() - t < 2.0

    def test_finds_the_last_free_point(self):
        space = self._space()
        gs = space.grid_sampler()
        free = gs.pool_indices().tolist()
        last = free.pop(7)
        pt, idx = gs.random_point_excluding(
            set(free), rng=np.random.default_rng(0))
        assert idx == last
        assert np.array_equal(pt, gs.index_to_point(last))


class TestSCERestartsUseTheSeed:
    def test_independent_of_global_random_state(self, num_space):
        # 0.1.0 drew the restart kicks from the global ``random`` module,
        # so the result depended on n_jobs.
        import random
        from mergen.algorithms import get_optimizer
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        crit = mergen.criteria.get_criterion('cd2')
        gs = num_space.grid_sampler()
        init, _ = gs.balanced_lhs_seed(
            selected=np.empty((0, 2)), budget=10, reserved=set(),
            rng=np.random.default_rng(0))
        out = []
        for g in (1, 2):
            random.seed(g)
            opt = get_optimizer('sce')(n_restarts=3, max_iter=100)
            out.append(opt.optimize(init.copy(), num_space, crit, seed=5,
                                    verbose=False).design)
        assert np.array_equal(out[0], out[1])


def test_ese_swap_rejects_banned_point():
    from mergen.algorithms.ese import ESEOptimizer
    space = mergen.ParameterSpace({'a': [0, 1], 'b': [0, 1]})
    crit = mergen.criteria.get_criterion('cd2')
    design = np.array([[0.0, 0.0], [1.0, 1.0]])
    # Every swap of these two rows creates [0, 1] or [1, 0].
    banned = np.array([[0.0, 1.0]])
    out = ESEOptimizer(M=20, J=10)._run_one(
        design, set(), space.grid_sampler(), space.gmins, space.granges,
        space, crit, 0, 0, 0, 0, False, banned)
    assert np.array_equal(out[0], design)
    restart, _ = ESEOptimizer()._perturb_restart(
        design, set(), space.grid_sampler(), 0, space, 0, 1, banned)
    assert np.array_equal(restart, design)


def test_no_free_point_for_the_seed_corner_is_an_error():
    space = mergen.ParameterSpace({'a': [0, 1, 2], 'b': [0, 1]})
    s = mergen.Sampler(space)
    s.add_set('all', space.candidate_pool.tolist())
    s.set_design(n_samples=1, n_validation=0)
    with pytest.raises(ValueError, match="No free feasible grid point"):
        s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=1, verbose=False)


def test_swap_that_leaves_the_design_unchanged_is_rejected():
    # The two rows differ only on axis 1, so every swap either exchanges
    # equal values or turns each row into the other one.
    from mergen.algorithms.ese import ESEOptimizer
    from mergen.algorithms.sa import SAOptimizer
    space = mergen.ParameterSpace({'a': [0, 1], 'b': [0, 1]})
    crit = mergen.criteria.get_criterion('phi_p')
    design = np.array([[0.0, 0.0], [0.0, 1.0]])
    X = (design - space.gmins) / space.granges
    score = crit.evaluate(X, space)
    sa, rng = SAOptimizer(), np.random.default_rng(0)
    for _ in range(50):
        accepted, new_score, _ = sa._try_swap(
            design, X.copy(), 0, 0, score, space, crit, [], space.names,
            1.0, rng)
        assert not accepted and new_score == score
    out = ESEOptimizer(M=20, J=10)._run_one(
        design, set(), space.grid_sampler(), space.gmins, space.granges,
        space, crit, 0, 0, 0, 0, False)
    assert out[4] == 0
    assert out[2] == pytest.approx(score)



class TestRoundFixes:
    """Hyperparameter checks, frozen designs, ESE candidates, LHS start."""

    @pytest.mark.parametrize('alg, kw', [
        ('sa', dict(max_iter=0)), ('sa', dict(T_start=-1.0)),
        ('sa', dict(cooling=1.5)), ('sa', dict(T_end_ratio=0.0)),
        ('sa', dict(target_accept=1.0)), ('sa', dict(perturbation_size=0)),
        ('sce', dict(max_iter=0))])
    def test_invalid_hyperparameters_are_refused(self, alg, kw):
        from mergen.algorithms import get_optimizer
        from mergen.criteria import PhiP
        space = mergen.ParameterSpace({'x': [0, 1, 2, 3], 'y': [0, 1, 2, 3]})
        D = np.array([[0., 0.], [1., 2.], [3., 1.], [2., 3.]])
        with pytest.raises(ValueError, match=list(kw)[0]):
            get_optimizer(alg)(**kw).optimize(D, space, PhiP(),
                                              verbose=False)

    @pytest.mark.parametrize('alg', ['sa', 'sce', 'ese'])
    def test_all_rows_frozen_reports_the_real_score(self, alg):
        from mergen.algorithms import get_optimizer
        from mergen.criteria import PhiP
        space = mergen.ParameterSpace({'x': [0, 1, 2, 3], 'y': [0, 1, 2, 3]})
        D = np.array([[0., 0.], [1., 2.], [3., 1.], [2., 3.]])
        r = get_optimizer(alg)().optimize(D, space, PhiP(), n_frozen=4,
                                          verbose=False)
        assert np.isclose(r.score, PhiP().evaluate(D / 3.0, space))

    def test_ese_scores_j_exchanges_that_change_the_design(self):
        from mergen.algorithms.ese import ESEOptimizer
        from mergen.criteria import CD2

        class Counting(CD2):
            calls = 0

            def incremental(self, *a, **k):
                Counting.calls += 1
                return super().incremental(*a, **k)

        # Column x is mostly one level, so most random draws in it swap
        # equal values and must not count.
        space = mergen.ParameterSpace({'x': [0, 1, 2], 'y': list(range(10))})
        D = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                      [1, 6], [2, 7]], dtype=float)
        ESEOptimizer(n_restarts=1, M=1, J=5, Q=1).optimize(
            D, space, Counting(), verbose=False)
        # Two single-row updates score one exchange.
        assert Counting.calls == 2 * 5

    def test_lhs_start_covers_every_stratum(self):
        # 20 points on 100-level axes, with a corner anchor at the top
        # of axis c: every one of the 20 strata must hold a level.
        space = mergen.ParameterSpace(
            {k: ('continuous', 0.0, 1.0) for k in 'abc'})
        gs = space.grid_sampler()
        anchor = np.array([[0.0, 0.0, 1.0]])
        for seed in range(20):
            D, _ = gs.balanced_lhs_seed(anchor, 19, set(),
                                        rng=np.random.default_rng(seed))
            for v in range(3):
                levels = [gs.level_of(v, x) for x in D[:, v]]
                assert len({lv * 20 // 100 for lv in levels}) == 20


class TestSATemperatureProbe:
    """The Ben-Ameur probe scores moves the chain itself could make."""

    def _setup(self):
        from mergen.criteria import PhiP
        space = mergen.ParameterSpace({'x': list(range(9)),
                                       'y': list(range(16))})
        gs = space.grid_sampler()
        rng = np.random.default_rng(0)
        idx = rng.choice(gs.n_candidates, 12, replace=False)
        design = np.array([gs.index_to_point(int(i)) for i in idx])
        banned = {int(i) for i in rng.choice(
            np.setdiff1d(np.arange(gs.n_candidates), idx), 30,
            replace=False)}
        reserved = banned | {int(i) for i in idx}
        calls = []

        class Spy(PhiP):
            def incremental(self, X, i, new_pt, space, current_score):
                calls.append((i, np.array(new_pt, dtype=float)))
                return super().incremental(X, i, new_pt, space,
                                           current_score)

        return space, gs, design, banned, reserved, Spy(), calls

    def test_probe_moves_only_free_nodes_and_movable_rows(self):
        from mergen.algorithms.sa import SAOptimizer, banned_points
        space, gs, design, banned, reserved, crit, calls = self._setup()
        n_frozen = 3
        X = (design - space.gmins) / space.granges
        score = crit.evaluate(X, space)
        T = SAOptimizer()._auto_temp(
            design, X, set(reserved), n_frozen, 0, score, gs, space.gmins,
            space.granges, crit, space, 1, 0,
            banned_points(gs, banned, design))
        assert T > 0 and calls
        for i, new in calls:
            assert i >= n_frozen
            node = gs.point_to_index(new * space.granges + space.gmins)
            assert node not in banned

    def test_given_T_start_skips_the_probe(self, monkeypatch):
        from mergen.algorithms.sa import SAOptimizer

        def boom(*a, **k):
            raise AssertionError("probe ran although T_start was given")

        monkeypatch.setattr(SAOptimizer, '_auto_temp', boom)
        space = mergen.ParameterSpace({'x': list(range(9)),
                                       'y': list(range(16))})
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200, T_start=0.05)
        s.run(criteria='phi_p', seed=1, n_repeats=1, n_jobs=1,
              verbose=False)
