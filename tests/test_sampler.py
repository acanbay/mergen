"""Tests for mergen.sampler.Sampler."""
from __future__ import annotations

import numpy as np
import pytest

import mergen


class TestBasicRun:
    def test_run_returns_result(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=500)
        r = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=1, verbose=False)
        assert r is not None
        assert len(r.best_design) == 10
        assert np.isfinite(r.best_score)

    def test_reproducibility(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=500)
        r1 = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=42, verbose=False)
        r2 = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=42, verbose=False)
        assert np.isclose(r1.best_score, r2.best_score)


class TestMultiAlgorithm:
    @pytest.mark.parametrize("alg", ['sa', 'sce'])
    def test_sa_sce_run(self, alg, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer(alg, n_restarts=1, max_iter=300)
        r = s.run(n_repeats=1, criteria='cd2', algorithm=alg, seed=1, verbose=False)
        assert np.isfinite(r.best_score)

    def test_ese_runs(self, num_space):
        """ESE uses J/M/Q instead of max_iter."""
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('ese', n_restarts=1)
        r = s.run(n_repeats=1, criteria='cd2', algorithm='ese', seed=1, verbose=False)
        assert np.isfinite(r.best_score)

    def test_multi_algorithm_list(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(n_repeats=1, criteria='cd2', algorithm=['sa', 'sce'],
                  seed=1, verbose=False)
        assert np.isfinite(r.best_score)


class TestCategoricalValidation:
    @pytest.mark.parametrize("bad_crit",
                             ['cd2', 'maxpro', 'umaxpro', 'phi_p', 'stratified'])
    def test_nominal_refuses_incompatible(self, bad_crit, mixed_space):
        s = mergen.Sampler(mixed_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        with pytest.raises((ValueError, SystemExit)):
            s.run(n_repeats=1, criteria=bad_crit, algorithm='sa', seed=1, verbose=False)

    @pytest.mark.parametrize("good_crit", ['maxproqq', 'qqd'])
    def test_nominal_accepts_compatible(self, good_crit, mixed_space):
        s = mergen.Sampler(mixed_space)
        s.set_design(n_samples=12, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=500)
        r = s.run(n_repeats=1, criteria=good_crit, algorithm='sa',
                  seed=1, verbose=False)
        assert np.isfinite(r.best_score)

    def test_ordinal_accepted_by_all(self):
        """Ordinal is treated numerically -> every criterion works."""
        space = mergen.ParameterSpace({
            'x': ('continuous', 0.0, 1.0),
            'q': ('ordinal', ['lo', 'mid', 'hi']),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        for c in ['cd2', 'maxpro', 'maxproqq', 'qqd', 'stratified']:
            r = s.run(n_repeats=1, criteria=c, algorithm='sa', seed=1, verbose=False)
            assert np.isfinite(r.best_score), c


class TestBackwardCompatibility:
    """Numerical-only workflows must reproduce previous scores exactly."""

    def test_cd2_deterministic(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=15, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=1500)
        r = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=44, verbose=False)
        assert np.isclose(r.best_score, 0.059619, rtol=1e-4)


class TestParallel:
    @pytest.mark.skipif(
        __import__('os').cpu_count() < 2,
        reason="requires >= 2 CPUs",
    )
    def test_multi_alg_parallel(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(n_repeats=1, criteria='cd2', algorithm=['sa', 'sce'],
                  seed=1, verbose=False, n_jobs=2)
        assert np.isfinite(r.best_score)

    def test_invalid_n_jobs_zero(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        with pytest.raises((ValueError, SystemExit)):
            s.run(n_repeats=1, criteria='cd2', algorithm=['sa', 'sce'],
                  seed=1, verbose=False, n_jobs=0)


class TestHoldoutSets:
    """Validation and extra sets: Kennard-Stone on the feasible pool."""

    def _sampler(self):
        space = mergen.ParameterSpace({'x': list(range(11)),
                                       'y': list(range(11))})
        space.add_constraint(lambda p: p['x'] + p['y'] <= 14)
        s = mergen.Sampler(space)
        s.add_prescribed([[10, 2]], in_design=False)
        s.add_set('ext', [[0, 10]])
        s.set_design(n_samples=8, n_validation=4, extra_sets={'test': 3})
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        return s

    def test_sets_are_the_maximin_sets_of_the_feasible_pool(self):
        from mergen.distances import maximin_subset
        s   = self._sampler()
        res = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=3, verbose=False)
        space = s.space
        pool  = space.candidate_pool
        E     = (pool - space.gmins) / space.granges
        names = space.names

        def rows(df):
            return df[names].to_numpy(float)

        # Reference: every delivered point, including the out-of-budget
        # prescribed point and the user set.
        ref  = np.vstack([rows(res.samples), rows(res.sets['ext'])])
        free = ~np.array([any(np.array_equal(p, r) for r in ref)
                          for p in pool])
        R    = (ref - space.gmins) / space.granges
        # Validation first, then the extra set kept away from it too.
        val  = maximin_subset(E, 4, ref=R, available=free, space=space)
        free[val] = False
        test = maximin_subset(E, 3, ref=np.vstack([R, E[val]]),
                              available=free, space=space)
        assert np.array_equal(rows(res.validation), pool[val])
        assert np.array_equal(rows(res.sets['test']), pool[test])
        assert np.all(pool[np.r_[val, test]].sum(axis=1) <= 14)

    def test_independent_of_global_random_state(self):
        # 0.1.0 sampled candidates with the global ``random`` module on
        # grids above 500,000 points.
        import random
        space = mergen.ParameterSpace({'x': list(range(800)),
                                       'y': list(range(700))})
        s, gs = mergen.Sampler(space), space.grid_sampler()
        ref = np.array([[0.0, 0.0], [799.0, 699.0]])
        out = []
        for seed in (1, 2):
            random.seed(seed)
            out.append(s._kennard_stone(gs, set(), ref, 3))
        assert np.array_equal(out[0], out[1])
        assert np.array_equal(out[0][0], [0.0, 699.0])

    def test_moved_seed_corner_is_not_reserved(self, monkeypatch):
        # 0.1.0 kept the seed corner reserved after the optimiser had
        # moved it, so holdout sets could never use that node.
        space = mergen.ParameterSpace({'x': list(range(6)),
                                       'y': list(range(6))})
        s = mergen.Sampler(space)
        s.set_design(n_samples=6, n_validation=3)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        seen = []
        real = mergen.Sampler._kennard_stone

        def spy(self, gs, reserved, ref, n):
            seen.append(set(reserved))
            return real(self, gs, reserved, ref, n)

        monkeypatch.setattr(mergen.Sampler, '_kennard_stone', spy)
        r = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=2, verbose=False)
        corner = space.grid_sampler().point_to_index([0, 5])
        assert [0.0, 5.0] not in r.samples[['x', 'y']].to_numpy().tolist()
        assert corner not in seen[0]


class TestLabelsAndFocus:
    def _run(self):
        space = mergen.ParameterSpace({'x': list(range(11)),
                                       'y': list(range(11))})
        space.add_constraint(lambda p: p['x'] + p['y'] <= 14)
        s = mergen.Sampler(space)
        s.add_prescribed([[0, 0]], in_design=True, in_optim=False)
        s.add_prescribed([[3, 3]], in_design=True, in_optim=True)
        s.add_focus([9, 1], spread=3.0, n_samples=3, in_design=True,
                    in_optim=False)
        s.add_focus([2, 9], spread=3.0, n_samples=2, in_design=True,
                    in_optim=True)
        s.add_exclusion([8, 2])
        s.set_design(n_samples=12, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        return s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=1, verbose=False)

    def test_prescribed_rows_are_labelled_prescribed(self):
        # 0.1.0 swapped Prescribed and Focus labels when in_optim was
        # mixed between them.
        df = self._run().samples
        pts = df[['x', 'y']].to_numpy().tolist()
        types = df['point_type'].tolist()
        assert types[pts.index([0.0, 0.0])] == 'Prescribed'
        assert types[pts.index([3.0, 3.0])] == 'Prescribed'
        assert types.count('Prescribed') == 2
        assert types.count('Focus') == 5

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_focus_points_are_feasible_and_not_excluded(self, seed):
        # 0.1.0 drew focus neighbours from the whole grid, ignoring
        # constraints and exclusion centres.
        space = mergen.ParameterSpace({'x': list(range(11)),
                                       'y': list(range(11))})
        space.add_constraint(lambda p: p['x'] + p['y'] <= 14)
        s = mergen.Sampler(space)
        s.add_focus([7, 7], spread=2.0, n_samples=8)
        s.add_exclusion([7, 6])
        s.add_exclusion([6, 7])
        s.set_design(n_samples=12, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        df = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=seed,
                   verbose=False).samples
        P = df.loc[df.point_type == 'Focus', ['x', 'y']].to_numpy()
        assert len(P) == 8
        assert np.all(P.sum(axis=1) <= 14)
        for e in ([7, 6], [6, 7]):
            assert not np.any(np.all(P == e, axis=1))


def test_set_dimension_weights_warns_that_it_has_no_effect(num_space):
    s = mergen.Sampler(num_space)
    with pytest.warns(UserWarning, match="no effect"):
        s.set_dimension_weights([2.0, 1.0])
