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
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        assert r is not None
        assert len(r.best_design) == 10
        assert np.isfinite(r.best_score)

    def test_reproducibility(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=500)
        r1 = s.run(criteria='cd2', algorithm='sa', seed=42, verbose=False)
        r2 = s.run(criteria='cd2', algorithm='sa', seed=42, verbose=False)
        assert np.isclose(r1.best_score, r2.best_score)


class TestMultiAlgorithm:
    @pytest.mark.parametrize("alg", ['sa', 'sce'])
    def test_sa_sce_run(self, alg, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer(alg, n_restarts=1, max_iter=300)
        r = s.run(criteria='cd2', algorithm=alg, seed=1, verbose=False)
        assert np.isfinite(r.best_score)

    def test_ese_runs(self, num_space):
        """ESE uses J/M/Q instead of max_iter."""
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('ese', n_restarts=1)
        r = s.run(criteria='cd2', algorithm='ese', seed=1, verbose=False)
        assert np.isfinite(r.best_score)

    def test_multi_algorithm_list(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(criteria='cd2', algorithm=['sa', 'sce'],
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
            s.run(criteria=bad_crit, algorithm='sa', seed=1, verbose=False)

    @pytest.mark.parametrize("good_crit", ['maxproqq', 'qqd'])
    def test_nominal_accepts_compatible(self, good_crit, mixed_space):
        s = mergen.Sampler(mixed_space)
        s.set_design(n_samples=12, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=500)
        r = s.run(criteria=good_crit, algorithm='sa', seed=1, verbose=False)
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
            r = s.run(criteria=c, algorithm='sa', seed=1, verbose=False)
            assert np.isfinite(r.best_score), c


class TestBackwardCompatibility:
    """Numerical-only workflows must reproduce previous scores exactly."""

    def test_cd2_deterministic(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=15, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=1500)
        r = s.run(criteria='cd2', algorithm='sa', seed=44, verbose=False)
        assert np.isclose(r.best_score, 0.056862, rtol=1e-4)


class TestParallel:
    @pytest.mark.skipif(
        __import__('os').cpu_count() < 2,
        reason="requires >= 2 CPUs",
    )
    def test_multi_alg_parallel(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(criteria='cd2', algorithm=['sa', 'sce'],
                  seed=1, verbose=False, n_jobs=2)
        assert np.isfinite(r.best_score)

    def test_invalid_n_jobs_zero(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        with pytest.raises((ValueError, SystemExit)):
            s.run(criteria='cd2', algorithm=['sa', 'sce'],
                  seed=1, verbose=False, n_jobs=0)