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
        r = s.run(criteria=crit, algorithm=alg, seed=1, verbose=False)
        assert np.isfinite(r.best_score)

    @pytest.mark.parametrize("alg", ['sa', 'sce', 'ese'])
    @pytest.mark.parametrize("crit", ['maxproqq', 'qqd'])
    def test_alg_crit_mixed(self, alg, crit, mixed_space):
        s = mergen.Sampler(mixed_space)
        s.set_design(n_samples=12, n_validation=0)
        _configure(s, alg)
        r = s.run(criteria=crit, algorithm=alg, seed=1, verbose=False)
        assert np.isfinite(r.best_score)


class TestSeedDeterminism:
    def test_sa_deterministic(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r1 = s.run(criteria='cd2', algorithm='sa', seed=7, verbose=False)
        r2 = s.run(criteria='cd2', algorithm='sa', seed=7, verbose=False)
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
        seq = s.run(criteria='cd2', algorithm=['sa', 'sce'],
                    seed=7, verbose=False, n_jobs=None)
        par = s.run(criteria='cd2', algorithm=['sa', 'sce'],
                    seed=7, verbose=False, n_jobs=2)
        assert np.isclose(seq.best_score, par.best_score)