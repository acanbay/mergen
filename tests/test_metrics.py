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
