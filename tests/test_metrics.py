"""Tests for mergen.metrics — quality metrics and quality_report."""

import numpy as np
import pandas as pd
import pytest

import mergen.metrics as metrics


def _norm(space, n=10):
    pool = space.candidate_pool
    norm = (pool - space.gmins) / space.granges
    return norm[:n]


class TestMetricFunctions:

    def test_min_distance_positive(self, simple_space):
        assert metrics.min_distance(_norm(simple_space)) > 0

    def test_min_distance_single_point(self):
        assert metrics.min_distance(np.array([[0.5, 0.5]])) == 0.0

    def test_mean_distance_positive(self, simple_space):
        assert metrics.mean_distance(_norm(simple_space)) > 0

    def test_cv_distances_nonneg(self, simple_space):
        assert metrics.cv_distances(_norm(simple_space)) >= 0

    def test_minimax_bounded(self, simple_space):
        v = metrics.minimax(_norm(simple_space))
        assert 0 < v <= np.sqrt(simple_space.n_parameters)

    def test_max_abs_correlation_bounded(self, simple_space):
        v = metrics.max_abs_correlation(_norm(simple_space))
        assert 0 <= v <= 1

    def test_max_abs_correlation_single_param(self):
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        assert metrics.max_abs_correlation(X) == 0.0

    def test_projection_cd2_nonneg(self, simple_space):
        assert metrics.projection_cd2(_norm(simple_space)) >= 0

    def test_projection_cd2_requires_at_least_2d(self):
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        assert metrics.projection_cd2(X) == 0.0

    def test_uniform_design_lower_cv(self, simple_space):
        """A more uniform design should have lower CV than a clustered one."""
        pool     = simple_space.candidate_pool
        norm     = (pool - simple_space.gmins) / simple_space.granges
        # Uniform: evenly spaced
        uniform  = norm[::4][:10]
        # Clustered: first 10 rows (all close together in grid order)
        clustered = norm[:10]
        cv_u = metrics.cv_distances(uniform)
        cv_c = metrics.cv_distances(clustered)
        assert cv_u <= cv_c


class TestQualityReport:

    def test_returns_dict(self, basic_result):
        stats = basic_result.quality_report(mc_samples=0)
        assert isinstance(stats, dict)

    def test_default_metrics_present(self, basic_result):
        stats = basic_result.quality_report(mc_samples=0)
        for m in metrics._DEFAULT_METRICS:
            assert m in stats

    def test_values_are_numeric(self, basic_result):
        stats = basic_result.quality_report(mc_samples=0)
        for m in metrics._DEFAULT_METRICS:
            assert isinstance(stats[m], float)
            assert not np.isnan(stats[m])

    def test_with_mc_adds_percentile_rank(self, basic_result):
        stats = basic_result.quality_report(mc_samples=50)
        assert 'min_distance_percentile_rank' in stats
        rank  = stats['min_distance_percentile_rank']
        assert 0 <= rank <= 100

    def test_criteria_metrics_added(self, basic_result):
        stats = basic_result.quality_report(
            criteria_metrics=['maxpro'], mc_samples=0)
        assert 'criterion_maxpro' in stats

    def test_criteria_metrics_with_mc(self, basic_result):
        stats = basic_result.quality_report(
            criteria_metrics=['maxpro'], mc_samples=50)
        assert 'criterion_maxpro_percentile_rank' in stats

    def test_selected_metrics_only(self, basic_result):
        stats = basic_result.quality_report(
            metrics=['min_distance', 'minimax'], mc_samples=0)
        assert 'min_distance' in stats
        assert 'minimax'      in stats
        assert 'cv_distances' not in stats

    def test_unknown_metric_raises(self, basic_result):
        with pytest.raises(ValueError):
            basic_result.quality_report(metrics=['nonexistent'])

    def test_run_criteria_auto_included(self, basic_result):
        """The criterion used in run() should always appear in the report."""
        stats = basic_result.quality_report(mc_samples=0)
        crit  = basic_result._meta.get('criteria', 'umaxpro')
        assert f'criterion_{crit}' in stats


class TestComparison:

    def test_returns_dataframe(self, basic_result):
        basic_result.designs['umaxpro'] = basic_result.samples
        df = basic_result.comparison()
        assert isinstance(df, pd.DataFrame)

    def test_default_metrics_as_columns(self, basic_result):
        basic_result.designs['umaxpro'] = basic_result.samples
        df = basic_result.comparison()
        for m in metrics._DEFAULT_METRICS:
            assert m in df.columns

    def test_index_is_criterion_name(self, basic_result):
        basic_result.designs['umaxpro'] = basic_result.samples
        df = basic_result.comparison()
        assert 'umaxpro' in df.index


class TestMergenInfo:

    def test_info_runs(self):
        import mergen
        mergen.info()

    def test_banner_contains_version(self):
        import mergen
        banner = mergen._banner()
        assert mergen.__version__ in banner

    def test_banner_contains_name(self):
        import mergen
        banner = mergen._banner()
        assert 'mergen' in banner.lower()

    def test_banner_hides_none_fields(self):
        import mergen
        original_docs = mergen.__docs__
        mergen.__docs__ = None
        banner = mergen._banner()
        assert 'Docs' not in banner
        mergen.__docs__ = original_docs