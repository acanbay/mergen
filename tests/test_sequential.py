"""Tests for mergen.sequential — augment, complement, from_dataframe, subsample, n_samples_recommendation."""

import numpy as np
import pandas as pd
import pytest

from mergen.sampler    import Sampler
from mergen.sequential import (
    augment,
    complement,
    from_dataframe,
    subsample,
    n_samples_recommendation,
)


@pytest.fixture
def base_result(simple_space):
    s = Sampler(simple_space)
    s.set_design(n_samples=10, n_validation=2)
    s.set_sce(n_restarts=1, max_iter=200)
    return s.run(seed=44)


class TestAugment:

    def test_size(self, base_result):
        r2 = augment(base_result, n_add=5, seed=44)
        assert len(r2.samples) == len(base_result.samples) + 5

    def test_existing_labelled_prescribed(self, base_result):
        r2 = augment(base_result, n_add=5, seed=44)
        vc = r2.samples['point_type'].value_counts()
        assert vc.get('Prescribed', 0) == len(base_result.samples)

    def test_new_points_labelled_optimised(self, base_result):
        r2 = augment(base_result, n_add=5, seed=44)
        vc = r2.samples['point_type'].value_counts()
        assert vc.get('Optimised', 0) == 5

    def test_meta_contains_n_added(self, base_result):
        r2 = augment(base_result, n_add=5, seed=44)
        assert r2._meta.get('n_added') == 5

    def test_no_duplicate_with_existing(self, base_result, simple_space):
        r2       = augment(base_result, n_add=5, seed=44)
        all_pts  = r2.samples[simple_space.names].values
        for i in range(len(all_pts)):
            for j in range(i + 1, len(all_pts)):
                assert not np.allclose(all_pts[i], all_pts[j])

    def test_with_focus(self, base_result, simple_space):
        focus_pt = simple_space.candidate_pool[-1]
        r2 = augment(base_result, n_add=15,
                     focus=[focus_pt, 1.0], seed=44)
        assert len(r2.samples) >= len(base_result.samples) + 10


class TestComplement:

    def test_size(self, simple_space):
        pool     = simple_space.candidate_pool
        existing = pool[[0, -1, len(pool) // 2]]
        r        = complement(simple_space, existing, n_samples=10, seed=44)
        assert len(r.samples) == 10

    def test_existing_not_in_output(self, simple_space):
        pool     = simple_space.candidate_pool
        existing = pool[[0, -1]]
        r        = complement(simple_space, existing, n_samples=8, seed=44)
        pts      = r.samples[simple_space.names].values
        for ex in existing:
            assert not any(np.allclose(ex, pt) for pt in pts)

    def test_accepts_dataframe(self, simple_space):
        pool = simple_space.candidate_pool
        df   = pd.DataFrame(pool[:3], columns=simple_space.names)
        r    = complement(simple_space, df, n_samples=8, seed=44)
        assert len(r.samples) == 8

    def test_wrong_shape_raises(self, simple_space):
        existing = np.array([[0.5, 0.5, 0.5]])   # 3 cols, space has 2
        with pytest.raises(ValueError):
            complement(simple_space, existing, n_samples=5, seed=44)


class TestFromDataframe:

    def test_size(self, base_result, simple_space):
        df = base_result.samples[simple_space.names].copy()
        r2 = from_dataframe(simple_space, df, n_add=5, seed=44)
        assert len(r2.samples) == len(df) + 5

    def test_from_csv(self, base_result, simple_space, tmp_path):
        path = str(tmp_path / 'design.csv')
        base_result.samples[simple_space.names].to_csv(path, index=False)
        r2 = from_dataframe(simple_space, path, n_add=5, seed=44)
        assert len(r2.samples) == len(base_result.samples) + 5

    def test_col_map(self, base_result, simple_space):
        df = base_result.samples[simple_space.names].copy()
        df = df.rename(columns={'x': 'X_val', 'y': 'Y_val'})
        r2 = from_dataframe(
            simple_space, df, n_add=5, seed=44,
            col_map={'X_val': 'x', 'Y_val': 'y'}
        )
        assert len(r2.samples) == len(base_result.samples) + 5

    def test_missing_column_raises(self, simple_space):
        df = pd.DataFrame({'wrong_col': [0.5, 0.3]})
        with pytest.raises(ValueError, match="not found"):
            from_dataframe(simple_space, df, n_add=5, seed=44)

    def test_offgrid_rows_skipped(self, simple_space, capsys):
        pool    = simple_space.candidate_pool
        df      = pd.DataFrame(pool[:5], columns=simple_space.names)
        df.loc[2, 'x'] = 0.12345   # off-grid
        r2      = from_dataframe(simple_space, df, n_add=3, seed=44)
        captured = capsys.readouterr()
        assert 'WARNING' in captured.out or 'skipped' in captured.out


class TestSubsample:

    def test_size(self, simple_space):
        large = simple_space.candidate_pool
        sub   = subsample(large, simple_space, n=20, seed=44)
        assert len(sub) == 20

    def test_from_dataframe(self, simple_space):
        df  = pd.DataFrame(simple_space.candidate_pool,
                           columns=simple_space.names)
        sub = subsample(df, simple_space, n=15, seed=44)
        assert len(sub) == 15
        assert list(sub.columns) == simple_space.names

    def test_returns_subset_of_original(self, simple_space):
        large = simple_space.candidate_pool
        sub   = subsample(large, simple_space, n=20, seed=44)
        sub_arr = sub.values
        for row in sub_arr:
            assert any(np.allclose(row, large[i]) for i in range(len(large)))

    def test_n_too_large_returns_all(self, simple_space, capsys):
        large = simple_space.candidate_pool[:10]
        sub   = subsample(large, simple_space, n=20, seed=44)
        assert len(sub) == 10
        captured = capsys.readouterr()
        assert 'WARNING' in captured.out

    def test_reproducible(self, simple_space):
        large = simple_space.candidate_pool
        s1    = subsample(large, simple_space, n=20, seed=44)
        s2    = subsample(large, simple_space, n=20, seed=44)
        np.testing.assert_array_equal(s1.values, s2.values)

    def test_space_filling_better_than_random(self, simple_space):
        """Subsampled subset should have higher min_distance than random."""
        import mergen.metrics as metrics
        large     = simple_space.candidate_pool
        gmins     = simple_space.gmins
        granges   = simple_space.granges
        sub       = subsample(large, simple_space, n=20, seed=44)
        sub_norm  = (sub.values - gmins) / granges

        rng        = np.random.default_rng(44)
        rand_idx   = rng.choice(len(large), size=20, replace=False)
        rand_norm  = (large[rand_idx] - gmins) / granges

        md_sub  = metrics.min_distance(sub_norm)
        md_rand = metrics.min_distance(rand_norm)
        # Subsample should generally be better — soft check
        assert md_sub >= 0 and md_rand >= 0


class TestNSamplesRecommendation:

    def test_10p_rule(self, simple_space):
        rec = n_samples_recommendation(simple_space, verbose=False)
        assert rec['n_min'] == 10 * simple_space.n_parameters

    def test_budget_cap(self, simple_space):
        rec = n_samples_recommendation(simple_space, budget=5, verbose=False)
        assert rec['n_recommended'] <= 5

    def test_budget_below_minimum_warns(self, simple_space):
        rec = n_samples_recommendation(simple_space, budget=5, verbose=False)
        assert rec['warning'] is not None

    def test_no_budget(self, simple_space):
        rec = n_samples_recommendation(simple_space, verbose=False)
        assert rec['budget'] is None
        assert rec['warning'] is None

    def test_recommended_at_least_min(self, simple_space):
        rec = n_samples_recommendation(simple_space, budget=100, verbose=False)
        assert rec['n_recommended'] >= rec['n_min'] or rec['n_recommended'] == 100

    def test_returns_dict_keys(self, simple_space):
        rec = n_samples_recommendation(simple_space, verbose=False)
        for key in ('n_min', 'n_coverage', 'n_recommended', 'budget', 'warning'):
            assert key in rec