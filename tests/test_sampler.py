"""Tests for mergen.sampler — Sampler, SamplingResult, FocusPoint, ExclusionPoint."""

import numpy as np
import pytest

from mergen.space   import ParameterSpace
from mergen.sampler import Sampler, FocusPoint, ExclusionPoint


def _make_sampler(space, n_samples=10, n_validation=None, **sa_kwargs):
    s = Sampler(space)
    s.set_design(n_samples=n_samples, n_validation=n_validation)
    s.set_sa(n_restarts=1, max_iter=200, **sa_kwargs)
    return s


class TestSamplerBasic:

    def test_basic_run_size(self, simple_space):
        r = _make_sampler(simple_space).run(seed=44)
        assert len(r.samples) == 10

    def test_validation_set_size(self, simple_space):
        s = Sampler(simple_space)
        s.set_design(n_samples=10, n_validation=3)
        s.set_sa(n_restarts=1, max_iter=200)
        r = s.run(seed=44)
        assert len(r.validation) == 3

    def test_extra_sets(self, simple_space):
        s = Sampler(simple_space)
        s.set_design(n_samples=10, n_validation=2,
                     extra_sets={'test': 5, 'holdout': 3})
        s.set_sa(n_restarts=1, max_iter=200)
        r = s.run(seed=44)
        assert len(r.sets['test'])    == 5
        assert len(r.sets['holdout']) == 3

    def test_all_points_on_grid(self, basic_result):
        space = basic_result.space
        for _, row in basic_result.samples.iterrows():
            pt = row[space.names].values.astype(float)
            assert space.on_grid(pt) >= 0

    def test_no_duplicate_design_points(self, basic_result):
        pts = basic_result.samples[basic_result.space.names].values
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert not np.allclose(pts[i], pts[j])

    def test_reproducibility(self, simple_space):
        def _run():
            return _make_sampler(simple_space).run(seed=44)
        r1, r2 = _run(), _run()
        np.testing.assert_array_equal(
            r1.samples[simple_space.names].values,
            r2.samples[simple_space.names].values,
        )

    def test_mixed_parameters(self, mixed_space):
        r = _make_sampler(mixed_space, n_samples=15).run(seed=44)
        assert len(r.samples) == 15

    def test_constrained_space(self, constrained_space):
        r = _make_sampler(constrained_space, n_samples=8).run(seed=44)
        assert len(r.samples) == 8
        for _, row in r.samples.iterrows():
            assert row['x'] + row['y'] <= 10


class TestSamplerPrescribed:

    def test_in_design_true(self, simple_space):
        pt = simple_space.candidate_pool[0]
        s  = Sampler(simple_space)
        s.add_prescribed([pt], in_design=True, in_sa=False)
        s.set_design(n_samples=10)
        s.set_sa(n_restarts=1, max_iter=200)
        r  = s.run(seed=44)
        assert len(r.samples) == 10
        vc = r.samples['point_type'].value_counts()
        assert vc.get('Prescribed', 0) == 1
        assert vc.get('Optimised',  0) == 9

    def test_in_design_false(self, simple_space):
        pt = simple_space.candidate_pool[0]
        s  = Sampler(simple_space)
        s.add_prescribed([pt], in_design=False, in_sa=True)
        s.set_design(n_samples=8)
        s.set_sa(n_restarts=1, max_iter=200)
        assert len(s.run(seed=44).samples) == 9   # 8 + 1 extra

    def test_multiple_prescribed(self, simple_space):
        pts = simple_space.candidate_pool[:3]
        s   = Sampler(simple_space)
        s.add_prescribed(pts, in_design=True, in_sa=False)
        s.set_design(n_samples=10)
        s.set_sa(n_restarts=1, max_iter=200)
        r   = s.run(seed=44)
        assert len(r.samples) == 10
        assert r.samples['point_type'].value_counts().get('Prescribed', 0) == 3

    def test_prescribed_point_in_output(self, simple_space):
        pt = simple_space.candidate_pool[7]
        s  = Sampler(simple_space)
        s.add_prescribed([pt], in_design=True, in_sa=False)
        s.set_design(n_samples=10)
        s.set_sa(n_restarts=1, max_iter=200)
        r  = s.run(seed=44)
        pts = r.samples[simple_space.names].values
        assert any(np.allclose(pt, p) for p in pts)


class TestSamplerCriteria:

    @pytest.mark.parametrize("crit",
        ['umaxpro', 'maxpro', 'phi_p', 'cd2', 'stratified'])
    def test_all_criteria_run(self, simple_space, crit):
        s = Sampler(simple_space)
        s.set_design(n_samples=8)
        s.set_sa(n_restarts=1, max_iter=100)
        r = s.run(criteria=crit, seed=44)
        assert len(r.samples) == 8

    def test_criteria_stored_in_meta(self, simple_space):
        s = Sampler(simple_space)
        s.set_design(n_samples=8)
        s.set_sa(n_restarts=1, max_iter=100)
        r = s.run(criteria='umaxpro', seed=44)
        assert r._meta.get('criteria') == 'umaxpro'


class TestSamplerConflicts:

    def test_prescribed_focus_conflict(self, simple_space):
        pt = simple_space.candidate_pool[5]
        s  = Sampler(simple_space)
        s.add_prescribed([pt])
        s.add_focus(pt, spread=1.0)
        with pytest.raises(ValueError):
            s.run(seed=44)

    def test_prescribed_exclusion_conflict(self, simple_space):
        pt = simple_space.candidate_pool[5]
        s  = Sampler(simple_space)
        s.add_prescribed([pt])
        s.add_exclusion(pt, spread=1.0)
        with pytest.raises(ValueError):
            s.run(seed=44)

    def test_focus_exclusion_conflict(self, simple_space):
        pt = simple_space.candidate_pool[5]
        s  = Sampler(simple_space)
        s.add_focus(pt, spread=1.0)
        s.add_exclusion(pt, spread=1.0)
        with pytest.raises(ValueError):
            s.run(seed=44)


class TestSamplingResult:

    def test_summary_runs(self, basic_result):
        basic_result.summary()

    def test_repr(self, basic_result):
        assert 'SamplingResult' in repr(basic_result)

    def test_samples_has_point_type(self, basic_result):
        assert 'point_type' in basic_result.samples.columns

    def test_point_types_valid(self, basic_result):
        valid = {'Prescribed', 'Focus', 'Optimised'}
        types = set(basic_result.samples['point_type'].unique())
        assert types.issubset(valid)

    def test_validation_has_point_type(self, basic_result):
        if len(basic_result.validation):
            assert 'point_type' in basic_result.validation.columns


class TestFocusExclusionPoint:

    def test_focus_point_repr(self, simple_space):
        pt = simple_space.candidate_pool[0]
        fp = FocusPoint(pt, spread=1.5)
        assert 'FocusPoint' in repr(fp)

    def test_exclusion_point_repr(self, simple_space):
        pt = simple_space.candidate_pool[0]
        ep = ExclusionPoint(pt, spread=1.0)
        assert 'ExclusionPoint' in repr(ep)

    def test_focus_invalid_spread(self, simple_space):
        pt = simple_space.candidate_pool[0]
        with pytest.raises(ValueError):
            FocusPoint(pt, spread=-1.0)

    def test_exclusion_invalid_spread(self, simple_space):
        pt = simple_space.candidate_pool[0]
        with pytest.raises(ValueError):
            ExclusionPoint(pt, spread=0.0)

    def test_focus_resolve_n_samples(self, simple_space):
        pt = simple_space.candidate_pool[0]
        fp = FocusPoint(pt, spread=1.0)
        assert fp.n_samples is None
        fp.resolve_n_samples(2)
        assert fp.n_samples is not None and fp.n_samples >= 1


class TestSamplerAdvanced:

    def test_n_restarts_multiple(self, simple_space):
        """SA with multiple restarts should still produce correct output."""
        s = Sampler(simple_space)
        s.set_design(n_samples=8)
        s.set_sa(n_restarts=3, max_iter=100)
        r = s.run(seed=44)
        assert len(r.samples) == 8
        assert r._meta.get('n_restarts') == 3

    def test_focus_and_exclusion_together(self, simple_space):
        """Focus + exclusion should not conflict and produce valid output."""
        pool = simple_space.candidate_pool
        s    = Sampler(simple_space)
        s.add_focus(pool[0],  spread=1.5, in_design=True, in_sa=True)
        s.add_exclusion(pool[-1], spread=1.0)
        s.set_design(n_samples=10)
        s.set_sa(n_restarts=1, max_iter=200)
        r = s.run(seed=44)
        assert len(r.samples) >= 10
        # Exclusion point should not appear in output
        excl = pool[-1]
        pts  = r.samples[simple_space.names].values
        assert not any(np.allclose(excl, p) for p in pts)

    def test_focus_include_true(self, simple_space):
        """Focus with include=True guarantees the point is in the design."""
        pool = simple_space.candidate_pool
        pt   = pool[10]
        s    = Sampler(simple_space)
        s.add_focus(pt, spread=1.0, include_center=True,
                    in_design=True, in_sa=True)
        s.set_design(n_samples=10)
        s.set_sa(n_restarts=1, max_iter=200)
        r = s.run(seed=44)
        pts = r.samples[simple_space.names].values
        assert any(np.allclose(pt, p) for p in pts)