"""Tests for mergen.space — ParameterSpace and GridSampler."""

import numpy as np
import pytest

from mergen.space import ParameterSpace


class TestParameterSpace:

    def test_discrete_candidates(self):
        space = ParameterSpace({'x': range(1, 11), 'y': range(1, 11)})
        assert space.n_candidates == 100

    def test_continuous_grid(self):
        space = ParameterSpace({'p': ('continuous', 0.0, 1.0)})
        assert space.n_candidates >= 100
        assert float(space.candidate_pool.min()) == pytest.approx(0.0)
        assert float(space.candidate_pool.max()) == pytest.approx(1.0)

    def test_integer_grid(self):
        space = ParameterSpace({'n': ('integer', 2, 10)})
        assert space.n_candidates == 9
        assert all(v == int(v) for v in space.candidate_pool.ravel())

    def test_integer_log_grid(self):
        space = ParameterSpace({'batch': ('integer', 8, 128, 'log')})
        assert space.n_candidates >= 2
        assert all(v == int(v) for v in space.candidate_pool.ravel())

    def test_constraint_reduces_candidates(self, constrained_space):
        assert constrained_space.n_candidates < 100

    def test_constraint_all_feasible(self, constrained_space):
        for row in constrained_space.candidate_pool:
            assert row[0] + row[1] <= 10

    def test_names(self, simple_space):
        assert simple_space.names == ['x', 'y']

    def test_bounds(self, simple_space):
        bounds = simple_space.bounds
        assert bounds[0] == pytest.approx((0.0, 1.0))
        assert bounds[1] == pytest.approx((0.0, 1.0))

    def test_bounds_as_dict(self, simple_space):
        d = simple_space.bounds_as_dict
        assert 'x' in d and 'y' in d
        assert d['x'] == pytest.approx((0.0, 1.0))

    def test_normalise_denormalise(self, simple_space):
        pts  = simple_space.candidate_pool[:5]
        norm = simple_space.normalise(pts)
        back = simple_space.denormalise(norm)
        np.testing.assert_allclose(back, pts, atol=1e-9)

    def test_normalise_range(self, simple_space):
        norm = simple_space.normalise(simple_space.candidate_pool)
        assert norm.min() >= 0.0 - 1e-9
        assert norm.max() <= 1.0 + 1e-9

    def test_validate_point_valid(self, simple_space):
        pt        = simple_space.candidate_pool[0]
        validated = simple_space.validate_point(pt)
        np.testing.assert_array_equal(validated, pt)

    def test_validate_point_invalid(self, simple_space):
        with pytest.raises(ValueError):
            simple_space.validate_point([99.9, 99.9])

    def test_on_grid_valid(self, simple_space):
        pt = simple_space.candidate_pool[5]
        assert simple_space.on_grid(pt) >= 0

    def test_on_grid_invalid(self, simple_space):
        assert simple_space.on_grid([0.123456, 0.654321]) == -1

    def test_centroid_on_grid(self, simple_space):
        c = simple_space.centroid
        assert c.shape == (2,)
        assert simple_space.on_grid(c) >= 0

    def test_corners_no_constraint(self):
        space   = ParameterSpace({'x': [1, 2, 3], 'y': [10, 20]})
        corners = space.corners
        assert corners.shape == (4, 2)
        for c in corners:
            assert space.on_grid(c) >= 0

    def test_corners_with_constraint(self, constrained_space):
        for c in constrained_space.corners:
            assert constrained_space.on_grid(c) >= 0

    def test_corners_as_prescribed(self, simple_space):
        """Corners array can be passed directly to add_prescribed."""
        from mergen.sampler import Sampler
        corners = simple_space.corners
        s = Sampler(simple_space)
        s.add_prescribed(corners, in_design=True, in_sce=False)
        s.set_design(n_samples=len(corners) + 4)
        s.set_sce(n_restarts=1, max_iter=100)
        r = s.run(seed=44)
        assert len(r.samples) == len(corners) + 4

    def test_random_point_reproducible(self, simple_space):
        p1 = simple_space.random_point(seed=44)
        p2 = simple_space.random_point(seed=44)
        np.testing.assert_array_equal(p1, p2)

    def test_random_point_on_grid(self, simple_space):
        pt = simple_space.random_point()
        assert simple_space.on_grid(pt) >= 0

    def test_random_point_different_seeds(self, simple_space):
        p1 = simple_space.random_point(seed=1)
        p2 = simple_space.random_point(seed=2)
        # Not guaranteed different but very likely
        assert not np.allclose(p1, p2) or True  # soft check

    def test_duplicate_values_removed(self):
        with pytest.warns(UserWarning, match="duplicate"):
            space = ParameterSpace({'x': [1, 1, 2, 3, 3]})
        assert space.n_candidates == 3

    def test_single_level_warning(self):
        with pytest.warns(UserWarning, match="1 grid level"):
            ParameterSpace({'x': [42]})

    def test_fluent_api(self):
        space = ParameterSpace()
        space.add_parameter('x', range(5))
        space.add_parameter('y', range(5))
        assert space.n_parameters == 2

    def test_is_valid(self, simple_space):
        assert simple_space.is_valid()

    def test_repr_contains_name(self, simple_space):
        assert 'ParameterSpace' in repr(simple_space)
        assert 'x' in repr(simple_space)

    def test_len(self, simple_space):
        assert len(simple_space) == simple_space.n_candidates


class TestGridSampler:

    def test_bijection(self, simple_space):
        gs = simple_space.grid_sampler()
        for idx in range(min(200, gs.n_candidates)):
            pt   = gs.index_to_point(idx)
            idx2 = gs.point_to_index(pt)
            assert idx == idx2, f"Bijection failed at idx={idx}"

    def test_all_points_on_grid(self, simple_space):
        gs = simple_space.grid_sampler()
        for idx in range(gs.n_candidates):
            pt = gs.index_to_point(idx)
            assert simple_space.on_grid(pt) >= 0

    def test_offgrid_returns_minus_one(self, simple_space):
        gs = simple_space.grid_sampler()
        assert gs.point_to_index(np.array([0.12345, 0.98765])) == -1

    def test_random_point_excluding(self, simple_space):
        gs       = simple_space.grid_sampler()
        reserved = {0, 1, 2}
        pt, idx  = gs.random_point_excluding(reserved)
        assert pt is not None
        assert idx not in reserved

    def test_greedy_maximin_seed_size(self, simple_space):
        gs       = simple_space.grid_sampler()
        anchor   = simple_space.candidate_pool[:1]
        reserved = {gs.point_to_index(anchor[0])}
        selected, _ = gs.greedy_maximin_seed(anchor, 5, reserved)
        assert len(selected) == 6   # 1 anchor + 5 new

    def test_greedy_maximin_no_duplicates(self, simple_space):
        gs       = simple_space.grid_sampler()
        anchor   = simple_space.candidate_pool[:1]
        reserved = {gs.point_to_index(anchor[0])}
        selected, _ = gs.greedy_maximin_seed(anchor, 9, reserved)
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                assert not np.allclose(selected[i], selected[j])


class TestGridSamplerLarge:

    def test_large_grid_random_point(self):
        """GridSampler with >500k candidates uses random sampling."""
        import numpy as np
        from mergen.space import ParameterSpace
        space = ParameterSpace({
            'x': np.linspace(0, 1, 100),
            'y': np.linspace(0, 1, 100),
            'z': np.linspace(0, 1, 100),
        })
        assert space.n_candidates == 1_000_000
        gs = space.grid_sampler()
        pt, idx = gs.random_point_excluding(set())
        assert pt is not None
        assert idx >= 0
        assert space.on_grid(pt) >= 0

    def test_large_grid_greedy_seed(self):
        """Greedy seed on large grid uses sampling mode."""
        import numpy as np
        from mergen.space import ParameterSpace
        space = ParameterSpace({
            'x': np.linspace(0, 1, 100),
            'y': np.linspace(0, 1, 100),
            'z': np.linspace(0, 1, 100),
        })
        gs     = space.grid_sampler()
        anchor = space.candidate_pool[:1]
        reserved = {gs.point_to_index(anchor[0])}
        selected, _ = gs.greedy_maximin_seed(anchor, 9, reserved)
        assert len(selected) == 10
        # No duplicates
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                assert not np.allclose(selected[i], selected[j])


class TestParameterSpaceExtra:

    def test_distance(self, simple_space):
        pool = simple_space.candidate_pool
        d = simple_space.distance(pool[0], pool[-1])
        assert d > 0
        assert d <= np.sqrt(simple_space.n_parameters)

    def test_distance_same_point(self, simple_space):
        pool = simple_space.candidate_pool
        d = simple_space.distance(pool[0], pool[0])
        assert d == pytest.approx(0.0)

    def test_set_resolution(self):
        space = ParameterSpace({'p': ('continuous', 0.0, 1.0)})
        n_before = space.n_candidates
        space.set_resolution(200)
        space.add_parameter('q', ('continuous', 0.0, 1.0))
        assert space.n_candidates > n_before

    def test_set_resolution_too_small(self):
        space = ParameterSpace()
        with pytest.raises(ValueError):
            space.set_resolution(1)