"""Tests for mergen.space.ParameterSpace."""
from __future__ import annotations

import numpy as np
import pytest

import mergen


class TestBasicParameterTypes:
    def test_discrete_list(self):
        s = mergen.ParameterSpace({'x': [1, 2, 3, 4]})
        assert s.n_parameters == 1
        assert s.param_types['x'] == 'discrete'
        assert s.n_levels == [4]

    def test_continuous_linear(self):
        s = mergen.ParameterSpace({'x': ('continuous', 0.0, 1.0)})
        assert s.param_types['x'] == 'continuous'
        assert s.bounds == [(0.0, 1.0)]

    def test_continuous_log(self):
        s = mergen.ParameterSpace({'lr': ('continuous', 1e-4, 1e-1, 'log')})
        vals = s.values[0]
        assert vals.min() >= 1e-4 and vals.max() <= 1e-1

    def test_integer_range(self):
        s = mergen.ParameterSpace({'n': ('integer', 2, 10)})
        assert s.param_types['n'] == 'integer'
        assert s.n_levels[0] == 9

    def test_custom_resolution(self):
        s = mergen.ParameterSpace(
            {'x': ('continuous', 0.0, 1.0, {'resolution': 50})}
        )
        assert s.n_levels[0] == 50

    def test_range_object(self):
        s = mergen.ParameterSpace({'x': range(0, 10, 2)})
        assert s.n_levels[0] == 5


class TestCategoricalTypes:
    def test_nominal(self):
        s = mergen.ParameterSpace({'f': ('nominal', ['A', 'B', 'C'])})
        assert s.param_types['f'] == 'nominal'
        assert s.category_labels('f') == ['A', 'B', 'C']
        assert s.has_nominal is True
        assert s.has_categorical is True
        assert s.is_nominal('f')

    def test_ordinal(self):
        s = mergen.ParameterSpace({'q': ('ordinal', ['low', 'med', 'high'])})
        assert s.param_types['q'] == 'ordinal'
        assert s.is_ordinal('q')
        assert s.has_categorical is True
        assert s.has_nominal is False

    def test_nominal_stored_as_integers(self):
        s = mergen.ParameterSpace({'f': ('nominal', ['A', 'B', 'C'])})
        assert np.array_equal(s.values[0], np.array([0.0, 1.0, 2.0]))

    def test_categorical_names(self):
        s = mergen.ParameterSpace({
            'x': ('continuous', 0.0, 1.0),
            'f': ('nominal', ['A', 'B']),
            'q': ('ordinal', ['lo', 'hi']),
        })
        assert s.nominal_names == ['f']
        assert s.ordinal_names == ['q']
        assert set(s.categorical_names) == {'f', 'q'}

    def test_is_mask(self):
        s = mergen.ParameterSpace({
            'x': ('continuous', 0.0, 1.0),
            'f': ('nominal', ['A', 'B']),
            'y': ('continuous', 0.0, 1.0),
        })
        assert np.array_equal(s.is_mask, np.array([False, True, False]))

    def test_duplicate_labels_fatal(self):
        with pytest.raises((ValueError, SystemExit)):
            mergen.ParameterSpace({'f': ('nominal', ['a', 'b', 'a'])})

    def test_empty_labels_fatal(self):
        with pytest.raises((ValueError, SystemExit)):
            mergen.ParameterSpace({'f': ('nominal', [])})


class TestSpaceGeometry:
    def test_candidate_pool_matches_product(self, mixed_space):
        """n_candidates should be product of per-parameter level counts."""
        expected = int(np.prod(mixed_space.n_levels))
        assert mixed_space.n_candidates == expected

    def test_gmins_granges(self, num_space):
        assert np.allclose(num_space.gmins, [0.0, 0.0])
        assert np.allclose(num_space.granges, [1.0, 1.0])

    def test_bad_bounds_fatal(self):
        with pytest.raises((ValueError, SystemExit)):
            mergen.ParameterSpace({'x': ('continuous', 1.0, 0.0)})

    def test_duplicate_name_fatal(self):
        s = mergen.ParameterSpace({'x': [1, 2]})
        with pytest.raises((ValueError, SystemExit)):
            s.add_parameter('x', [3, 4])


class TestFluentAPI:
    def test_add_parameter_chain(self):
        s = (mergen.ParameterSpace()
             .add_parameter('x', [1, 2, 3])
             .add_parameter('y', ('continuous', 0.0, 1.0)))
        assert s.n_parameters == 2
        assert s.names == ['x', 'y']


class TestGridTolerance:
    """0.1.0 matched values with an absolute 1e-9, merging tiny levels."""

    def _space(self):
        return mergen.ParameterSpace({
            'gap': [1.0e-12, 2.0e-12, 3.0e-12],
            'big': [1.0e6, 1.0e6 + 1.0, 1.0e6 + 2.0],
        })

    def test_point_to_index_separates_tiny_levels(self):
        space = self._space()
        gs = space.grid_sampler()
        got = [gs.point_to_index(p) for p in space.candidate_pool]
        assert got == list(range(9))

    def test_off_grid_value_is_rejected(self):
        gs = self._space().grid_sampler()
        assert gs.point_to_index([1.5e-12, 1.0e6]) == -1
        assert gs.point_to_index([1.0e-12, 1.0e6 + 0.5]) == -1

    def test_on_grid_matches_within_tolerance(self):
        space = self._space()
        assert space.on_grid([2.0e-12 * (1 + 1e-9), 1.0e6 + 1.0]) == 4
        assert space.on_grid([2.5e-12, 1.0e6 + 1.0]) == -1

    def test_tiny_levels_are_not_duplicates_in_a_design(self):
        s = mergen.Sampler(self._space())
        s.set_design(n_samples=6, n_validation=2)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        r = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=1, verbose=False)
        P = np.vstack([r.samples[['gap', 'big']].to_numpy(float),
                       r.validation[['gap', 'big']].to_numpy(float)])
        assert len(np.unique(P, axis=0)) == len(P) == 8

    def test_tolerance_is_at_least_a_few_ulps(self):
        # 1e-6 of a gap of 0.01 is below one ulp at 1.7e9.
        space = mergen.ParameterSpace({'t': [1.7e9, 1.7e9 + 0.01]})
        off = np.nextafter(1.7e9, 2e9)
        assert space.on_grid([off]) == 0

    def test_gmins_and_granges_are_cached_read_only(self):
        space = self._space()
        assert space.gmins is space.gmins
        with pytest.raises(ValueError):
            space.granges[0] = 1.0

    @pytest.mark.parametrize("alg", ['sa', 'sce', 'ese'])
    def test_design_does_not_depend_on_the_unit(self, alg):
        # 0.1.0 treated a range below 1e-10 as degenerate in the
        # metrics and SCE skipped levels closer than 1e-12.
        out = []
        for scale in (1.0, 1e-12):
            space = mergen.ParameterSpace({
                'g': [k * scale for k in range(8)], 'y': list(range(8))})
            s = mergen.Sampler(space)
            s.set_design(n_samples=8, n_validation=2)
            s.set_optimizer(alg, **({'n_restarts': 2, 'max_iter': 300}
                                    if alg != 'ese'
                                    else {'n_restarts': 2, 'M': 10, 'J': 10}))
            r = s.run(n_repeats=1, criteria='phi_p', algorithm=alg, seed=3,
                      verbose=False)
            D = r.samples[['g', 'y']].to_numpy().copy()
            D[:, 0] /= scale
            rp = r.quality_report(mc_samples=10, verbose=False)
            out.append((np.round(D).tolist(), r.best_score,
                        rp['min_distance']))
        assert out[0][0] == out[1][0]
        assert out[0][1] == pytest.approx(out[1][1])
        assert out[0][2] == pytest.approx(out[1][2])

    def test_duplicate_prescribed_points_at_tiny_scale_are_distinct(self):
        s = mergen.Sampler(self._space())
        s.add_prescribed([[1.0e-12, 1.0e6], [2.0e-12, 1.0e6]])
        s.add_exclusion([3.0e-12, 1.0e6])
        s._check_conflicts()


def test_integer_range_holds_only_the_integers_inside():
    space = mergen.ParameterSpace({'a': ('integer', 1.5, 3.7)})
    assert space.grid_sampler().values[0].tolist() == [2.0, 3.0]


def test_set_resolution_rebuilds_default_grids_only():
    space = mergen.ParameterSpace({
        'a': ('continuous', 0, 1),
        'b': ('continuous', 0, 1, {'resolution': 5}),
        'c': [1, 2, 3]})
    space.set_resolution(11)
    assert space.grid_sampler().n_levels == [11, 5, 3]
    assert space.n_candidates == 11 * 5 * 3


def test_a_validated_point_is_the_grid_node_itself():
    space = mergen.ParameterSpace(
        {'a': ('continuous', 0.0, 1.0, {'resolution': 11})})
    node = space.grid_sampler().values[0][3]
    assert space.validate_point([0.3])[0] == node
