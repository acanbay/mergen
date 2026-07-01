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