"""Tests for mergen.distances (HEOM)."""
from __future__ import annotations

import numpy as np
import pytest

import mergen
from mergen.distances import heom, heom_squared, heom_pairwise


class TestHEOMBackwardCompat:
    """No nominal → HEOM must reduce to plain Euclidean."""

    def test_heom_no_mask_matches_euclidean(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 2.0])
        assert np.isclose(heom(a, b), np.sqrt(1 + 4 + 4))

    def test_heom_squared_no_mask(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert np.isclose(heom_squared(a, b), 25.0)

    def test_heom_pairwise_no_nominal(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        D = heom_pairwise(X)
        assert D.shape == (3, 3)
        assert D[0, 0] == 0.0
        assert np.isclose(D[0, 1], 1.0)
        assert np.isclose(D[1, 2], np.sqrt(2))


class TestHEOMWithNominal:
    """Nominal columns → 0/1 overlap indicator."""

    def test_nominal_indicator(self):
        # 2 cols: col 0 numeric, col 1 nominal
        a = np.array([0.0, 0.0])
        b = np.array([0.5, 1.0])
        mask = np.array([False, True])
        # d² = 0.5² + 1² = 1.25
        assert np.isclose(heom(a, b, nominal_mask=mask), np.sqrt(1.25))

    def test_nominal_same_level_no_penalty(self):
        # Same nominal level → indicator = 0
        a = np.array([0.5, 1.0])
        b = np.array([0.5, 1.0])
        mask = np.array([False, True])
        assert heom(a, b, nominal_mask=mask) == 0.0

    def test_heom_from_space(self, mixed_space):
        # mixed_space: continuous + nominal
        X = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])
        D = heom_pairwise(X, space=mixed_space)
        # d(0,1)² = 0.25 + 1 = 1.25
        assert np.isclose(D[0, 1], np.sqrt(1.25))

    def test_pairwise_diagonal_zero(self, mixed_space):
        X = np.array([[0.0, 0.0], [1.0, 2.0]])
        D = heom_pairwise(X, space=mixed_space)
        assert D[0, 0] == 0.0
        assert D[1, 1] == 0.0

    def test_pairwise_symmetric(self, mixed_space):
        X = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])
        D = heom_pairwise(X, space=mixed_space)
        assert np.allclose(D, D.T)

    def test_squared_matches_squared(self, mixed_space):
        X = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])
        D  = heom_pairwise(X, space=mixed_space, squared=False)
        D2 = heom_pairwise(X, space=mixed_space, squared=True)
        assert np.allclose(D2, D ** 2)


class TestHEOMErrors:
    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            heom_squared(np.zeros(3), np.zeros(4))

    def test_mask_wrong_shape(self):
        with pytest.raises(ValueError):
            heom(np.zeros(3), np.zeros(3), nominal_mask=np.array([True, False]))