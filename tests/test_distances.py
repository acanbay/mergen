"""Tests for mergen.distances (HEOM)."""
from __future__ import annotations

import numpy as np
import pytest

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


def _brute_ks(pool, ref, free, n, mask=None):
    """Plain-loop Kennard-Stone; ties within 1e-9 go to the lowest row."""
    pool, free, chosen = np.asarray(pool, float), list(free), []
    ref = [np.asarray(r, float) for r in ref]
    mask = np.zeros(pool.shape[1], bool) if mask is None else mask
    for _ in range(n):
        dist = [min(heom_squared(p, r, nominal_mask=mask) for r in ref)
                if ref else np.inf for p in pool]
        best = max(d for d, f in zip(dist, free) if f)
        i = next(k for k, (d, f) in enumerate(zip(dist, free))
                 if f and d >= best * (1 - 1e-9))
        chosen.append(i)
        free[i] = False
        ref.append(pool[i])
    return chosen


class TestKennardStone:
    def _lattice(self):
        g = np.arange(6) / 5
        return np.array([[a, b] for a in g for b in g])

    def test_matches_brute_force_with_lowest_index_ties(self):
        from mergen.distances import kennard_stone
        pool = self._lattice()
        ref  = pool[[0, 35, 14]]
        free = np.ones(len(pool), bool)
        free[[0, 35, 14]] = False
        got = kennard_stone(pool, 10, ref=ref, available=free)
        assert got.tolist() == _brute_ks(pool, ref, free, 10)

    def test_without_reference_starts_at_lowest_available_row(self):
        from mergen.distances import kennard_stone
        pool = self._lattice()
        free = np.ones(len(pool), bool)
        free[:3] = False
        assert kennard_stone(pool, 4, available=free).tolist() == \
            _brute_ks(pool, [], free, 4)
        assert kennard_stone(pool, 4, available=free)[0] == 3

    def test_stops_when_rows_run_out(self):
        from mergen.distances import kennard_stone
        pool = self._lattice()[:5]
        free = np.array([True, False, True, False, False])
        assert sorted(kennard_stone(pool, 4, available=free)) == [0, 2]

    def test_nominal_columns_use_overlap(self, mixed_space):
        # Euclidean would prefer row 0 (0.9 > 0.5); HEOM counts the
        # different nominal level as distance 1.
        from mergen.distances import kennard_stone
        pool = np.array([[0.9, 0.0], [0.0, 0.5]])
        got  = kennard_stone(pool, 1, ref=[[0.0, 0.0]], space=mixed_space)
        assert got.tolist() == [1]
        assert kennard_stone(pool, 1, ref=[[0.0, 0.0]]).tolist() == [0]


def _set_key(X, ref, S):
    """The maximin_subset score of rows S, by plain loops."""
    d_ref = [round(float(min(np.linalg.norm(X[i] - r) for r in ref)), 9)
             for i in S]
    pw = min(round(float(np.linalg.norm(X[a] - X[b])), 9)
             for k, a in enumerate(S) for b in S[k + 1:])
    return (min(min(d_ref), pw), pw, round(sum(d_ref), 9))


class TestMaximinSubset:
    @pytest.mark.parametrize('seed', range(6))
    def test_matches_exhaustive_search(self, seed):
        from itertools import combinations
        from mergen.distances import maximin_subset
        rng = np.random.default_rng(seed)
        g = np.arange(6) / 5
        X = np.array([[a, b] for a in g for b in g])
        taken = rng.choice(len(X), 6, replace=False)
        free = np.ones(len(X), bool)
        free[taken] = False
        ref = X[taken]
        best = max(combinations(np.flatnonzero(free), 3),
                   key=lambda S: (_set_key(X, ref, list(S)),
                                  tuple(-i for i in S)))
        got = maximin_subset(X, 3, ref=ref, available=free)
        assert tuple(got) == tuple(best)

    def test_never_worse_than_kennard_stone(self):
        from mergen.distances import kennard_stone, maximin_subset
        rng = np.random.default_rng(3)
        X = rng.random((60, 3))
        ref = X[:8]
        free = np.ones(60, bool)
        free[:8] = False
        ks = kennard_stone(X, 4, ref=ref, available=free)
        ms = maximin_subset(X, 4, ref=ref, available=free)
        assert _set_key(X, ref, sorted(ms)) >= _set_key(X, ref, sorted(ks))

    def test_falls_back_to_kennard_stone_beyond_the_budget(self,
                                                             monkeypatch):
        import mergen.distances as dist
        monkeypatch.setattr(dist, '_MAXIMIN_CANDIDATES', 1)
        rng = np.random.default_rng(3)
        X = rng.random((60, 3))
        free = np.ones(60, bool)
        free[:8] = False
        ks = dist.kennard_stone(X, 4, ref=X[:8], available=free)
        with pytest.warns(UserWarning, match='Kennard-Stone set is used'):
            ms = dist.maximin_subset(X, 4, ref=X[:8], available=free)
        assert ms.tolist() == ks.tolist()

    def test_work_limit_keeps_the_best_set_found(self, monkeypatch):
        import mergen.distances as dist
        # A limit that stops the search after it has improved on the
        # Kennard-Stone set but before it has finished.
        monkeypatch.setattr(dist, '_MAXIMIN_WORK', 50_000)
        rng = np.random.default_rng(3)
        X = rng.random((60, 3))
        free = np.ones(60, bool)
        free[:8] = False
        ks = dist.kennard_stone(X, 4, ref=X[:8], available=free)
        with pytest.warns(UserWarning, match='work limit'):
            ms = dist.maximin_subset(X, 4, ref=X[:8], available=free)
        assert (_set_key(X, X[:8], sorted(ms))
                > _set_key(X, X[:8], sorted(ks)))
