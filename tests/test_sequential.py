"""Tests for mergen.sequential."""
from __future__ import annotations

import numpy as np
import pytest

import mergen


@pytest.fixture
def fitted(num_space):
    """Sampler with an optimized result (design as DataFrame)."""
    s = mergen.Sampler(num_space)
    s.set_design(n_samples=10, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=300)
    r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
    return s, r


@pytest.fixture
def mixed_fitted(mixed_space):
    s = mergen.Sampler(mixed_space)
    s.set_design(n_samples=12, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=500)
    r = s.run(criteria='maxproqq', algorithm='sa', seed=1, verbose=False)
    return s, r


class TestSubsample:
    def test_basic(self, fitted, num_space):
        s, _ = fitted
        picks = mergen.sequential.subsample(
            s, num_space.candidate_pool, n_select=5, anchor='center',
        )
        assert picks.shape == (5, 2)

    def test_maximin_anchor(self, fitted, num_space):
        s, _ = fitted
        picks = mergen.sequential.subsample(
            s, num_space.candidate_pool, n_select=5, anchor='maximin',
        )
        assert picks.shape == (5, 2)

    def test_heom_nominal_balance(self, mixed_fitted, mixed_space):
        """Perfect 3/3/3 balance on nominal levels with HEOM."""
        s, _ = mixed_fitted
        picks = mergen.sequential.subsample(
            s, mixed_space.candidate_pool, n_select=9, anchor='maximin',
        )
        _, counts = np.unique(picks[:, 1], return_counts=True)
        assert sorted(counts.tolist()) == [3, 3, 3]


class TestRunOrder:
    def test_returns_ordered_dataframe(self, fitted):
        s, r = fitted
        ordered = mergen.sequential.run_order(s, r.best_design)
        assert 'run_order' in ordered.columns
        assert set(ordered['run_order']) == set(range(len(r.best_design)))


class TestExtend:
    def test_appends_new_points(self, fitted):
        s, r = fitted
        n_old = len(r.best_design)
        extended = mergen.sequential.extend(
            s, r.best_design, n_new=3,
            criteria='cd2', algorithm='sa', verbose=False,
        )
        assert len(extended.best_design) == n_old + 3


class TestKFoldSplit:
    def test_partitions_design(self, fitted):
        s, r = fitted
        folds = mergen.sequential.k_fold_split(s, r.best_design, k=5)
        assert len(folds) == 5
        # each fold is (train_idx, test_idx); test indices union covers all
        all_test = np.concatenate([test for _, test in folds])
        assert len(np.unique(all_test)) == len(r.best_design)


class TestFillAround:
    def test_returns_n_new(self, fitted):
        s, r = fitted
        ref = r.best_design.iloc[:3]
        new = mergen.sequential.fill_around(
            s, ref, n_new=5,
            criteria='cd2', algorithm='sa', verbose=False,
        )
        assert len(new.best_design) == 5
