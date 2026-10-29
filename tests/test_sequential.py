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
    r = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=1, verbose=False)
    return s, r


@pytest.fixture
def mixed_fitted(mixed_space):
    s = mergen.Sampler(mixed_space)
    s.set_design(n_samples=12, n_validation=0)
    s.set_optimizer('sa', n_restarts=1, max_iter=500)
    r = s.run(n_repeats=1, criteria='maxproqq', algorithm='sa', seed=1, verbose=False)
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
            criteria='cd2', algorithm='sa', verbose=False, n_repeats=1
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
            criteria='cd2', algorithm='sa', verbose=False, n_repeats=1
        )
        assert len(new.best_design) == 5


class TestExtendAfterPilot:
    """0.1.0 crashed when the sampler had prescribed or focus points."""

    def _pilot(self):
        space = mergen.ParameterSpace({'x': list(range(11)),
                                       'y': list(range(11))})
        s = mergen.Sampler(space)
        s.add_prescribed([[0, 0]])
        s.add_focus([5, 5], spread=1.0, n_samples=2)
        s.set_design(n_samples=8, n_validation=0, extra_sets={'test': 2})
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        return s, s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=1,
                        verbose=False)

    def test_extend_keeps_rows_labels_and_sampler_state(self):
        s, pilot = self._pilot()
        ext = mergen.sequential.extend(s, pilot.samples, n_new=4, seed=2,
                                verbose=False, n_repeats=1)
        names = s.space.names
        assert len(ext.samples) == 12
        assert np.array_equal(ext.samples[names].to_numpy()[:8],
                              pilot.samples[names].to_numpy())
        assert (ext.samples['point_type'].tolist()[:8]
                == pilot.samples['point_type'].tolist())
        assert len(s._prescribed) == 1 and len(s._focus) == 1
        assert s._extra_sets == {'test': 2}
        assert len(ext.sets['test']) == 2

    def test_extend_keeps_an_out_of_budget_point_out_of_budget(self):
        space = mergen.ParameterSpace({'x': list(range(11)),
                                       'y': list(range(11))})
        s = mergen.Sampler(space)
        s.add_prescribed([[0, 0]])
        s.add_prescribed([[10, 10]], in_design=False)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        pilot = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=1,
                      verbose=False)
        ext = mergen.sequential.extend(s, pilot.samples, n_new=4, seed=2,
                                       verbose=False, n_repeats=1)
        names = space.names
        # 8 in the budget + 4 new, plus the out-of-budget point.
        assert len(pilot.samples) == 9 and len(ext.samples) == 13
        assert len(ext.algorithm_results['sa'].design) == 12
        assert np.array_equal(ext.samples[names].to_numpy()[:9],
                              pilot.samples[names].to_numpy())
        assert (ext.samples['point_type'].tolist()[:9]
                == pilot.samples['point_type'].tolist())
        assert ext.samples['point_type'].tolist()[9:] == ['Optimised'] * 4

    def test_extend_keeps_the_input_row_order(self):
        space = mergen.ParameterSpace({'x': list(range(11)),
                                       'y': list(range(11))})
        s = mergen.Sampler(space)
        s.add_prescribed([[10, 10]], in_design=False)
        s.set_design(n_samples=6, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        pilot = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=1,
                      verbose=False)
        rows = pilot.samples.iloc[::-1].reset_index(drop=True)
        ext = mergen.sequential.extend(s, rows, n_new=2, seed=2,
                                       verbose=False, n_repeats=1)
        names = space.names
        assert np.array_equal(ext.samples[names].to_numpy()[:len(rows)],
                              rows[names].to_numpy())

    def test_extend_warns_about_a_dropped_prescribed_point(self, capsys):
        s, pilot = self._pilot()
        capsys.readouterr()
        rows = pilot.samples[pilot.samples.point_type != 'Prescribed']
        ext = mergen.sequential.extend(s, rows, n_new=2, seed=2,
                                       verbose=False, n_repeats=1)
        assert len(ext.samples) == len(rows) + 2
        assert 'left out' in capsys.readouterr().out

    def test_extend_a_loaded_design(self):
        s, pilot = self._pilot()
        s2 = mergen.Sampler(s.space)
        s2.load_design(pilot.samples[s.space.names].to_numpy())
        s2.set_optimizer('sa', n_restarts=1, max_iter=200)
        ext = mergen.sequential.extend(s2, pilot.samples, n_new=3, seed=2,
                                verbose=False, n_repeats=1)
        assert len(ext.samples) == 11
        assert s2._loaded_design is not None


class TestFillAroundWithFixedPoints:
    def _sampler(self):
        space = mergen.ParameterSpace({'x': list(range(11)),
                                       'y': list(range(11))})
        s = mergen.Sampler(space)
        s.add_prescribed([[0, 0]])
        s.add_prescribed([[10, 10]], in_design=False)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        return s

    def test_reference_equal_to_a_prescribed_point(self):
        # 0.1.0: "appears in add_prescribed() more than once".
        s = self._sampler()
        r = mergen.sequential.fill_around(s, [[0, 0], [5, 5]], n_new=4,
                                          seed=1, verbose=False, n_repeats=1)
        P = r.samples[s.space.names].to_numpy().tolist()
        assert len(P) == 4 and r._meta['n_design'] == 4
        for ref in ([0.0, 0.0], [5.0, 5.0], [10.0, 10.0]):
            assert ref not in P
        assert len(s._prescribed) == 2

    def test_loaded_design_sampler(self):
        s = mergen.Sampler(self._sampler().space)
        s.load_design([[1, 1], [9, 9]])
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        r = mergen.sequential.fill_around(s, [[5, 5]], n_new=3, seed=1,
                                          verbose=False, n_repeats=1)
        assert len(r.samples) == 3
        assert s._loaded_design is not None


def test_fill_around_report_uses_an_unconditioned_baseline():
    space = mergen.ParameterSpace({'x': list(range(11)),
                                   'y': list(range(11))})
    s = mergen.Sampler(space)
    s.set_optimizer('sa', n_restarts=1, max_iter=200)
    res = mergen.sequential.fill_around(
        s, [[2, 2], [8, 8], [2, 8], [8, 2], [5, 5], [0, 0]], n_new=8,
        seed=1, verbose=False, n_repeats=1)
    rep = res.quality_report(mc_samples=20, verbose=False)
    assert rep['baseline_fixed_points'] == 0
    assert res._meta['fixed_points'] == []
    assert len(res._meta['baseline_reserved']) == 6


def test_default_extend_runs_on_a_nominal_space():
    space = mergen.ParameterSpace({'x': list(range(6)),
                                   'c': ('nominal', ['A', 'B', 'C'])})
    s = mergen.Sampler(space)
    s.set_design(n_samples=4, n_validation=0)
    pilot = s.run(n_repeats=1, seed=1, verbose=False)
    ext = mergen.sequential.extend(s, pilot.samples[space.names].to_numpy(),
                                   n_new=2, verbose=False, n_repeats=1)
    assert len(ext.samples) == 6 and ext._meta['criteria'] == 'maxproqq'


@pytest.mark.parametrize('bad', [2.7, 0, True, 'two'])
def test_sequential_counts_must_be_positive_integers(bad):
    space = mergen.ParameterSpace({'x': list(range(6)), 'y': list(range(6))})
    s = mergen.Sampler(space)
    with pytest.raises(ValueError, match='positive integer'):
        mergen.sequential.extend(s, [[0, 0]], n_new=bad, verbose=False)
    with pytest.raises(ValueError, match='positive integer'):
        mergen.sequential.nested(s, n_outer=6, n_inner=bad, verbose=False)
