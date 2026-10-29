"""Tests for n_repeats='auto' and the canonical form of delivered designs."""
from __future__ import annotations

import numpy as np
import pytest

import mergen
from mergen.sampler import _canonical_design, _mirror_axes


def _grid_sampler(n=6):
    space = mergen.ParameterSpace({'x': [0, 1, 2, 3, 4],
                                   'y': [0, 1, 2, 3, 4]})
    s = mergen.Sampler(space)
    s.set_design(n_samples=n, n_validation=0)
    return s


def _hard_sampler(n=12):
    # The maintainer's 9 x 16 grid with a short search: repeats and
    # seeds give different designs here, unlike on the 5 x 5 grid.
    s = mergen.Sampler(mergen.ParameterSpace({
        'k': [round(0.1 * i, 1) for i in range(2, 11)],
        'm': list(range(1000, 2501, 100))}))
    s.set_design(n_samples=n, n_validation=2)
    s.set_optimizer('sa', n_restarts=1, max_iter=300)
    return s


class TestAutoRepeats:
    def test_auto_stops_once_the_result_is_settled(self):
        r = _grid_sampler().run(criteria='phi_p', algorithm='sce', seed=44,
                                verbose=False)
        info = r._repeat_info['sce']
        assert info['converged'] and info['hits'] >= 5
        assert info['n_repeats'] == len(r._repeats_by_alg['sce'])

    def test_design_does_not_depend_on_the_seed(self):
        designs = [_grid_sampler().run(criteria='phi_p', algorithm='sce',
                                       seed=seed, verbose=False)
                   .samples[['x', 'y']].to_numpy()
                   for seed in (44, 7, 123)]
        assert all(np.array_equal(d, designs[0]) for d in designs)

    def test_integer_runs_exactly_that_many_repeats(self):
        r = _grid_sampler().run(criteria='phi_p', algorithm='sce', seed=44,
                                n_repeats=3, verbose=False)
        assert len(r._repeats_by_alg['sce']) == 3
        assert r._repeat_info['sce']['budget_factor'] == 1

    def test_user_budget_is_never_scaled(self, monkeypatch):
        # Every repeat counts as a different result, so the default
        # search length doubles; a short work limit keeps the test fast.
        import mergen.sampler as sampler_mod
        monkeypatch.setattr(sampler_mod, '_same_values', lambda a, b: a is b)
        monkeypatch.setattr(sampler_mod, '_AUTO_MAX_EFFORT', 15)

        def run(**opt):
            s = mergen.Sampler(mergen.ParameterSpace({
                'k': [round(0.1 * i, 1) for i in range(2, 11)],
                'm': list(range(1000, 2501, 100))}))
            s.set_design(n_samples=12, n_validation=0)
            if opt:
                s.set_optimizer('sa', **opt)
            return s.run(criteria='phi_p', algorithm='sa', seed=44,
                         n_jobs=-1, verbose=False)._repeat_info['sa']

        assert run()['budget_factor'] == 2
        info = run(max_iter=2000)
        assert info['budget_factor'] == 1
        assert info['n_repeats'] == 15

    def test_n_jobs_does_not_change_the_result(self):
        a = _hard_sampler().run(criteria='cd2', algorithm='sa', seed=44,
                                n_repeats=6, n_jobs=1, verbose=False)
        b = _hard_sampler().run(criteria='cd2', algorithm='sa', seed=44,
                                n_repeats=6, n_jobs=2, verbose=False)
        # The repeats really differ, so a seed tied to the worker would
        # show.
        scores = [r.score for r in a._repeats_by_alg['sa']]
        assert len(set(scores)) > 1
        assert a.samples.equals(b.samples)
        assert a.validation.equals(b.validation)
        assert scores == [r.score for r in b._repeats_by_alg['sa']]

    def test_a_doubled_round_past_the_limit_runs_at_the_current_length(
            self, monkeypatch):
        # Every repeat counts as a different result, so each round asks
        # for a doubled search. Rounds cost 5, then 10; a third doubled
        # round (20) would pass the limit of 26, so it runs at the
        # current length (10).
        import mergen.sampler as sampler_mod
        monkeypatch.setattr(sampler_mod, '_same_values', lambda a, b: a is b)
        monkeypatch.setattr(sampler_mod, '_AUTO_MAX_EFFORT', 26)
        info = _grid_sampler().run(criteria='phi_p', algorithm='sa',
                                   seed=44, verbose=False)._repeat_info['sa']
        assert info['n_repeats'] == 15 and info['budget_factor'] == 2
        assert not info['converged']

    @pytest.mark.parametrize('bad', [0, -1, 2.5, True, 'many'])
    def test_invalid_values_are_refused(self, bad):
        with pytest.raises(ValueError, match="positive integer or 'auto'"):
            _grid_sampler().run(criteria='cd2', n_repeats=bad, verbose=False)
        with pytest.raises(ValueError, match="positive integer or 'auto'"):
            _grid_sampler().compare(criteria=['cd2'], algorithms=['sa'],
                                    n_repeats=bad, verbose=False)


class TestCanonicalDesign:
    def _space(self):
        return mergen.ParameterSpace({'x': [0, 1, 2, 3], 'y': [0, 1, 2]})

    def test_mirror_images_share_one_canonical_form(self):
        space = self._space()
        gs = space.grid_sampler()
        axes = _mirror_axes(space, gs, set(), [])
        assert axes == [0, 1]
        D = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
        images = [D, D * [-1, 1] + [3, 0], D * [1, -1] + [0, 2],
                  D * [-1, -1] + [3, 2]]
        forms = [_canonical_design(X, axes, gs, 0) for X in images]
        assert all(np.array_equal(f, forms[0]) for f in forms)

    def test_asymmetric_constraint_blocks_the_reflection(self):
        space = self._space()
        space.add_constraint(lambda p: p['x'] + p['y'] <= 3)
        gs = space.grid_sampler()
        assert _mirror_axes(space, gs, set(), []) == []

    def test_banned_node_off_the_mirror_blocks_the_reflection(self):
        space = self._space()
        gs = space.grid_sampler()
        banned = {gs.point_to_index(np.array([0.0, 1.0]))}
        assert _mirror_axes(space, gs, banned, []) == [1]

    def test_frozen_rows_stay_in_place(self):
        space = self._space()
        gs = space.grid_sampler()
        # Reflecting y maps the movable rows to [[3, 2], [0, 0]], which
        # sorts first; the frozen row on the mirror line stays put.
        D = np.array([[1.0, 1.0], [3.0, 0.0], [0.0, 2.0]])
        out = _canonical_design(D, [1], gs, n_frozen=1)
        assert np.array_equal(out, [[1.0, 1.0], [0.0, 0.0], [3.0, 2.0]])


def test_lambda_constraint_runs_in_parallel():
    # The fork backend cannot pickle a lambda; loky (cloudpickle) can.
    space = mergen.ParameterSpace({'x': [0, 1, 2, 3, 4], 'y': [0, 1, 2, 3, 4]})
    space.add_constraint(lambda p: p['x'] + p['y'] <= 6)
    s = mergen.Sampler(space)
    s.set_design(n_samples=6, n_validation=1)
    kw = dict(criteria='phi_p', algorithm='sce', seed=44, n_repeats=4,
              verbose=False)
    par = s.run(n_jobs=2, **kw)
    seq = s.run(n_jobs=1, **kw)
    assert par.samples.equals(seq.samples)
    cmp = s.compare(criteria=['phi_p', 'cd2'], algorithms=['sce'],
                    n_repeats=2, n_jobs=2, mc_samples=20, verbose=False)
    assert cmp.best in cmp.results


def test_prescribed_point_in_a_user_set_is_refused():
    s = mergen.Sampler(mergen.ParameterSpace({'x': [0, 1, 2, 3]}))
    s.set_design(n_samples=2, n_validation=0)
    s.add_set('ext', [[1]])
    s.add_prescribed([[1]])
    with pytest.raises(ValueError, match="in the set 'ext'"):
        s.run(n_repeats=1, verbose=False)


def test_run_leaves_the_global_random_state_alone():
    import random
    np.random.seed(5)
    random.seed(5)
    expected = (np.random.random(), random.random())
    np.random.seed(5)
    random.seed(5)
    _grid_sampler().run(criteria='phi_p', algorithm='sce', seed=44,
                        n_repeats=1, n_jobs=1, verbose=False)
    assert (np.random.random(), random.random()) == expected


@pytest.mark.parametrize('kw', [dict(n_validation=-2), dict(n_validation=1.7),
                                dict(n_samples=0), dict(n_samples=True),
                                dict(extra_sets={'Validation': 2}),
                                dict(extra_sets={'test': -1})])
def test_set_design_refuses_invalid_sizes_and_names(kw):
    s = mergen.Sampler(mergen.ParameterSpace({'x': [0, 1, 2, 3]}))
    with pytest.raises(ValueError):
        s.set_design(**kw)


def test_extra_set_cannot_reuse_a_user_set_name():
    s = mergen.Sampler(mergen.ParameterSpace({'x': [0, 1, 2, 3]}))
    s.add_set('test', [[1]])
    with pytest.raises(ValueError, match="already in use"):
        s.set_design(n_samples=2, extra_sets={'test': 1})


def test_compare_does_not_depend_on_n_jobs():
    def sweep(n_jobs):
        return _hard_sampler().compare(criteria=['phi_p', 'cd2'],
                                       algorithms=['sa'], n_repeats=4,
                                       mc_samples=20, n_jobs=n_jobs,
                                       verbose=False)
    a, b = sweep(1), sweep(2)
    assert a.best == b.best
    assert a.metrics.equals(b.metrics)
    assert a.best_result.samples.equals(b.best_result.samples)
    assert a.best_result.validation.equals(b.best_result.validation)
    assert np.array_equal(a.best_result.algorithm_results['sa'].design,
                          a.results[a.best].design)


def test_best_result_equals_a_fresh_run_of_the_winner():
    s = _hard_sampler()
    c = s.compare(criteria=['phi_p', 'cd2'], algorithms=['sa'],
                  mc_samples=20, verbose=False, polish=False)
    cr, alg = c.best
    reps = c._context['repeats'][c.best]
    assert len({r.score for r in reps}) > 1
    fresh = s.run(criteria=cr, algorithm=alg, priority=c.priority,
                  verbose=False)
    assert c.best_result.samples.equals(fresh.samples)
    assert c.best_result.validation.equals(fresh.validation)
    assert c.best_result._repeat_info == fresh._repeat_info


def test_compare_names_every_unsettled_combination(capsys, monkeypatch):
    import mergen.sampler as sampler_mod
    monkeypatch.setattr(sampler_mod, '_same_values', lambda a, b: a is b)
    monkeypatch.setattr(sampler_mod, '_AUTO_MAX_EFFORT', 5)
    _grid_sampler().compare(criteria=['phi_p', 'cd2'], algorithms=['sa'],
                            mc_samples=20, n_jobs=1, verbose=False)
    out = capsys.readouterr().out
    assert '2 of 2 combinations did not find' in out
    assert 'phi_p + sa' in out and 'cd2 + sa' in out


def test_run_refuses_several_criteria():
    with pytest.raises(ValueError, match='one criterion'):
        _grid_sampler().run(criteria=['phi_p', 'cd2'], n_repeats=1,
                            verbose=False)


def test_run_without_verbose_prints_no_banner(capsys):
    _grid_sampler().run(criteria='phi_p', algorithm='sce', n_repeats=1,
                        verbose=False)
    assert 'MERGEN' not in capsys.readouterr().out
    s = mergen.Sampler(mergen.ParameterSpace({'x': [0, 1, 2, 3]}))
    s.load_design([[0], [3]])
    s.set_design(n_validation=1)
    s.run(verbose=False)
    assert 'MERGEN' not in capsys.readouterr().out


def test_a_holdout_set_larger_than_the_free_points_warns(capsys):
    s = mergen.Sampler(mergen.ParameterSpace({'x': [0, 1, 2, 3, 4]}))
    s.set_design(n_samples=2, n_validation=10)
    r = s.run(criteria='phi_p', algorithm='sce', n_repeats=1, verbose=False)
    assert len(r.validation) == 3
    assert 'Only 3 free feasible points' in capsys.readouterr().out


def test_holdout_search_limit_warns(monkeypatch):
    import mergen.distances as dist
    monkeypatch.setattr(dist, '_MAXIMIN_CANDIDATES', 1)
    s = _grid_sampler()
    s.set_design(n_samples=6, n_validation=3)
    with pytest.warns(UserWarning, match='Holdout selection of 3 points'):
        s.run(criteria='phi_p', algorithm='sce', n_repeats=1, verbose=False)


def test_extra_set_names_must_be_strings():
    s = mergen.Sampler(mergen.ParameterSpace({'x': [0, 1, 2, 3]}))
    with pytest.raises(ValueError, match='must be strings'):
        s.set_design(n_samples=2, extra_sets={1: 2})


def test_unsettled_warning_reports_the_gap_of_the_other_repeats(
        monkeypatch, capsys):
    # The warning states the measured seed effect: how far the other
    # repeats' criterion scores lie from the delivered one.
    import mergen.sampler as sampler_mod
    monkeypatch.setattr(sampler_mod, '_same_values', lambda a, b: a is b)
    monkeypatch.setattr(sampler_mod, '_AUTO_MAX_EFFORT', 5)
    s = mergen.Sampler(mergen.ParameterSpace({'x': list(range(9)),
                                              'y': list(range(16))}))
    s.set_design(n_samples=12, n_validation=0)
    s.set_optimizer('sa', max_iter=300)
    r = s.run(criteria='phi_p', algorithm='sa', seed=44, n_jobs=1,
              verbose=False)
    info = r._repeat_info['sa']
    scores = [x.score for x in r._repeats_by_alg['sa']]
    gaps = [abs(x / r.best_score - 1) for x in scores
            if x != r.best_score]
    assert info['gap_max'] == pytest.approx(max(gaps))
    assert info['gap_median'] == pytest.approx(np.median(gaps))
    out = capsys.readouterr().out
    assert f"at most {100 * max(gaps):.1f} %" in out
