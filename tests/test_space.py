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


class TestLabels:
    """Nominal and ordinal levels go in and come out as labels."""

    def _space(self):
        return mergen.ParameterSpace({
            'x': list(range(6)),
            'opt': ('nominal', ['adam', 'sgd', 'rmsprop']),
            'size': ('ordinal', ['small', 'medium', 'large'])})

    def test_constraint_receives_labels(self):
        space = self._space()
        space.add_constraint(lambda p: p['opt'] != 'sgd'
                             and p['size'] != 'large')
        pool = space.candidate_pool
        names = space.names
        assert len(pool) == 6 * 2 * 2
        assert not np.any(pool[:, names.index('opt')] == 1.0)
        assert not np.any(pool[:, names.index('size')] == 2.0)

    def test_samples_and_exports_show_labels(self, tmp_path):
        import json
        space = self._space()
        s = mergen.Sampler(space)
        s.add_prescribed([[0, 'rmsprop', 'small']])
        s.add_focus([3, 'adam', 'medium'], spread=1.0, n_samples=2)
        s.add_exclusion([5, 'sgd', 'large'])
        s.set_design(n_samples=8, n_validation=2)
        r = s.run(n_repeats=1, seed=1, verbose=False)
        assert set(r.samples['opt']) <= {'adam', 'sgd', 'rmsprop'}
        assert set(r.validation['size']) <= {'small', 'medium', 'large'}
        assert ['0', 'rmsprop', 'small'] in [
            [str(int(a)), b, c] for a, b, c
            in r.samples[space.names].itertuples(index=False)]
        r.output_dir = str(tmp_path)
        r.to_json('d.json')
        rows = json.loads((tmp_path / 'd.json').read_text(encoding='utf-8'))
        assert {row['opt'] for row in rows} <= {'adam', 'sgd', 'rmsprop'}
        r.quality_report(mc_samples=10, verbose=False)

    def test_extend_accepts_the_labelled_samples(self):
        space = self._space()
        s = mergen.Sampler(space)
        s.set_design(n_samples=5, n_validation=0)
        pilot = s.run(n_repeats=1, seed=1, verbose=False)
        ext = mergen.sequential.extend(s, pilot.samples, n_new=2,
                                       verbose=False, n_repeats=1)
        assert ext.samples.iloc[:5][space.names].equals(
            pilot.samples[space.names])

    def test_codes_are_accepted_unless_labels_are_numbers(self):
        space = self._space()
        assert space.encode_points([[1, 2, 0]]).tolist() == [[1, 2, 0]]
        with pytest.raises(ValueError, match='Valid levels'):
            space.encode_points([[1, 'lion', 0]])
        # A label that is a number is matched by value or by its text,
        # never read as a level index.
        num = mergen.ParameterSpace({'g': ('ordinal', [16, 32, 64])})
        assert num.encode_points([[32], ['64'], [16.0]]).tolist() == [
            [1.0], [2.0], [0.0]]
        with pytest.raises(ValueError, match='Valid levels'):
            num.encode_points([[1]])

    def _numeric_labels(self):
        space = mergen.ParameterSpace({'x': list(range(8)),
                                       'bs': ('ordinal', [16, 32, 64])})
        s = mergen.Sampler(space)
        s.set_design(n_samples=5, n_validation=0)
        return space, s

    def test_number_labels_round_trip(self, tmp_path):
        import pandas as pd
        space, s = self._numeric_labels()
        space.add_constraint(lambda p: p['bs'] <= 32 or p['x'] < 4)
        r = s.run(n_repeats=1, seed=1, verbose=False)
        assert set(r.samples['bs']) <= {16, 32, 64}
        assert all(row.bs <= 32 or row.x < 4
                   for row in r.samples.itertuples())
        r.output_dir = str(tmp_path)
        r.to_csv('d.csv')
        back = pd.read_csv(tmp_path / 'd.csv')
        s2 = mergen.Sampler(space)
        s2.load_design(back[space.names])
        assert np.array_equal(s2._loaded_design[0],
                              space.encode_frame(r.samples[space.names])
                              .to_numpy(float))
        order = mergen.sequential.run_order(s, back[space.names])
        assert set(order['bs']) <= {16, 32, 64}

    def test_extend_and_fill_around_with_number_labels(self):
        space, s = self._numeric_labels()
        pilot = s.run(n_repeats=1, seed=1, verbose=False)
        ext = mergen.sequential.extend(s, pilot.samples, n_new=2,
                                       verbose=False, n_repeats=1)
        assert len(ext.samples) == 7
        fill = mergen.sequential.fill_around(s, [[3, 32]], n_new=2,
                                             verbose=False, n_repeats=1)
        assert len(fill.samples) == 2

    def test_point_functions_take_a_dataframe_by_column(self):
        import pandas as pd
        space, s = self._numeric_labels()
        df = pd.DataFrame({'bs': [64, 16], 'x': [1, 6]})
        s.add_prescribed(df)
        s.add_set('ext', df.assign(x=[2, 7]))
        assert [p[0].tolist() for p in s._prescribed] == [[1, 2], [6, 0]]

    def test_run_order_of_an_array_returns_labels(self):
        space = self._space()
        s = mergen.Sampler(space)
        out = mergen.sequential.run_order(s, [[0, 'adam', 'small'],
                                              [5, 'sgd', 'large']])
        assert set(out['opt']) == {'adam', 'sgd'}


class TestSpecChecks:
    def test_non_finite_values_are_refused(self):
        with pytest.raises(ValueError, match='finite'):
            mergen.ParameterSpace({'x': [1, float('nan'), 3]})

    def test_generator_spec_survives_set_resolution(self):
        sp = mergen.ParameterSpace({'x': (v for v in range(5)),
                                    'y': ('continuous', 0, 1)})
        sp.set_resolution(11)
        assert sp.n_levels == [5, 11]

    @pytest.mark.parametrize('res', [1, 0, 2.5, True])
    def test_resolution_must_be_at_least_two(self, res):
        with pytest.raises(ValueError, match='>= 2'):
            mergen.ParameterSpace({'x': ('continuous', 0, 1)},
                                  resolution=res)
        with pytest.raises(ValueError, match='>= 2'):
            mergen.ParameterSpace({'x': ('continuous', 0, 1,
                                         {'resolution': res})})

    def test_categorical_spec_with_extra_items_names_the_problem(self):
        with pytest.raises(ValueError, match='exactly one list'):
            mergen.ParameterSpace({'g': ('ordinal', ['a', 'b'], 'x')})

    def test_infeasible_grid_point_names_the_constraint(self):
        sp = mergen.ParameterSpace({'x': list(range(5)),
                                    'y': list(range(5))})
        sp.add_constraint(lambda p: p['x'] <= p['y'])
        with pytest.raises(ValueError, match='violates a constraint'):
            sp.validate_point([4, 0])
        with pytest.raises(ValueError, match='not on the parameter grid'):
            sp.validate_point([4.5, 0])


def test_range_bounds_must_be_finite():
    with pytest.raises(ValueError, match='finite'):
        mergen.ParameterSpace({'x': ('continuous', float('nan'), 1)})


class TestLabelEdges:
    def test_labels_naming_the_same_number_are_refused(self):
        for labels in ([1, 1.0], ['01', '1'], [0.1, '0.1']):
            with pytest.raises(ValueError, match='duplicate'):
                mergen.ParameterSpace({'g': ('nominal', labels)})

    def test_subsample_of_an_array_returns_labels(self):
        space = mergen.ParameterSpace({'x': list(range(4)),
                                       'bs': ('ordinal', [16, 32, 64])})
        s = mergen.Sampler(space)
        out = mergen.sequential.subsample(
            s, np.array([[0, 16], [3, 64], [1, 32]]), 2)
        assert set(out[:, 1]) <= {16, 32, 64}

    def test_messages_show_labels(self):
        space = mergen.ParameterSpace({'x': list(range(4)),
                                       'opt': ('nominal', ['adam', 'sgd'])})
        space.add_constraint(lambda p: p['opt'] != 'sgd' or p['x'] < 2)
        with pytest.raises(ValueError, match="'sgd'"):
            space.validate_point([3, 'sgd'])

    def test_report_range_lists_escaped_labels(self, tmp_path):
        space = mergen.ParameterSpace({'x': list(range(6)),
                                       'opt': ('nominal', ['rms_prop', 'a|b'])})
        s = mergen.Sampler(space)
        s.set_design(n_samples=4, n_validation=0)
        r = s.run(n_repeats=1, seed=1, verbose=False)
        r.output_dir = str(tmp_path)
        r.to_markdown('r.md')
        r.to_latex('r.tex')
        r.to_html('r.html')
        assert r'rms\_prop, a\|b' in (tmp_path / 'r.md').read_text(encoding='utf-8')
        tex = (tmp_path / 'r.tex').read_text(encoding='utf-8')
        assert r'rms\_prop, a\textbar{}b' in tex
        assert 'rms_prop, a|b' in (tmp_path / 'r.html').read_text(encoding='utf-8')

    def test_reports_escape_every_name(self, tmp_path):
        space = mergen.ParameterSpace({'lr_x': list(range(6)),
                                       'c|d': list(range(3)),
                                       r'$\tan\beta$': [1, 2],
                                       'opt': ('nominal', ['<b>', 'x&y',
                                                           'a|b'])})
        s = mergen.Sampler(space)
        s.load_design([[0, 0, 1, '<b>'], [5, 2, 2, 'x&y'], [3, 1, 1, 'a|b']],
                      name='old<design>#')
        s.add_set('my_set&', [[1, 0, 2, 'x&y']])
        r = s.run(verbose=False)
        r.output_dir = str(tmp_path)
        r.to_latex('r.tex')
        r.to_html('r.html')
        r.to_markdown('r.md')
        tex, html, md = [(tmp_path / f).read_text(encoding='utf-8')
                         for f in ('r.tex', 'r.html', 'r.md')]
        for raw in ('old<design>#', 'my_set&', 'lr_x &', 'x&y', 'a|b'):
            assert raw not in tex
        assert r'old\textless{}design\textgreater{}\#' in tex
        assert r'my\_set\&' in tex and r'$\tan\beta$' in tex
        assert '<h3>old&lt;design&gt;# (3 points)</h3>' in html
        assert 'my_set&amp;' in html and '&lt;b&gt;' in html
        assert r'### old\<design\># (3 points)' in md
        assert r'- **old\<design\>#**: 3' in md
        assert r'## my\_set\& (1 points)' in md
        assert r'a\|b' in md and r'c\|d' in md and '| a|b ' not in md
        assert r'$\tan\beta$' in md


def test_nan_labels_are_refused_and_bool_labels_read_back():
    for labels in ([float('nan'), 'x'], ['nan', 'NaN']):
        with pytest.raises(ValueError, match='NaN'):
            mergen.ParameterSpace({'g': ('nominal', labels)})
    sp = mergen.ParameterSpace({'g': ('nominal', [True, 1])})
    assert sp.encode_points([['True'], [1]]).tolist() == [[0.0], [1.0]]


def test_a_number_label_is_found_however_it_is_written():
    sp = mergen.ParameterSpace({'g': ('ordinal', [1, 2, 4])})
    assert sp.encode_points([['1.0'], [2.0], ['4']]).tolist() == [
        [0.0], [1.0], [2.0]]
