"""Tests for focus neighbourhoods (add_focus)."""
from __future__ import annotations

import numpy as np
import pytest

import mergen


def _focus_offsets(scale, res=11, spread=1.0):
    space = mergen.ParameterSpace({
        'T': ('continuous', 200 * scale, 400 * scale, {'resolution': res}),
        'P': ('continuous', 1.0, 5.0, {'resolution': res})})
    s = mergen.Sampler(space)
    s.set_design(n_samples=12, n_validation=0)
    s.add_focus([300 * scale, 3.0], spread=spread, n_samples=4)
    r = s.run(criteria='phi_p', algorithm='sce', seed=44, n_repeats=1,
              verbose=False)
    f = r.samples[r.samples.point_type == 'Focus'][['T', 'P']].to_numpy()
    step = np.array([200 * scale, 4.0]) / (res - 1)
    return np.round((f - [300 * scale, 3.0]) / step).astype(int)


def test_spread_is_in_grid_steps_whatever_the_units():
    a = _focus_offsets(1.0)
    assert np.array_equal(a, _focus_offsets(1e-6))
    assert np.array_equal(a, _focus_offsets(1.0, res=101))
    assert np.abs(a).max() <= 4


def test_neighbours_stay_near_the_centre_on_a_fine_grid():
    # 0.1.0 drew candidates from the whole grid, so on a fine grid the
    # "focus" points landed anywhere.
    off = _focus_offsets(1.0, res=1001, spread=2.0)
    assert np.abs(off).max() <= 8


def test_nominal_axis_keeps_the_centre_category():
    space = mergen.ParameterSpace({'x': ('continuous', 0, 1,
                                         {'resolution': 21}),
                                   'c': ('nominal', ['A', 'B', 'C'])})
    s = mergen.Sampler(space)
    s.set_design(n_samples=9, n_validation=0)
    s.add_focus([0.5, 1], spread=1.0, n_samples=3)
    r = s.run(criteria='maxproqq', algorithm='sce', seed=44, n_repeats=1,
              verbose=False)
    focus = r.samples[r.samples.point_type == 'Focus']
    assert len(focus) >= 3 and set(focus['c']) == {'B'}


@pytest.mark.parametrize('bad', [0, -1.0, 'wide'])
def test_spread_must_be_positive(bad):
    s = mergen.Sampler(mergen.ParameterSpace({'x': [0, 1, 2]}))
    with pytest.raises(ValueError, match='spread must be a positive'):
        s.add_focus([1], spread=bad)


def test_a_short_focus_region_keeps_the_design_size():
    # spread=0.25 limits the neighbours to the 3x3 box around the
    # centre, fewer than the 12 requested; the optimiser takes the rest.
    space = mergen.ParameterSpace({'x': list(range(10)),
                                   'y': list(range(10))})
    s = mergen.Sampler(space)
    s.set_design(n_samples=20, n_validation=0)
    s.add_focus([5, 5], spread=0.25, n_samples=12)
    r = s.run(criteria='phi_p', algorithm='sce', seed=44, n_repeats=1,
              verbose=False)
    assert len(r.samples) == 20


def test_overlapping_focus_regions_get_all_their_points():
    # The second centre is already a point of the first region; the
    # second region still gets its 3 points.
    space = mergen.ParameterSpace({'x': list(range(12)),
                                   'y': list(range(12))})
    s = mergen.Sampler(space)
    s.set_design(n_samples=20, n_validation=0)
    # spread=0.25 keeps the first region to the 3x3 box around (5, 5),
    # so its 9 points take the second centre (6, 5).
    s.add_focus([5, 5], spread=0.25, n_samples=9)
    s.add_focus([6, 5], spread=1.0, n_samples=3, include_center=True)
    r = s.run(criteria='phi_p', algorithm='sce', seed=44, n_repeats=1,
              verbose=False)
    assert (r.samples.point_type == 'Focus').sum() == 12
