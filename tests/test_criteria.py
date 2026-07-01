"""Tests for mergen.criteria (all 7 registered criteria)."""
from __future__ import annotations

import numpy as np
import pytest

import mergen
from mergen.criteria import (
    BaseCriterion,
    UMaxPro, MaxPro, MaxProQQ, PhiP, CD2, QQD, StratifiedL2,
    get_criterion, list_criteria, nominal_supporting_criteria,
)


class TestRegistry:
    def test_list_criteria(self):
        expected = {'umaxpro', 'maxpro', 'maxproqq',
                    'phi_p', 'cd2', 'qqd', 'stratified'}
        assert set(list_criteria()) == expected

    def test_aliases(self):
        assert isinstance(get_criterion('maxpro_qq'), MaxProQQ)
        assert isinstance(get_criterion('phip'),      PhiP)
        assert isinstance(get_criterion('stratified_l2'), StratifiedL2)

    def test_case_insensitive(self):
        assert isinstance(get_criterion('CD2'),      CD2)
        assert isinstance(get_criterion('MaxProQQ'), MaxProQQ)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_criterion('does_not_exist')

    def test_nominal_supporting_list(self):
        assert set(nominal_supporting_criteria()) == {'maxproqq', 'qqd'}

    def test_supports_nominal_attribute(self):
        assert MaxProQQ.supports_nominal is True
        assert QQD.supports_nominal is True
        assert MaxPro.supports_nominal is False
        assert UMaxPro.supports_nominal is False
        assert CD2.supports_nominal is False
        assert PhiP.supports_nominal is False
        assert StratifiedL2.supports_nominal is False
        assert BaseCriterion.supports_nominal is False


class TestNumericalCriteria:
    """Sanity: return finite positive scores on well-behaved LHD-ish data."""

    @pytest.mark.parametrize("name",
                             ['umaxpro', 'maxpro', 'phi_p', 'cd2', 'stratified'])
    def test_evaluate_finite(self, name, cont_space, small_design):
        c = get_criterion(name)
        s = c.evaluate(small_design, cont_space)
        assert np.isfinite(s)
        assert s >= 0.0


class TestMaxProQQ:
    def test_reduces_to_maxpro_on_continuous(self, cont_space, small_design):
        s_mp = MaxPro().evaluate(small_design, cont_space)
        s_qq = MaxProQQ().evaluate(small_design, cont_space)
        assert np.isclose(s_mp, s_qq, rtol=1e-6)

    def test_finite_on_mixed(self, mixed_space, rng):
        X = np.column_stack([rng.random(10), rng.choice([0.0, 0.5, 1.0], 10)])
        s = MaxProQQ().evaluate(X, mixed_space)
        assert np.isfinite(s) and s > 0

    def test_incremental_matches_evaluate(self, mixed_space, rng):
        X = np.column_stack([rng.random(10), rng.choice([0.0, 0.5, 1.0], 10)])
        crit = MaxProQQ()
        s0 = crit.evaluate(X, mixed_space)
        X_new = X.copy();  X_new[3] = np.array([0.42, 0.5])
        s_full = crit.evaluate(X_new, mixed_space)
        _, s_inc = crit.incremental(X, 3, X_new[3], mixed_space, s0)
        assert np.isclose(s_full, s_inc, rtol=1e-9)


class TestQQD:
    def test_reduces_to_wd_no_nominal(self, cont_space, small_design):
        """QQD on all-quantitative == Hickernell WD²."""
        n, q = small_design.shape
        diff = np.abs(small_design[:, None, :] - small_design[None, :, :])
        K = 1.5 - diff + diff * diff
        wd_sq = -(4.0/3.0)**q + K.prod(axis=-1).sum() / (n*n)
        qqd_sq = QQD().evaluate(small_design, cont_space)
        assert np.isclose(qqd_sq, wd_sq, rtol=1e-10)

    def test_zhou_2021_example_D1_exact(self):
        """QQD²(D^(1)) == 0.0213 from Zhang, Yang & Zhou (2021)."""
        raw = np.array([0, 2, 4, 6, 1, 3, 5, 7])
        transform = lambda x: (2.0 * x + 1.0) / 16.0
        D1_nom = np.array([0, 0, 0, 0, 1, 1, 1, 1]) / 1.0
        D_1 = np.column_stack([D1_nom, transform(raw), transform(raw)])
        space = mergen.ParameterSpace({
            'D1':  ('nominal', ['A', 'B']),
            'x1':  ('continuous', 0.0, 1.0),
            'x2':  ('continuous', 0.0, 1.0),
        })
        score = QQD().evaluate(D_1, space)
        assert abs(score - 0.0213) < 1e-4

    def test_positive_on_mixed(self, mixed_space, rng):
        X = np.column_stack([rng.random(10), rng.choice([0.0, 0.5, 1.0], 10)])
        s = QQD().evaluate(X, mixed_space)
        assert s > 0

    def test_incremental_matches_evaluate(self, mixed_space, rng):
        X = np.column_stack([rng.random(10), rng.choice([0.0, 0.5, 1.0], 10)])
        crit = QQD()
        s0 = crit.evaluate(X, mixed_space)
        X_new = X.copy();  X_new[3] = np.array([0.42, 0.5])
        s_full = crit.evaluate(X_new, mixed_space)
        _, s_inc = crit.incremental(X, 3, X_new[3], mixed_space, s0)
        assert np.isclose(s_full, s_inc, rtol=1e-12)


class TestNumericalStability:
    def test_no_warnings_on_collision(self, mixed_space, recwarn):
        """MaxProQQ floor keeps scores finite even on colliding rows."""
        X = np.array([[0.0, 0.0], [0.0, 1.0], [0.5, 0.5]])  # rows collide on col0
        s = MaxProQQ().evaluate(X, mixed_space)
        assert np.isfinite(s)
        assert not any('divide' in str(w.message) for w in recwarn)