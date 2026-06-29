"""Tests for mergen.criteria — all optimisation criteria."""

import numpy as np
import pytest

from mergen.criteria import get_criterion, list_criteria
from mergen.criteria import UMaxPro, MaxPro, PhiP, CD2, StratifiedL2


class TestCriteriaRegistry:

    def test_list_criteria_contains_all(self):
        crits = list_criteria()
        for name in ('umaxpro', 'maxpro', 'phi_p', 'cd2', 'stratified'):
            assert name in crits

    def test_get_criterion_returns_instance(self):
        for name in list_criteria():
            c = get_criterion(name)
            assert c is not None

    def test_get_criterion_alias_phip(self):
        assert type(get_criterion('phi_p')) == type(get_criterion('phip'))

    def test_get_criterion_alias_stratified(self):
        assert type(get_criterion('stratified')) == \
               type(get_criterion('stratified_l2'))

    def test_get_criterion_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown criterion"):
            get_criterion('nonexistent')

    def test_get_criterion_case_insensitive(self):
        assert type(get_criterion('UMaxPro')) == type(get_criterion('umaxpro'))


class TestCriteriaEvaluate:

    @pytest.fixture
    def X(self):
        return np.random.default_rng(44).uniform(0, 1, (12, 3))

    @pytest.mark.parametrize("name",
        ['umaxpro', 'maxpro', 'phi_p', 'cd2', 'stratified'])
    def test_evaluate_positive(self, name, X):
        assert get_criterion(name).evaluate(X, space=None) > 0

    @pytest.mark.parametrize("name",
        ['umaxpro', 'maxpro', 'phi_p', 'cd2', 'stratified'])
    def test_evaluate_scalar(self, name, X):
        score = get_criterion(name).evaluate(X, space=None)
        assert isinstance(score, float)

    def test_umaxpro_differs_from_maxpro(self, X):
        """Periodic vs Euclidean distance should give different scores."""
        s_u = UMaxPro().evaluate(X, None)
        s_m = MaxPro().evaluate(X, None)
        assert s_u != pytest.approx(s_m)

    def test_phi_p_with_custom_p(self, X):
        s5  = PhiP(p=5).evaluate(X, None)
        s15 = PhiP(p=15).evaluate(X, None)
        assert s5 != pytest.approx(s15)

    def test_cd2_between_zero_and_one(self, X):
        score = CD2().evaluate(X, None)
        assert 0 <= score <= 1.5   # practical upper bound

    def test_stratified_l2_greater_than_cd2(self, X):
        """Stratified L2 sums over all projections, so typically larger."""
        s_strat = StratifiedL2().evaluate(X, None)
        s_cd2   = CD2().evaluate(X, None)
        assert s_strat >= s_cd2


class TestCriteriaIncremental:
    """
    incremental() must match a full recompute after the swap.
    Tested for all criteria.
    """

    @pytest.fixture
    def setup(self):
        rng    = np.random.default_rng(44)
        X      = rng.uniform(0, 1, (12, 2))
        new_pt = rng.uniform(0, 1, 2)
        return X, new_pt

    @pytest.mark.parametrize("name",
        ['umaxpro', 'maxpro', 'phi_p', 'cd2', 'stratified'])
    def test_incremental_matches_full(self, name, setup):
        X, new_pt = setup
        c         = get_criterion(name)
        score     = c.evaluate(X, space=None)

        _, new_score = c.incremental(X, 3, new_pt, space=None,
                                     current_score=score)

        X_swap    = X.copy(); X_swap[3] = new_pt
        ref_score = c.evaluate(X_swap, space=None)
        assert new_score == pytest.approx(ref_score, rel=1e-6)

    @pytest.mark.parametrize("name",
        ['umaxpro', 'maxpro', 'phi_p', 'cd2', 'stratified'])
    def test_incremental_returns_log_delta(self, name, setup):
        X, new_pt = setup
        c         = get_criterion(name)
        score     = c.evaluate(X, space=None)
        log_d, _  = c.incremental(X, 3, new_pt, space=None,
                                   current_score=score)
        assert isinstance(log_d, float)

    def test_incremental_improvement_negative_log_delta(self):
        """Swapping a bad point for a better one should give log_delta < 0."""
        # Clustered design — first point very close to second
        X = np.array([
            [0.01, 0.01],
            [0.02, 0.02],
            [0.5,  0.5 ],
            [1.0,  1.0 ],
        ])
        c     = UMaxPro()
        score = c.evaluate(X, None)
        # Replace [0.02, 0.02] with [0.33, 0.67] — better spread
        log_d, _ = c.incremental(X, 1, np.array([0.33, 0.67]),
                                  None, score)
        assert log_d < 0