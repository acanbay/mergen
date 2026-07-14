"""Scientific validation tests against independent references.

Unlike the unit tests (which check internal consistency: finiteness,
incremental-vs-full agreement, API contracts), the tests in this file
validate Mergen's implementations against *external* authorities:

- independent reference implementations (``scipy.stats.qmc``, naive
  pure-Python re-implementations written from the source formulas),
- analytically derived closed-form values,
- defining mathematical properties (invariances) of each criterion.

A passing unit suite shows the code does what it was written to do;
a passing validation suite shows what it was written to do is correct.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import qmc

from mergen.criteria import CD2, MaxPro, PhiP, UMaxPro


# ─────────────────────────────────────────────────────────────────────
# Naive reference implementations (independent code path)
# ─────────────────────────────────────────────────────────────────────
# Written directly from the published formulas as plain double loops,
# with no vectorisation and no code shared with mergen.criteria. Any
# systematic error in the package implementation cannot be replicated
# here by construction.

def _phi_p_naive(X: np.ndarray, p: float) -> float:
    """Morris & Mitchell (1995): phi_p = (sum_{i<j} d_ij^-p)^(1/p)."""
    n, d = X.shape
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt(sum((X[i, v] - X[j, v]) ** 2 for v in range(d)))
            total += dist ** (-p)
    return total ** (1.0 / p)


def _maxpro_naive(X: np.ndarray) -> float:
    """Joseph, Gul & Ba (2015) inner sum: sum_{i<j} 1/prod_v (dx_v)^2.

    Mergen reports the raw inner pairwise sum rather than Joseph's
    ``(sum / C(n,2))^(1/d)``; the two differ by a strictly monotone
    transform, so they share the same minimiser. The naive reference
    replicates Mergen's documented convention.
    """
    n, d = X.shape
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            prod = 1.0
            for v in range(d):
                prod *= (X[i, v] - X[j, v]) ** 2
            total += 1.0 / prod
    return total


def _umaxpro_naive(X: np.ndarray) -> float:
    """Vorechovsky & Elias (2026): MaxPro sum with periodic distance.

    Per-axis squared toroidal distance min(|dx|, 1-|dx|)^2 replaces
    the Euclidean (dx)^2 of plain MaxPro.
    """
    n, d = X.shape
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            prod = 1.0
            for v in range(d):
                delta = abs(X[i, v] - X[j, v])
                prod *= min(delta, 1.0 - delta) ** 2
            total += 1.0 / prod
    return total


# ─────────────────────────────────────────────────────────────────────
# CD2 — centred L2 discrepancy (Hickernell 1998)
# ─────────────────────────────────────────────────────────────────────
class TestCD2AgainstScipy:
    """Validate CD2 against scipy's independent implementation.

    ``scipy.stats.qmc.discrepancy(X, method='CD')`` computes the
    *squared* centred discrepancy CD²; Mergen follows Hickernell
    (1998) and reports CD = sqrt(CD²). Hence the expected identity is

        mergen_cd2(X) == sqrt(scipy_cd(X))

    References
    ----------
    Hickernell, F. J. (1998). A generalized discrepancy and quadrature
        error bound. *Mathematics of Computation*, 67(221), 299-322.
    """

    @pytest.mark.parametrize("n,d", [(5, 2), (8, 2), (10, 3), (20, 4), (50, 5)])
    def test_matches_scipy_random_designs(self, n, d):
        """Bit-level agreement with scipy on fixed-seed random designs."""
        X = np.random.default_rng(44).random((n, d))
        ours   = CD2().evaluate(X, space=None)
        theirs = float(np.sqrt(qmc.discrepancy(X, method='CD')))
        assert np.isclose(ours, theirs, rtol=1e-10, atol=0.0)

    def test_single_center_point_analytic(self):
        """d=1, x=0.5: closed form CD² = 13/12 - 2 + 1 = 1/12.

        For a single point at the centre both product terms in the
        Hickernell formula equal 1, so CD = sqrt(1/12) exactly.
        """
        X = np.array([[0.5]])
        assert np.isclose(CD2().evaluate(X, space=None),
                          np.sqrt(1.0 / 12.0), rtol=1e-12, atol=0.0)


# ─────────────────────────────────────────────────────────────────────
# phi_p — Morris & Mitchell (1995)
# ─────────────────────────────────────────────────────────────────────
class TestPhiPIndependent:
    """Validate PhiP against a naive reference and closed forms.

    References
    ----------
    Morris, M. D. & Mitchell, T. J. (1995). Exploratory designs for
        computational experiments. *Journal of Statistical Planning
        and Inference*, 43(3), 381-402.
    """

    @pytest.mark.parametrize("n,d", [(5, 2), (10, 3), (20, 4)])
    @pytest.mark.parametrize("p", [2, 15, 50])
    def test_matches_naive_random_designs(self, n, d, p):
        X = np.random.default_rng(44).random((n, d))
        assert np.isclose(PhiP(p=p).evaluate(X, space=None),
                          _phi_p_naive(X, p), rtol=1e-10, atol=0.0)

    def test_two_points_closed_form(self):
        """n=2: single pair, so phi_p = (d^-p)^(1/p) = 1/d for any p.

        For the unit-square diagonal d = sqrt(2), hence phi_p = 1/sqrt(2).
        """
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        for p in (2, 15):
            assert np.isclose(PhiP(p=p).evaluate(X, space=None),
                              1.0 / math.sqrt(2.0), rtol=1e-12, atol=0.0)

    def test_three_equispaced_points_closed_form(self):
        """d=1, X={0, 1/2, 1}, p=2: sum = 4 + 4 + 1 = 9, phi_2 = 3."""
        X = np.array([[0.0], [0.5], [1.0]])
        assert np.isclose(PhiP(p=2).evaluate(X, space=None),
                          3.0, rtol=1e-12, atol=0.0)


# ─────────────────────────────────────────────────────────────────────
# MaxPro — Joseph, Gul & Ba (2015)
# ─────────────────────────────────────────────────────────────────────
class TestMaxProIndependent:
    """Validate MaxPro against a naive reference and a closed form.

    References
    ----------
    Joseph, V. R., Gul, E. & Ba, S. (2015). Maximum projection designs
        for computer experiments. *Biometrika*, 102(2), 371-380.
    """

    @pytest.mark.parametrize("n,d", [(5, 2), (10, 3), (20, 4)])
    def test_matches_naive_random_designs(self, n, d):
        X = np.random.default_rng(44).random((n, d))
        assert np.isclose(MaxPro().evaluate(X, space=None),
                          _maxpro_naive(X), rtol=1e-10, atol=0.0)

    def test_three_diagonal_points_closed_form(self):
        """X = {(0,0), (1/2,1/2), (1,1)}: 1/0.0625 + 1/1 + 1/0.0625 = 33."""
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        assert np.isclose(MaxPro().evaluate(X, space=None),
                          33.0, rtol=1e-12, atol=0.0)


# ─────────────────────────────────────────────────────────────────────
# uMaxPro — Vorechovsky & Elias (2026)
# ─────────────────────────────────────────────────────────────────────
class TestUMaxProIndependent:
    """Validate UMaxPro's periodic-distance construction.

    No third-party oracle exists for uMaxPro (the method is new), so
    validation rests on (a) a naive reference implementation, (b) a
    closed-form value, and (c) the *defining* property of the toroidal
    metric: invariance under cyclic shifts of the design, which plain
    MaxPro must (and does) violate.

    References
    ----------
    Vorechovsky, M. & Elias, J. (2026). Uniform maximum projection
        designs for computer experiments. *Computers & Structures*.
    """

    @pytest.mark.parametrize("n,d", [(5, 2), (10, 3), (20, 4)])
    def test_matches_naive_random_designs(self, n, d):
        X = np.random.default_rng(44).random((n, d))
        assert np.isclose(UMaxPro().evaluate(X, space=None),
                          _umaxpro_naive(X), rtol=1e-10, atol=0.0)

    def test_two_points_closed_form(self):
        """d=1, X={0.1, 0.9}: delta = min(0.8, 0.2) = 0.2, score = 1/0.04 = 25."""
        X = np.array([[0.1], [0.9]])
        assert np.isclose(UMaxPro().evaluate(X, space=None),
                          25.0, rtol=1e-12, atol=0.0)

    def test_toroidal_shift_invariance(self):
        """(X + s) mod 1 must leave the uMaxPro score unchanged.

        Every per-axis periodic delta min(|dx|, 1-|dx|) is invariant
        under a common cyclic shift, so the score is too. This is the
        property that removes the boundary bias of plain MaxPro.
        """
        X = np.random.default_rng(44).random((12, 3))
        crit = UMaxPro()
        base = crit.evaluate(X, space=None)
        for shift in (0.17, 0.37, 0.5, 0.83):
            shifted = np.mod(X + shift, 1.0)
            assert np.isclose(crit.evaluate(shifted, space=None),
                              base, rtol=1e-9)

    def test_plain_maxpro_breaks_under_shift(self):
        """Differentiator: the same shift changes the MaxPro score.

        X = {0.1, 0.9} shifted by +0.2 becomes {0.3, 0.1}: the plain
        Euclidean gap collapses from 0.8 to 0.2 (score 1.5625 -> 25)
        while the periodic gap stays 0.2. This confirms the two
        criteria genuinely differ where they are supposed to.
        """
        X       = np.array([[0.1], [0.9]])
        shifted = np.mod(X + 0.2, 1.0)
        mp = MaxPro()
        assert np.isclose(mp.evaluate(X,       space=None), 1.0 / 0.64,
                          rtol=1e-12)
        assert np.isclose(mp.evaluate(shifted, space=None), 25.0,
                          rtol=1e-12)
        ump = UMaxPro()
        assert np.isclose(ump.evaluate(X, space=None),
                          ump.evaluate(shifted, space=None), rtol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# Invariance battery — properties every numeric criterion must satisfy
# ─────────────────────────────────────────────────────────────────────
_NUMERIC_CRITERIA = [
    pytest.param(CD2(),     id='cd2'),
    pytest.param(PhiP(),    id='phi_p'),
    pytest.param(MaxPro(),  id='maxpro'),
    pytest.param(UMaxPro(), id='umaxpro'),
]


class TestCriterionInvariances:
    """Symmetries that follow directly from each criterion's formula.

    All four numeric criteria are built from pairwise, per-axis terms
    that are (a) symmetric in the two points, (b) symmetric across
    axes, and (c) functions of |x - y| and |x - 0.5| only. Hence every
    criterion must be invariant under row permutation, column
    permutation, and the reflection x -> 1 - x. A violation would
    indicate an indexing or normalisation bug invisible to unit tests.
    """

    @pytest.mark.parametrize("crit", _NUMERIC_CRITERIA)
    def test_row_permutation_invariance(self, crit):
        X    = np.random.default_rng(44).random((10, 3))
        perm = np.random.default_rng(7).permutation(len(X))
        assert np.isclose(crit.evaluate(X, space=None),
                          crit.evaluate(X[perm], space=None), rtol=1e-10)

    @pytest.mark.parametrize("crit", _NUMERIC_CRITERIA)
    def test_column_permutation_invariance(self, crit):
        X = np.random.default_rng(44).random((10, 3))
        assert np.isclose(crit.evaluate(X, space=None),
                          crit.evaluate(X[:, [2, 0, 1]], space=None),
                          rtol=1e-10)

    @pytest.mark.parametrize("crit", _NUMERIC_CRITERIA)
    def test_reflection_invariance(self, crit):
        X = np.random.default_rng(44).random((10, 3))
        assert np.isclose(crit.evaluate(X, space=None),
                          crit.evaluate(1.0 - X, space=None), rtol=1e-10)
