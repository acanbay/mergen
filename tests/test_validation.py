"""Scientific validation tests against independent references.

Unlike the unit tests (which check internal consistency: finiteness,
incremental-vs-full agreement, API contracts), the tests in this file
validate Mergen's implementations against *external* authorities:

- independent reference implementations (``scipy.stats.qmc``, naive
  pure-Python re-implementations written from the source formulas),
- analytically derived closed-form values,
- published numerical examples from the literature,
- exhaustive enumeration of small design spaces,
- defining mathematical properties (invariances) of each criterion.

A passing unit suite shows the code does what it was written to do;
a passing validation suite shows what it was written to do is correct.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.stats import qmc

import mergen
from mergen.criteria import CD2, MaxPro, PhiP, StratifiedL2, UMaxPro


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
    """Vorechovsky & Masek (2026): MaxPro sum with periodic distance.

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


def _stratified_l2_naive(X: np.ndarray, s: int, p: int, w: list) -> float:
    """Tian & Xu (2025), Theorem 1 Eq. 13, as plain loops.

    delta_i(t, z) = 1{floor(s^i t) = floor(s^i z)}, bins clamped at
    s^i - 1 so that a coordinate of exactly 1 stays in the last bin.
    """
    n, m  = X.shape
    term1 = -(sum(w[i] * s ** (-2 * i) for i in range(p + 1)) ** m)
    total = 0.0
    for a in range(n):
        for b in range(n):
            prod = 1.0
            for j in range(m):
                inner = 0.0
                for i in range(p + 1):
                    if i == 0:
                        eq = 1.0
                    else:
                        N  = s ** i
                        ba = min(int(N * X[a, j]), N - 1)
                        bb = min(int(N * X[b, j]), N - 1)
                        eq = 1.0 if ba == bb else 0.0
                    inner += w[i] * s ** (-i) * eq
                prod *= inner
            total += prod
    return math.sqrt(max(term1 + total / n ** 2, 0.0))


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
# uMaxPro — Vorechovsky & Masek (2026)
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
    Vorechovsky, M. & Masek, J. (2026). Uniform maximum projection
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
# Stratified L2 — Tian & Xu (2025), published values
# ─────────────────────────────────────────────────────────────────────
class TestStratifiedL2AgainstPaper:
    """Validate StratifiedL2 against values published in Tian & Xu (2025).

    References
    ----------
    Tian, Y. & Xu, H. (2025). A stratified L2-discrepancy with
        application to space-filling designs. *Journal of the Royal
        Statistical Society, Series B*. doi:10.1093/jrsssb/qkaf055
    """

    # Example 2 designs (m = 2, n = 8), coordinates k/8 as printed.
    _P1 = np.array([[0, 0], [1, 1], [2, 4], [3, 5],
                    [4, 2], [5, 3], [6, 6], [7, 7]]) / 8.0
    _P2 = np.array([[0, 0], [2, 3], [3, 6], [1, 5],
                    [6, 2], [4, 1], [5, 4], [7, 7]]) / 8.0

    # Example 5: optimal GSOA(9, 8, 3^2, 1), levels in Z_9 (Table 3).
    _GSOA = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 1, 6, 8, 7, 3, 5, 4],
        [3, 6, 7, 1, 4, 5, 8, 2],
        [4, 8, 1, 5, 6, 2, 3, 7],
        [5, 7, 4, 6, 2, 8, 1, 3],
        [6, 3, 5, 2, 8, 7, 4, 1],
        [7, 5, 8, 3, 1, 4, 2, 6],
        [8, 4, 2, 7, 3, 1, 6, 5],
    ], dtype=float)

    def test_example2_P1(self):
        """SD(P1) = 0.2416 with s=2, p=3, constant weights (Example 2)."""
        crit = StratifiedL2(s=2, p=3, weights='constant')
        assert abs(crit.evaluate(self._P1, space=None) - 0.2416) < 5e-5

    def test_example2_P2(self):
        """SD(P2) = 0.1389 (Examples 2 and 4); P2 ranks better than P1."""
        crit = StratifiedL2(s=2, p=3, weights='constant')
        sd1  = crit.evaluate(self._P1, space=None)
        sd2  = crit.evaluate(self._P2, space=None)
        assert abs(sd2 - 0.1389) < 5e-5
        assert sd2 < sd1

    def test_example5_optimal_gsoa(self):
        """SD^2 = 1.148028 for the lower-bound GSOA(9, 8, 3^2, 1).

        Levels are transformed by P = (D + 0.5)/9 as prescribed for
        fixed-level designs; s=3, p=2, constant weights (Example 5).
        """
        P    = (self._GSOA + 0.5) / 9.0
        crit = StratifiedL2(s=3, p=2, weights='constant')
        assert abs(crit.evaluate(P, space=None) ** 2 - 1.148028) < 1e-5

    def test_example5_half_column_subarray(self):
        """SD^2 = 0.07583258 for the {1,3,4,5}-column GSOA(9, 4, 3^2, 2)."""
        P    = (self._GSOA[:, [0, 2, 3, 4]] + 0.5) / 9.0
        crit = StratifiedL2(s=3, p=2, weights='constant')
        assert abs(crit.evaluate(P, space=None) ** 2 - 0.07583258) < 1e-7

    @pytest.mark.parametrize("n,d", [(5, 2), (10, 3), (16, 4)])
    def test_matches_naive_random_designs(self, n, d):
        """Vectorised evaluate agrees with the naive Eq. 13 reference."""
        X    = np.random.default_rng(44).random((n, d))
        crit = StratifiedL2(s=2, p=3, weights='constant')
        ref  = _stratified_l2_naive(X, s=2, p=3, w=[1.0] * 4)
        assert np.isclose(crit.evaluate(X, space=None), ref,
                          rtol=1e-10, atol=1e-12)

    def test_perfectly_stratified_pair_is_zero(self):
        """d=1, X={0.25, 0.75}, s=2, p=1: one point per stratum -> SD = 0.

        Closed form: term1 = -(1 + 1/4), term2 = (2*1.5 + 2*1)/4 = 1.25,
        so SD^2 = 0 exactly. The defining property of the criterion.
        """
        crit = StratifiedL2(s=2, p=1, weights='constant')
        assert crit.evaluate(np.array([[0.25], [0.75]]), space=None) < 1e-6

    def test_unstratified_pair_closed_form(self):
        """d=1, X={0.1, 0.2} (same stratum): SD^2 = 1/4, SD = 0.5."""
        crit = StratifiedL2(s=2, p=1, weights='constant')
        assert np.isclose(crit.evaluate(np.array([[0.1], [0.2]]), space=None),
                          0.5, rtol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# Invariance battery — properties every numeric criterion must satisfy
# ─────────────────────────────────────────────────────────────────────
_NUMERIC_CRITERIA = [
    pytest.param(CD2(),     id='cd2'),
    pytest.param(PhiP(),    id='phi_p'),
    pytest.param(MaxPro(),  id='maxpro'),
    pytest.param(UMaxPro(), id='umaxpro'),
    pytest.param(StratifiedL2(s=2, p=3, weights='constant'),
                 id='stratified'),
]


class TestCriterionInvariances:
    """Symmetries that every numeric criterion must satisfy.

    The distance-based criteria (phi_p, MaxPro, uMaxPro, CD2) are
    built from pairwise, per-axis terms symmetric in the two points
    and across axes; StratifiedL2's symmetry follows from the
    reflection/permutation invariance of its dyadic strata (Tian & Xu
    2025, Section 3, properties i-ii). Hence all five criteria must
    be invariant under row permutation, column permutation, and the
    reflection x -> 1 - x. A violation would indicate an indexing or
    normalisation bug invisible to unit tests.
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


# ─────────────────────────────────────────────────────────────────────
# Optimiser validation — exhaustive and statistical baselines
# ─────────────────────────────────────────────────────────────────────
_LEVELS = np.round(np.arange(0.0, 1.01, 0.1), 10)


class TestOptimizerValidation:
    """Validate the optimisers against external baselines.

    Two kinds of evidence, both independent of the optimiser code:

    1. *Exhaustive global optimum.* In one dimension the design space
       is small enough to enumerate every possible design
       (C(11, 3) = 165, C(11, 5) = 462 candidate sets). The optimiser
       must attain the brute-force global optimum of phi_p exactly.
       For n = 3 the optimum is the equispaced set {0, 1/2, 1}, the
       classical maximin solution (Johnson, Moore & Ylvisaker 1990).
    2. *Statistical baseline.* In two dimensions the optimised design
       must score better (lower CD2) than the median of 200 random
       Latin hypercube designs drawn on the same grid.

    ESE is excluded from the 1D exhaustive test by construction: its
    perturbation operator is a within-column swap (Jin, Chen &
    Sudjianto 2005), which permutes rows without changing the design
    as a point set when d = 1. It is covered by the 2D baseline test.

    Adds roughly 8-10 s to the suite (five full optimiser runs).

    References
    ----------
    Johnson, M. E., Moore, L. M. & Ylvisaker, D. (1990). Minimax and
        maximin distance designs. *Journal of Statistical Planning
        and Inference*, 26(2), 131-148.
    Jin, R., Chen, W. & Sudjianto, A. (2005). An efficient algorithm
        for constructing optimal design of computer experiments.
        *Journal of Statistical Planning and Inference*, 134(1),
        268-287.
    Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P. (1983).
        Optimization by simulated annealing. *Science*, 220(4598),
        671-680.
    """

    @pytest.mark.parametrize("algorithm", ['sa', 'sce'])
    @pytest.mark.parametrize("n", [3, 5])
    def test_global_optimum_1d_exhaustive(self, algorithm, n):
        """SA and SCE attain the brute-force phi_p optimum in 1D."""
        best = min(_phi_p_naive(np.array(c).reshape(-1, 1), 15)
                   for c in itertools.combinations(_LEVELS, n))
        space = mergen.ParameterSpace({'x': _LEVELS.tolist()})
        s = mergen.Sampler(space)
        s.set_design(n_samples=n)
        res = s.run(criteria='phi_p', algorithm=algorithm,
                    seed=44, verbose=False)
        found = _phi_p_naive(res.samples[['x']].to_numpy(), 15)
        assert np.isclose(found, best, rtol=1e-9)

    @pytest.mark.parametrize("algorithm", ['sa', 'sce', 'ese'])
    def test_optimised_beats_random_lhs(self, algorithm):
        """Each optimiser beats the median of 200 random LHS designs."""
        space = mergen.ParameterSpace({'x': _LEVELS.tolist(),
                                       'y': _LEVELS.tolist()})
        s = mergen.Sampler(space)
        s.set_design(n_samples=10)
        res = s.run(criteria='cd2', algorithm=algorithm,
                    seed=44, verbose=False)
        opt = CD2().evaluate(res.samples[['x', 'y']].to_numpy(), space=None)

        rng      = np.random.default_rng(44)
        crit     = CD2()
        baseline = np.array([
            crit.evaluate(np.column_stack(
                [rng.permutation(_LEVELS)[:10] for _ in range(2)]), None)
            for _ in range(200)])
        assert opt < np.median(baseline)
