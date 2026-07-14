"""Scientific validation tests against independent references.

Unlike the unit tests (which check internal consistency: finiteness,
incremental-vs-full agreement, API contracts), the tests in this file
validate Mergen's implementations against *external* authorities:

- independent reference implementations (e.g. ``scipy.stats.qmc``),
- analytically derived closed-form values,
- published numerical examples from the literature.

A passing unit suite shows the code does what it was written to do;
a passing validation suite shows what it was written to do is correct.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

import mergen
from mergen.criteria import CD2


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
