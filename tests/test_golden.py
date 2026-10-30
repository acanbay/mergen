"""
Golden (snapshot) tests for :meth:`Sampler.run`.

Each scenario runs a small, fully seeded optimisation and compares the
delivered design, labels, validation set and extra sets with a stored
snapshot in ``tests/golden/``. The suite is a safety net for refactors:
a change that is meant to preserve behaviour must leave every snapshot
untouched, and a change that is meant to alter results must regenerate
the affected snapshots deliberately and say so in its commit message.

Regenerate with::

    MERGEN_UPDATE_GOLDEN=1 pytest tests/test_golden.py

Snapshots are recorded on Linux. Optimiser decisions can diverge on
other platforms through floating-point ties, so the comparison is
skipped elsewhere.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import mergen

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("MERGEN_UPDATE_GOLDEN") == "1"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") and not UPDATE,
    reason="golden snapshots are recorded on Linux",
)

_BUDGETS = {
    'sa':  dict(n_restarts=1, max_iter=400),
    'sce': dict(n_restarts=1, max_iter=400),
    'ese': dict(n_restarts=1, J=10, M=20),
}


def _lattice_sampler():
    """2-D lattice of the shape used in the ATLAS analysis report."""
    space = mergen.ParameterSpace({
        'kappa': [round(v, 1) for v in np.arange(0.2, 1.01, 0.1)],
        'mass':  list(range(1000, 2501, 100)),
    })
    s = mergen.Sampler(space)
    s.set_design(n_samples=12, n_validation=3)
    return s


def _constrained_sampler():
    space = mergen.ParameterSpace({'x': list(range(11)),
                                   'y': list(range(11))})
    space.add_constraint(lambda p: p['x'] + p['y'] <= 14)
    s = mergen.Sampler(space)
    s.add_prescribed([[0, 0]], in_design=True)
    s.add_prescribed([[10, 2]], in_design=False)
    s.add_focus([5, 5], spread=1.0, n_samples=2, in_design=True)
    s.set_design(n_samples=10, n_validation=2, extra_sets={'test': 2})
    return s


def _nominal_sampler():
    space = mergen.ParameterSpace({
        'temp': ('continuous', 0.0, 1.0, {'resolution': 11}),
        'cat':  ('nominal', ['A', 'B', 'C']),
    })
    s = mergen.Sampler(space)
    s.set_design(n_samples=9, n_validation=2)
    return s


SCENARIOS = {
    'lattice_cd2_sa':          (_lattice_sampler, 'cd2', 'sa', 1),
    'lattice_phi_p_sce':       (_lattice_sampler, 'phi_p', 'sce', 1),
    'lattice_umaxpro_ese':     (_lattice_sampler, 'umaxpro', 'ese', 1),
    'lattice_maxpro_sa_rep3':  (_lattice_sampler, 'maxpro', 'sa', 3),
    'constrained_cd2_sa':      (_constrained_sampler, 'cd2', 'sa', 1),
    'nominal_maxproqq_sa':     (_nominal_sampler, 'maxproqq', 'sa', 1),
}


def _snapshot(result):
    def frame(df):
        cols = [c for c in df.columns if c not in ('point_type', 'color')]
        # Nominal columns hold labels; the snapshot keeps level indices.
        num = result.space.encode_frame(df[cols])
        return {
            'values': np.asarray(num, dtype=float).tolist(),
            'labels': (df['point_type'].tolist()
                       if 'point_type' in df.columns else []),
        }
    return {
        'samples':    frame(result.samples),
        'validation': frame(result.validation),
        'sets':       {k: frame(v) for k, v in result.sets.items()},
        'score':      float(result.best_score),
    }


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_run_matches_golden(name, capsys):
    factory, crit, alg, n_repeats = SCENARIOS[name]
    s = factory()
    s.set_optimizer(alg, **_BUDGETS[alg])
    result = s.run(criteria=crit, algorithm=alg, seed=7,
                   n_repeats=n_repeats, n_jobs=1, verbose=False)
    snap = _snapshot(result)
    path = GOLDEN_DIR / f"{name}.json"

    if UPDATE:
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(snap, indent=1) + "\n")
        pytest.skip(f"snapshot written: {path.name}")

    if not path.exists():
        pytest.fail(f"missing snapshot {path.name}; "
                    f"run with MERGEN_UPDATE_GOLDEN=1")
    ref = json.loads(path.read_text(encoding='utf-8'))

    for part in ('samples', 'validation'):
        assert snap[part]['labels'] == ref[part]['labels'], part
        np.testing.assert_allclose(snap[part]['values'],
                                   ref[part]['values'],
                                   rtol=0, atol=1e-12, err_msg=part)
    assert sorted(snap['sets']) == sorted(ref['sets'])
    for k in ref['sets']:
        assert snap['sets'][k]['labels'] == ref['sets'][k]['labels'], k
        np.testing.assert_allclose(snap['sets'][k]['values'],
                                   ref['sets'][k]['values'],
                                   rtol=0, atol=1e-12, err_msg=k)
    assert snap['score'] == pytest.approx(ref['score'], rel=1e-12)
