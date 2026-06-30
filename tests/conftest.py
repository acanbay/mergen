"""
mergen test configuration
=========================
Shared fixtures and mergen package bootstrap for all test modules.
"""

import sys
import os
import types
import importlib.util

import numpy as np
import pytest

os.environ['MPLBACKEND'] = 'Agg'

# ── Bootstrap mergen package ──────────────────────────────────────────────
# Locate the mergen source directory.  Supports two layouts:
#   (a) src/mergen/   — canonical packaging layout (this repo)
#   (b) ../           — flat layout (legacy / dev sandbox)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CAND = [
    os.path.join(_HERE, '..', 'src', 'mergen'),
    os.path.join(_HERE, '..'),
]
SRC = next((p for p in _CAND
            if os.path.isfile(os.path.join(p, 'space.py'))), _CAND[-1])
SRC = os.path.abspath(SRC)


def _load(mod_name, filename):
    path = os.path.join(SRC, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m    = importlib.util.module_from_spec(spec)
    m.__package__ = 'mergen'
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


# Stub mergen package
_mergen = types.ModuleType('mergen')
_mergen.__path__     = [SRC]
_mergen.__package__  = 'mergen'
_mergen.__version__  = '0.1.0'
_mergen.__fullname__ = 'Multi-dimensional Experimental Run GENerator'
_mergen.__authors__  = [{'name': 'Ali Can Canbay',
                          'email': 'acanbay@ankara.edu.tr',
                          'orcid': 'https://orcid.org/0000-0003-4602-473X'}]
_mergen.__license__  = 'MIT'
_mergen.__github__   = 'https://github.com/acanbay/mergen'
_mergen.__docs__     = None
_mergen.__doi__      = {'software': None, 'papers': []}
_mergen._banner      = lambda: 'mergen v0.1.0'
_mergen.info         = lambda: print(_mergen._banner())
sys.modules['mergen'] = _mergen

_load('mergen.space',      'space.py')
_load('mergen.criteria',   'criteria.py')
_load('mergen.metrics',    'metrics.py')
_load('mergen.sampler',    'sampler.py')
_load('mergen.output',     'output.py')
_load('mergen.sequential', 'sequential.py')

from mergen.space   import ParameterSpace
from mergen.sampler import Sampler


# ======================================================================
# Shared fixtures
# ======================================================================

@pytest.fixture
def simple_space():
    return ParameterSpace({
        'x': np.linspace(0, 1, 20),
        'y': np.linspace(0, 1, 20),
    })


@pytest.fixture
def mixed_space():
    return ParameterSpace({
        'voltage':  [100, 200, 300, 400],
        'pressure': ('continuous', 0.5, 5.0),
        'n_layers': ('integer', 2, 8),
    })


@pytest.fixture
def constrained_space():
    space = ParameterSpace({'x': range(1, 11), 'y': range(1, 11)})
    space.add_constraint(lambda p: p['x'] + p['y'] <= 10)
    return space


@pytest.fixture
def basic_result(simple_space):
    s = Sampler(simple_space)
    s.set_design(n_samples=10, n_validation=3)
    s.set_sce(n_restarts=1, max_iter=200)
    return s.run(seed=44)