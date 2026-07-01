"""Shared pytest fixtures for the Mergen test suite."""
from __future__ import annotations

import os
import numpy as np
import pytest

# Silence Mergen's banner in tests
os.environ.setdefault("MERGEN_SILENT", "1")

import mergen


@pytest.fixture
def rng():
    """Deterministic numpy Generator."""
    return np.random.default_rng(42)


@pytest.fixture
def num_space():
    """Small 2-D numerical (grid-discrete) parameter space."""
    return mergen.ParameterSpace({
        'x': np.arange(0.0, 1.01, 0.1).tolist(),
        'y': np.arange(0.0, 1.01, 0.1).tolist(),
    })


@pytest.fixture
def cont_space():
    """2-D continuous space."""
    return mergen.ParameterSpace({
        'x': ('continuous', 0.0, 1.0),
        'y': ('continuous', 0.0, 1.0),
    })


@pytest.fixture
def mixed_space():
    """continuous + nominal (3 levels)."""
    return mergen.ParameterSpace({
        'temp':  ('continuous', 0.0, 1.0),
        'flour': ('nominal', ['A', 'B', 'C']),
    })


@pytest.fixture
def full_mixed_space():
    """continuous + nominal + ordinal."""
    return mergen.ParameterSpace({
        'temp':     ('continuous', 0.0, 1.0),
        'material': ('nominal', ['steel', 'wood', 'plastic']),
        'batch':    ('ordinal', ['low', 'med', 'high']),
    })


@pytest.fixture
def small_design(rng):
    """10x2 random design in [0,1]²."""
    return rng.random((10, 2))