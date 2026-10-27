"""Tests for mergen.output exports."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import mergen
from mergen import output


@pytest.fixture
def tiny_result(tmp_path):
    # SI-scale values far below 1e-10.
    space = mergen.ParameterSpace({
        'gap': [1.5e-12, 2.25e-12, 3.125e-12, 4.0e-12],
        'x':   [1 / 3, 2 / 3, 1.0],
    })
    s = mergen.Sampler(space)
    s.set_design(n_samples=4, n_validation=1)
    s.set_optimizer('sa', n_restarts=1, max_iter=100)
    r = s.run(n_repeats=1, criteria='cd2', algorithm='sa', seed=1, verbose=False)
    r.output_dir = str(tmp_path)
    return r


class TestPrecision:
    def test_json_keeps_every_digit(self, tiny_result, tmp_path):
        # 0.1.0 wrote 1.5e-12 as 0.0 (double_precision counts decimals).
        output.export_json(tiny_result)
        rows = json.loads((tmp_path / 'design.json').read_text())
        got = sorted(r['gap'] for r in rows)
        want = sorted(output._full_df(tiny_result)['gap'].tolist())
        assert got == want
        assert min(got) > 0

    def test_csv_keeps_every_digit(self, tiny_result, tmp_path):
        output.export_csv(tiny_result)
        df = pd.read_csv(tmp_path / 'design.csv')
        want = output._full_df(tiny_result)
        assert np.array_equal(df['gap'].to_numpy(), want['gap'].to_numpy())
        assert np.array_equal(df['x'].to_numpy(), want['x'].to_numpy())


class TestQualityTableFailure:
    def test_stdout_is_restored_and_a_warning_is_given(self, tiny_result,
                                                       monkeypatch, tmp_path):
        # 0.1.0 left sys.stdout redirected when the report raised, so all
        # later output was lost, and hid the error.
        def boom(*a, **k):
            print("partial output")
            raise RuntimeError("report failed")
        monkeypatch.setattr(tiny_result, 'quality_report', boom)
        before = sys.stdout
        with pytest.warns(UserWarning, match="report failed"):
            output.export_markdown(tiny_result)
        assert sys.stdout is before
        assert (tmp_path / 'report.md').exists()


def test_excel_keeps_every_digit(tiny_result, tmp_path):
    pytest.importorskip('openpyxl')
    output.export_excel(tiny_result)
    df = pd.read_excel(tmp_path / 'design.xlsx')
    want = output._full_df(tiny_result)
    assert np.array_equal(df['x'].to_numpy(), want['x'].to_numpy())
