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


def test_plots_label_nominal_axes(monkeypatch):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    space = mergen.ParameterSpace({'x': list(range(6)),
                                   'opt': ('nominal', ['adam', 'sgd'])})
    s = mergen.Sampler(space)
    s.set_design(n_samples=5, n_validation=1)
    r = s.run(n_repeats=1, seed=1, verbose=False)
    ticks = set()
    real_close = plt.close

    def keep_ticks(fig=None):
        if fig is not None and hasattr(fig, 'axes'):
            for ax in fig.axes:
                ticks.update(t.get_text() for t in
                             ax.get_xticklabels() + ax.get_yticklabels())
        real_close(fig)

    monkeypatch.setattr(plt, 'close', keep_ticks)
    for kind in ('pairplot', '1d', '2d', 'distances', 'correlation'):
        r.plot(kind, show=False)
    assert {'adam', 'sgd'} <= ticks


def test_every_set_is_drawn_and_in_the_legend(monkeypatch):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    space = mergen.ParameterSpace({'x': list(range(10)),
                                   'y': list(range(10))})
    s = mergen.Sampler(space)
    s.add_set('plain', [[0, 9], [9, 0]])
    s.add_set('coloured', [[5, 5]], color='#000000')
    s.set_design(n_samples=10, n_validation=2, extra_sets={'test': 3})
    r = s.run(n_repeats=1, seed=1, verbose=False)
    seen = {}
    real_close = plt.close

    def keep(fig=None):
        if fig is not None and hasattr(fig, 'axes'):
            ax = fig.axes[0]
            seen['n'] = sum(len(c.get_offsets()) for c in ax.collections)
            seen['legend'] = {t.get_text() for a in fig.axes
                              for leg in [a.get_legend()] if leg
                              for t in leg.get_texts()} | {
                t.get_text() for leg in fig.legends for t in leg.get_texts()}
        real_close(fig)

    monkeypatch.setattr(plt, 'close', keep)
    r.plot('2d', show=False)
    assert seen['n'] == 10 + 2 + 3 + 2 + 1
    assert {'plain', 'coloured', 'test'} <= seen['legend']


def test_reports_cover_loaded_designs_and_escape_names(tmp_path):
    space = mergen.ParameterSpace({'learning_rate': [0.1, 0.2, 0.3, 0.4],
                                   'n_layers': [1, 2, 3, 4]})
    s = mergen.Sampler(space)
    s.load_design([[0.1, 1], [0.4, 4], [0.2, 3]], name='pilot')
    s.add_set('ext', [[0.3, 2]])
    r = s.run(verbose=False)
    r.output_dir = str(tmp_path)
    r.to_markdown('r.md')
    r.to_latex('r.tex')
    r.to_html('r.html')
    md, tex = (tmp_path / 'r.md').read_text(), (tmp_path / 'r.tex').read_text()
    assert '### pilot (3 points)' in md
    assert 'Restarts' not in md and '?' not in md.split('## Run Settings')[1][:300]
    assert r'learning\_rate' in tex and 'learning_rate &' not in tex
    assert 'ext & 1' in tex
    assert 'pilot' in (tmp_path / 'r.html').read_text()


def test_report_names_the_repeats_and_csv_names_its_index(tiny_result,
                                                           tmp_path):
    output.export_markdown(tiny_result)
    assert '- **Repeats**: 1' in (tmp_path / 'report.md').read_text()
    output.export_csv(tiny_result)
    assert (tmp_path / 'design.csv').read_text().startswith('id,')


def test_plot_reports_a_bad_request_clearly():
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    one = mergen.Sampler(mergen.ParameterSpace({'x': list(range(8))}))
    one.set_design(n_samples=3, n_validation=0)
    r = one.run(n_repeats=1, seed=1, verbose=False)
    with pytest.raises(ValueError, match='two parameters'):
        r.plot('2d', show=False)
    with pytest.raises(ValueError, match='Unknown parameter'):
        r.plot('1d', params=['zz'], show=False)


def test_plots_survive_dollar_labels_and_check_pairs(tmp_path):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    space = mergen.ParameterSpace({'x': list(range(6)), 'y': list(range(6)),
                                   'c': ('nominal', [r'$\foo$', '$5-$10'])})
    s = mergen.Sampler(space)
    s.add_set('plain', [[0, 5, r'$\foo$']])
    s.set_design(n_samples=5, n_validation=0)
    r = s.run(n_repeats=1, seed=1, verbose=False)
    r.output_dir = str(tmp_path)
    for kind in ('pairplot', '1d', '2d', 'distances'):
        r.plot(kind, show=False, save=True)
    with pytest.raises(ValueError, match='pairs'):
        r.plot('2d', params=['x', 'y', 'c'], show=False)
    with pytest.raises(ValueError, match='pairs'):
        r.plot('2d', params=['x', 'x'], show=False)
    assert r.quality_report(mc_samples=10.0, verbose=False)['mc_samples'] == 10


def test_distance_plot_gives_a_set_its_fallback_colour():
    from mergen.output import _FALLBACK, _extra_colors
    space = mergen.ParameterSpace({'x': list(range(6)), 'y': list(range(6))})
    s = mergen.Sampler(space)
    s.add_set('plain', [[0, 5]])
    s.set_design(n_samples=4, n_validation=0)
    r = s.run(n_repeats=1, seed=1, verbose=False)
    assert _extra_colors(r)['plain'] == _FALLBACK[0]


def test_user_text_reaches_every_plot_safely(tmp_path, monkeypatch):
    # Invalid mathtext in a parameter name, a label and a set name must
    # not crash any plot; a set without a colour gets its fallback
    # colour in the distance plot too.
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from mergen.output import _FALLBACK
    space = mergen.ParameterSpace({r'a$\q$b': list(range(6)),
                                   'y': list(range(6)),
                                   'c': ('nominal', [r'$\q$', 'z'])})
    s = mergen.Sampler(space)
    # Four points give six distances, enough for a density line.
    s.add_set(r'S$\q$', [[0, 5, 'z'], [5, 0, 'z'], [0, 0, 'z'], [5, 5, 'z']])
    s.set_design(n_samples=5, n_validation=0)
    r = s.run(n_repeats=1, seed=1, verbose=False)
    r.output_dir = str(tmp_path)
    colours = []
    real_close = plt.close

    def keep(fig=None):
        if fig is not None and hasattr(fig, 'axes'):
            for ax in fig.axes:
                colours.extend(to_hex(ln.get_color()) for ln in ax.lines)
        real_close(fig)

    monkeypatch.setattr(plt, 'close', keep)
    for kind in ('pairplot', '1d', '2d', 'correlation'):
        r.plot(kind, show=False, save=True)
    colours.clear()
    r.plot('distances', show=False, save=True)
    assert _FALLBACK[0] in colours


def test_colour_codes_follow_the_output(monkeypatch):
    import io
    import sys
    from mergen import _term
    monkeypatch.setattr(sys, 'stdout', io.StringIO())
    monkeypatch.delitem(sys.modules, 'ipykernel', raising=False)
    assert not _term.colour_on() and f"{_term.RED}x" == 'x'
    monkeypatch.setitem(sys.modules, 'ipykernel', object())
    assert _term.colour_on() and f"{_term.RED}x".startswith('\033[')


@pytest.mark.parametrize('text, tex, md, html', [
    (r'$\tan\beta$', r'$\tan\beta$', r'$\tan\beta$', r'$\tan\beta$'),
    ('$5 | $10', '$5 | $10', r'$5 \| $10', '$5 | $10'),
    ('$5 & $10', r'$5 \& $10', '$5 & $10', '$5 &amp; $10'),
    ('$a%b$', r'\$a\%b\$', r'\$a%b\$', r'\$a%b\$'),
    (r'$\q$', r'\$\textbackslash{}q\$', r'\$\\q\$', r'\$\q\$'),
    ('$1', r'\$1', r'\$1', r'\$1'),
    (r'$5\%$', r'$5\%$', r'$5\%$', r'$5\%$'),
    ('~~x~~ &lt;', r'\textasciitilde{}\textasciitilde{}x\textasciitilde{}'
     r'\textasciitilde{} \&lt;', r'\~\~x\~\~ \&lt;', '~~x~~ &amp;lt;')])
def test_report_escaping_keeps_only_valid_math(text, tex, md, html):
    # A $...$ span stays math only if matplotlib can parse it, as in the
    # plots; then only the characters that would break a table or a
    # LaTeX line are escaped inside it.
    from mergen.output import _html_escape, _latex_escape, _md_escape
    assert _latex_escape(text) == tex
    assert _md_escape(text) == md
    assert _html_escape(text) == html


def test_markdown_table_rows_keep_their_columns(tmp_path):
    import re
    space = mergen.ParameterSpace({'x': list(range(6)),
                                   'price': ('nominal', ['$5 | $10', 'a|b'])})
    s = mergen.Sampler(space)
    s.set_design(n_samples=4, n_validation=0)
    r = s.run(n_repeats=1, seed=1, verbose=False)
    r.output_dir = str(tmp_path)
    r.to_markdown('r.md')
    text = (tmp_path / 'r.md').read_text()
    table = text.split('### Optimised')[1].split('\n\n')[1].splitlines()

    def cells(line):
        return len(re.split(r'(?<!\\)\|', line)) - 2

    assert len(table) == 6 and {cells(ln) for ln in table} == {3}


def test_coloured_report_matches_the_plain_one(monkeypatch, capsys):
    # Colour codes must not shift any column of the quality table.
    import re
    import mergen.metrics as M
    s = mergen.Sampler(mergen.ParameterSpace({'x': list(range(6)),
                                              'y': list(range(6))}))
    s.set_design(n_samples=5, n_validation=0)
    r = s.run(criteria='phi_p', n_repeats=1, seed=1, verbose=False)
    capsys.readouterr()
    r.quality_report(mc_samples=20)
    plain = capsys.readouterr().out
    from mergen import _term
    monkeypatch.setattr(M, '_colour_on', lambda: True)
    monkeypatch.setattr(_term, 'colour_on', lambda: True)
    r.quality_report(mc_samples=20)
    coloured = capsys.readouterr().out
    assert '\033[' in coloured
    assert re.sub(r'\033\[[0-9;]*m', '', coloured) == plain


def test_html_tables_and_markdown_headings_escape_like_the_page(tmp_path):
    space = mergen.ParameterSpace({r'p$\q$': list(range(4)),
                                   'c': ('nominal', ['lone $', '<b>'])})
    s = mergen.Sampler(space)
    s.load_design([[0, 'lone $'], [3, '<b>']], name='set $5 | $10')
    r = s.run(verbose=False)
    r.output_dir = str(tmp_path)
    r.to_html('r.html')
    r.to_markdown('r.md')
    html = (tmp_path / 'r.html').read_text()
    md = (tmp_path / 'r.md').read_text()
    assert r'<th>p\$\q\$</th>' in html and r'<td>lone \$</td>' in html
    assert '<td>&lt;b&gt;</td>' in html
    # In a heading the math keeps its plain '|'; in a table it is \|.
    assert '### set $5 | $10 (2 points)' in md
    assert '- **set $5 | $10**: 2' in md
    # A backslash before a math span must not hide it from MathJax.
    from mergen.output import _html_escape
    assert _html_escape(r'x\$y$') == r'x\<span></span>$y$'


def test_percentile_is_padded_before_it_is_coloured():
    from mergen.metrics import _fmt_pct
    assert len(_fmt_pct(55.0, 10)) == 10 and _fmt_pct(55.0, 10).endswith('pct')
