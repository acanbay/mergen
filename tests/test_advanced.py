"""Advanced tests: constraints, focus, prescribed, nested, quality regression."""
from __future__ import annotations

import numpy as np
import pytest

import mergen


# ─────────────────────────────────────────────────────────────────────
# Constraints
# ─────────────────────────────────────────────────────────────────────
class TestConstraints:
    def test_constraint_reduces_pool(self):
        s_free = mergen.ParameterSpace({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4]})
        s_con = (mergen.ParameterSpace({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4]})
                 .add_constraint(lambda p: p['x'] + p['y'] <= 5))
        assert s_con.n_candidates < s_free.n_candidates

    def test_all_candidates_satisfy_constraint(self):
        sp = (mergen.ParameterSpace({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4]})
              .add_constraint(lambda p: p['x'] + p['y'] <= 5))
        for pt in sp.candidate_pool:
            assert pt[0] + pt[1] <= 5

    def test_constrained_run(self):
        """Sampler.run completes on a constrained discrete space."""
        sp = (mergen.ParameterSpace({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4]})
              .add_constraint(lambda p: p['x'] + p['y'] <= 5))
        s = mergen.Sampler(sp)
        s.set_design(n_samples=6, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        pts = r.best_design[['x', 'y']].values
        # On a discrete pool the SA only picks from the (already filtered)
        # candidate pool, so every returned point must obey the constraint.
        assert np.all(pts.sum(axis=1) <= 5)


# ─────────────────────────────────────────────────────────────────────
# Focus / prescribed / exclusion
# ─────────────────────────────────────────────────────────────────────
class TestFocusPrescribedExclusion:
    """Focus, prescribed and exclusion points must land on the parameter grid."""

    @pytest.fixture
    def grid_space(self):
        # A 21x21 grid with 0.05 spacing so 0.5, 0.3, 0.7 all land exactly.
        return mergen.ParameterSpace({
            'x': np.round(np.arange(0.0, 1.001, 0.05), 3).tolist(),
            'y': np.round(np.arange(0.0, 1.001, 0.05), 3).tolist(),
        })

    def test_prescribed_appears_in_design(self, grid_space):
        s = mergen.Sampler(grid_space)
        s.add_prescribed([[0.30, 0.70], [0.80, 0.20]], in_design=True)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        pts = r.best_design[['x', 'y']].values
        for pp in [[0.30, 0.70], [0.80, 0.20]]:
            dists = np.linalg.norm(pts - np.array(pp), axis=1)
            assert dists.min() < 1e-6, f"prescribed {pp} not found"

    def test_focus_region_adds_points_near_centre(self, grid_space):
        s = mergen.Sampler(grid_space)
        s.add_focus(point=[0.5, 0.5], spread=0.2,
                    n_samples=3, in_design=True)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        pts = r.best_design[['x', 'y']].values
        dists = np.linalg.norm(pts - np.array([0.5, 0.5]), axis=1)
        assert np.sum(dists < 0.3) >= 3

    def test_exclusion_zone_avoided(self, grid_space):
        s = mergen.Sampler(grid_space)
        s.add_exclusion(point=[0.5, 0.5], spread=0.15)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        pts = r.best_design[['x', 'y']].values
        dists = np.linalg.norm(pts - np.array([0.5, 0.5]), axis=1)
        assert dists.min() >= 0.05 - 1e-6, (
            "a design point lies well inside the exclusion sphere"
        )


# ─────────────────────────────────────────────────────────────────────
# Validation set
# ─────────────────────────────────────────────────────────────────────
class TestValidationSet:
    def test_n_validation_reflected_in_result(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=4)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        # best_design is training only; validation lives elsewhere
        assert len(r.best_design) == 10


# ─────────────────────────────────────────────────────────────────────
# Sequential nested
# ─────────────────────────────────────────────────────────────────────
class TestNested:
    def test_nested_returns_two_frames(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=12, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        outer, inner = mergen.sequential.nested(
            s, n_outer=8, n_inner=4,
            criteria='cd2', algorithm='sa', seed=1, verbose=False,
        )
        assert len(outer) == 8
        assert len(inner) == 4

    def test_nested_inner_subset_of_outer(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=12, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        outer, inner = mergen.sequential.nested(
            s, n_outer=8, n_inner=4,
            criteria='cd2', algorithm='sa', seed=1, verbose=False,
        )
        # Every inner point must appear in outer (nested design property)
        outer_pts = outer[['x', 'y']].values
        inner_pts = inner[['x', 'y']].values
        for ip in inner_pts:
            assert np.min(np.linalg.norm(outer_pts - ip, axis=1)) < 1e-9


# ─────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_one_dimensional_space(self):
        s = mergen.Sampler(mergen.ParameterSpace({'x': ('continuous', 0., 1.)}))
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        assert len(r.best_design) == 8

    def test_high_dimensional_space(self):
        # Use low resolution to keep the Cartesian product tractable
        params = {
            f'x{i}': ('continuous', 0., 1., {'resolution': 10})
            for i in range(6)
        }
        s = mergen.Sampler(mergen.ParameterSpace(params))
        s.set_design(n_samples=15, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        assert np.isfinite(r.best_score)
        assert len(r.best_design) == 15

    def test_single_level_parameter_warns_not_fatal(self, recwarn):
        with pytest.warns(UserWarning):
            mergen.ParameterSpace({'x': [5.0]})   # only 1 level

    def test_many_level_nominal(self):
        sp = mergen.ParameterSpace({
            'x': ('continuous', 0., 1.),
            'f': ('nominal', [f'L{i}' for i in range(20)]),
        })
        s = mergen.Sampler(sp)
        s.set_design(n_samples=20, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(criteria='maxproqq', algorithm='sa', seed=1, verbose=False)
        assert np.isfinite(r.best_score)


# ─────────────────────────────────────────────────────────────────────
# Quality regression
# ─────────────────────────────────────────────────────────────────────
class TestQualityRegression:
    """Verify optimisers actually improve the criterion vs random."""

    def test_sa_beats_random_cd2(self, num_space):
        rng = np.random.default_rng(0)
        pool = num_space.candidate_pool
        # Random baseline: median CD2 over 30 random 10-point designs
        from mergen.criteria import CD2
        crit = CD2()
        random_scores = []
        for _ in range(30):
            idx = rng.choice(len(pool), size=10, replace=False)
            X = (pool[idx] - num_space.gmins) / num_space.granges
            random_scores.append(crit.evaluate(X, num_space))
        random_median = float(np.median(random_scores))

        # Optimised design
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=1000)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)

        assert r.best_score < random_median, (
            f"SA ({r.best_score:.4f}) not better than random median "
            f"({random_median:.4f})"
        )

    def test_more_iterations_helps(self, num_space):
        """Longer SA should not do worse than a very short SA (in general)."""
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=100)
        r_short = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        s.set_optimizer('sa', n_restarts=1, max_iter=2000)
        r_long  = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        # long is at least as good (allow 5% wiggle for stochastic ties)
        assert r_long.best_score <= r_short.best_score * 1.05


# ─────────────────────────────────────────────────────────────────────
# Extend / fill_around detail
# ─────────────────────────────────────────────────────────────────────
class TestExtendFillDetail:
    def test_extend_preserves_existing_points(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        original = r.best_design[['x', 'y']].values.copy()
        extended = mergen.sequential.extend(
            s, r.best_design, n_new=4,
            criteria='cd2', algorithm='sa', verbose=False,
        )
        ext_pts = extended.best_design[['x', 'y']].values
        # Every original point must survive in the extended design
        for op in original:
            assert np.min(np.linalg.norm(ext_pts - op, axis=1)) < 1e-9

    def test_kfold_split_disjoint_test_indices(self, num_space):
        s = mergen.Sampler(num_space)
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(criteria='cd2', algorithm='sa', seed=1, verbose=False)
        folds = mergen.sequential.k_fold_split(s, r.best_design, k=5)
        # Test indices across folds should be disjoint
        seen = set()
        for _, test in folds:
            test_set = set(test.tolist())
            assert seen.isdisjoint(test_set)
            seen |= test_set
        assert seen == set(range(len(r.best_design)))


# ─────────────────────────────────────────────────────────────────────
# add_set (user-supplied named sets)
# ─────────────────────────────────────────────────────────────────────
class TestAddSet:
    @pytest.fixture
    def grid10(self):
        return mergen.ParameterSpace({
            'x': list(range(1, 11)),
            'y': list(range(1, 11)),
        })

    def test_points_and_color_in_result(self, grid10):
        s = mergen.Sampler(grid10)
        s.add_set('test', [[2, 2], [8, 8]], color='#9b5de5')
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=200)
        r = s.run(seed=1, verbose=False)
        assert 'test' in r.sets
        assert len(r.sets['test']) == 2
        assert (r.sets['test']['color'] == '#9b5de5').all()
        assert (r.sets['test']['point_type'] == 'test').all()

    def test_points_not_reselected_by_design(self, grid10):
        s = mergen.Sampler(grid10)
        s.add_set('test', [[2, 2], [8, 8], [5, 9]])
        s.set_design(n_samples=10, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=300)
        r = s.run(seed=1, verbose=False)
        design = set(map(tuple, r.best_design[['x', 'y']].values))
        held   = {(2.0, 2.0), (8.0, 8.0), (5.0, 9.0)}
        assert design.isdisjoint(held)

    def test_builtin_name_rejected(self, grid10):
        s = mergen.Sampler(grid10)
        with pytest.raises(ValueError):
            s.add_set('Validation', [[1, 1]])

    def test_duplicate_name_rejected(self, grid10):
        s = mergen.Sampler(grid10)
        s.add_set('test', [[1, 1]])
        with pytest.raises(ValueError):
            s.add_set('test', [[2, 2]])


# ─────────────────────────────────────────────────────────────────────
# load_design (externally supplied designs)
# ─────────────────────────────────────────────────────────────────────
class TestLoadDesign:
    @pytest.fixture
    def grid10(self):
        return mergen.ParameterSpace({
            'x': list(range(1, 11)),
            'y': list(range(1, 11)),
        })

    def test_array_default_label_and_color(self, grid10):
        s = mergen.Sampler(grid10)
        s.load_design([[1, 1], [5, 5], [9, 9]])
        s.set_design(n_validation=2)
        r = s.run(seed=1, verbose=False)
        assert (r.samples['point_type'] == 'Existing').all()
        assert (r.samples['color'] == '#3a86ff').all()
        assert len(r.samples) == 3

    def test_dataframe_custom_label_color(self, grid10):
        import pandas as pd
        prev = pd.DataFrame({'x': [2, 4, 6], 'y': [8, 6, 4]})
        s = mergen.Sampler(grid10)
        s.load_design(prev, name='campaign1', color='#ff8800')
        s.set_design(n_validation=2)
        r = s.run(seed=1, verbose=False)
        assert (r.samples['point_type'] == 'campaign1').all()
        assert (r.samples['color'] == '#ff8800').all()

    def test_validation_generated_and_disjoint(self, grid10):
        s = mergen.Sampler(grid10)
        s.load_design([[1, 1], [5, 5], [9, 9], [3, 7]])
        s.set_design(n_validation=3)
        r = s.run(seed=1, verbose=False)
        assert len(r.validation) == 3
        design = set(map(tuple, r.samples[['x', 'y']].values))
        val    = set(map(tuple, r.validation[['x', 'y']].values))
        assert design.isdisjoint(val)

    def test_rejects_n_samples(self, grid10):
        s = mergen.Sampler(grid10)
        s.load_design([[1, 1]])
        s.set_design(n_samples=10)
        with pytest.raises(ValueError):
            s.run(seed=1, verbose=False)

    def test_rejects_prescribed(self, grid10):
        s = mergen.Sampler(grid10)
        s.load_design([[1, 1]])
        s.add_prescribed([[5, 5]])
        with pytest.raises(ValueError):
            s.run(seed=1, verbose=False)


# ─────────────────────────────────────────────────────────────────────
# compare (criterion / algorithm sweep)
# ─────────────────────────────────────────────────────────────────────
class TestCompare:
    @pytest.fixture
    def sampler_numeric(self):
        space = mergen.ParameterSpace({
            'x': list(range(1, 9)),
            'y': list(range(1, 9)),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=150)
        return s

    def test_basic_flow(self, sampler_numeric, capsys):
        cmp = sampler_numeric.compare(
            criteria=['umaxpro', 'cd2'], mc_samples=50, verbose=False)
        capsys.readouterr()
        assert len(cmp.results) == 2
        assert len(cmp.table) == 2
        assert cmp.best in cmp.results

    def test_numeric_auto_filter(self, sampler_numeric, capsys):
        cmp = sampler_numeric.compare(mc_samples=50, verbose=False)
        capsys.readouterr()
        crits = {k[0] for k in cmp.results}
        assert 'maxproqq' not in crits
        assert 'qqd' not in crits

    def test_nominal_auto_filter(self, capsys):
        space = mergen.ParameterSpace({
            'x':   list(range(1, 9)),
            'cat': ('nominal', ['A', 'B', 'C']),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=150)
        cmp = s.compare(mc_samples=50, verbose=False)
        capsys.readouterr()
        assert {k[0] for k in cmp.results} == {'maxproqq', 'qqd'}

    def test_invalid_priority_metric(self, sampler_numeric):
        with pytest.raises(ValueError):
            sampler_numeric.compare(
                criteria=['umaxpro'], priority=('no_such_metric',),
                mc_samples=50, verbose=False)

    def test_best_result_consistency(self, sampler_numeric, capsys):
        cmp = sampler_numeric.compare(
            criteria=['umaxpro', 'phi_p'], mc_samples=50, verbose=False)
        capsys.readouterr()
        assert cmp.best_result is cmp.results[cmp.best]

    def test_percentiles_in_range(self, sampler_numeric, capsys):
        cmp = sampler_numeric.compare(
            criteria=['umaxpro', 'cd2'], mc_samples=50, verbose=False)
        capsys.readouterr()
        cols = [c for c in cmp.table.columns
                if c not in ('best', 'criterion', 'algorithm')]
        vals = cmp.table[cols].to_numpy(dtype=float)
        assert not any(map(lambda v: v != v, vals.ravel()))   # no NaN
        assert (vals >= 0).all() and (vals <= 100).all()


class TestCompareSixMetrics:
    """compare() reports all six quality metrics, not a subset."""

    @pytest.fixture
    def sampler_numeric(self):
        space = mergen.ParameterSpace({
            'x': list(range(1, 9)),
            'y': list(range(1, 9)),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=150)
        return s

    def test_all_six_metrics_present(self, sampler_numeric, capsys):
        cmp = sampler_numeric.compare(
            criteria=['cd2', 'phi_p'], mc_samples=50, verbose=False)
        capsys.readouterr()
        cols = [c for c in cmp.table.columns
                if c not in ('best', 'criterion', 'algorithm')]
        for m in ('min_distance', 'minimax', 'max_abs_correlation',
                  'projection_cd2', 'cv_distances', 'mean_distance'):
            assert m in cols


class TestCompareOrdering:
    """Rows follow production order (criteria then algorithms as listed)."""

    @pytest.fixture
    def sampler_numeric(self):
        space = mergen.ParameterSpace({
            'x': list(range(1, 9)),
            'y': list(range(1, 9)),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=120)
        s.set_optimizer('sce', n_restarts=1, max_iter=120)
        return s

    def test_row_order_matches_input(self, sampler_numeric, capsys):
        crits = ['phi_p', 'cd2', 'maxpro']
        algs = ['sa', 'sce']
        cmp = sampler_numeric.compare(
            algorithms=algs, criteria=crits, mc_samples=40, verbose=False)
        capsys.readouterr()
        got = [(r.criterion, r.algorithm) for r in cmp.table.itertuples()]
        expected = [(c, a) for c in crits for a in algs]
        assert got == expected

    def test_best_marker_present(self, sampler_numeric, capsys):
        cmp = sampler_numeric.compare(
            criteria=['cd2', 'phi_p'], mc_samples=40, verbose=False)
        capsys.readouterr()
        marks = [r.best.strip() for r in cmp.table.itertuples()]
        assert marks.count('*') == 1


class TestComparisonResultOutputs:
    """ComparisonResult.plot() and .to_markdown() produce artefacts."""

    @pytest.fixture
    def comparison(self):
        space = mergen.ParameterSpace({
            'x': list(range(1, 9)),
            'y': list(range(1, 9)),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=120)
        return s.compare(criteria=['cd2', 'phi_p'],
                         mc_samples=40, verbose=False)

    def test_plot_saves_file(self, comparison, tmp_path, capsys):
        import matplotlib
        matplotlib.use('Agg')
        comparison.output_dir = str(tmp_path)
        comparison.plot(save=True, show=False, filename='cmp.png')
        capsys.readouterr()
        assert (tmp_path / 'cmp.png').exists()

    def test_to_markdown_saves_file(self, comparison, tmp_path, capsys):
        comparison.output_dir = str(tmp_path)
        comparison.to_markdown('cmp.md')
        capsys.readouterr()
        assert (tmp_path / 'cmp.md').exists()
        text = (tmp_path / 'cmp.md').read_text(encoding='utf-8')
        assert 'criterion' in text


class TestQualityReportDefaultPercentile:
    """quality_report() runs the MC baseline by default."""

    def test_default_includes_percentile(self, capsys):
        space = mergen.ParameterSpace({
            'x': list(range(1, 9)),
            'y': list(range(1, 9)),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=120)
        r = s.run(criteria='cd2', verbose=False)
        rep = r.quality_report(verbose=False)
        capsys.readouterr()
        assert any(k.endswith('_percentile_rank') for k in rep)

    def test_opt_out_with_zero(self, capsys):
        space = mergen.ParameterSpace({
            'x': list(range(1, 9)),
            'y': list(range(1, 9)),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=120)
        r = s.run(criteria='cd2', verbose=False)
        rep = r.quality_report(mc_samples=0, verbose=False)
        capsys.readouterr()
        assert not any(k.endswith('_percentile_rank') for k in rep)


class TestLatexLabels:
    """LaTeX label helpers for criteria and metrics."""

    def test_criterion_latex(self):
        from mergen.criteria import criterion_latex
        assert criterion_latex('phi_p') == r'$\phi_p$'
        assert criterion_latex('cd2') == r'$\mathrm{CD}_2$'
        # alias resolves
        assert criterion_latex('phip') == r'$\phi_p$'
        # unknown falls back to the name
        assert criterion_latex('nope') == 'nope'

    def test_metric_latex(self):
        from mergen.metrics import metric_latex
        assert metric_latex('min_distance') == r'$d_{\min}$'
        assert metric_latex('mean_distance') == r'$\bar{d}$'
        assert metric_latex('nope') == 'nope'


class TestReportExportsLatex:
    """LaTeX/HTML/Markdown reports carry math labels and modern markup."""

    @pytest.fixture
    def result(self, tmp_path):
        space = mergen.ParameterSpace({
            'x': list(range(1, 9)),
            'y': list(range(1, 9)),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=120)
        r = s.run(criteria='phi_p', verbose=False)
        r.output_dir = str(tmp_path)
        return r, tmp_path

    def test_latex_booktabs_and_label(self, result, capsys):
        r, tmp_path = result
        r.to_latex('r.tex')
        capsys.readouterr()
        text = (tmp_path / 'r.tex').read_text(encoding='utf-8')
        assert 'toprule' in text and 'hline' not in text
        assert r'$\phi_p$' in text

    def test_html_mathjax_and_label(self, result, capsys):
        r, tmp_path = result
        r.to_html('r.html')
        capsys.readouterr()
        text = (tmp_path / 'r.html').read_text(encoding='utf-8')
        assert 'MathJax' in text
        assert r'$\phi_p$' in text

    def test_markdown_label(self, result, capsys):
        r, tmp_path = result
        r.to_markdown('r.md')
        capsys.readouterr()
        text = (tmp_path / 'r.md').read_text(encoding='utf-8')
        assert r'$\phi_p$' in text


class TestNominalStringEncoding:
    """add_set / load_design accept string labels for nominal factors."""

    @pytest.fixture
    def sampler_nominal(self):
        space = mergen.ParameterSpace({
            'depth': list(range(1, 9)),
            'opt':   ('nominal', ['adam', 'sgd', 'rmsprop']),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=6, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=100)
        return s

    def test_add_set_string_label(self, sampler_nominal, capsys):
        sampler_nominal.add_set('baseline', [[2, 'adam'], [5, 'sgd']])
        capsys.readouterr()
        # no exception means the label was encoded to its category index

    def test_load_design_string_label(self, sampler_nominal, capsys):
        sampler_nominal.load_design([[2, 'adam'], [5, 'rmsprop']])
        capsys.readouterr()

    def test_invalid_label_rejected(self, sampler_nominal):
        with pytest.raises((ValueError, KeyError)):
            sampler_nominal.add_set('bad', [[2, 'nadam']])


class TestParetoUtopiaRanking:
    """compare() selects the best design by Pareto/Utopia, not by a
    single dominant metric."""

    def test_balanced_beats_lopsided(self, capsys):
        # Build a comparison, then verify the chosen 'best' is Pareto
        # optimal and closest to the Utopia point on the priority axes.
        import numpy as np
        space = mergen.ParameterSpace({
            'x': list(range(1, 11)),
            'y': list(range(1, 11)),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=12, n_validation=0)
        s.set_optimizer('sa', n_restarts=2, max_iter=200)
        s.set_optimizer('sce', n_restarts=2, max_iter=200)
        prio = ('min_distance', 'max_abs_correlation')
        cmp = s.compare(criteria=['cd2', 'maxpro', 'phi_p'],
                        algorithms=['sa', 'sce'],
                        priority=prio, mc_samples=60, verbose=False)
        capsys.readouterr()

        t = cmp.table
        pts = t[list(prio)].to_numpy(dtype=float)
        best_row = t[t['best'].str.strip() == '*'].iloc[0]
        best_vec = best_row[list(prio)].to_numpy(dtype=float)

        # 1. The winner must be Pareto-optimal (not dominated by anyone).
        dominated = any(
            np.all(p >= best_vec) and np.any(p > best_vec)
            for p in pts if not np.array_equal(p, best_vec))
        assert not dominated

        # 2. Among Pareto-optimal designs, the winner is closest to Utopia.
        utopia = np.full(len(prio), 100.0)

        def is_pareto(i):
            return not any(
                np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i])
                for j in range(len(pts)) if j != i)

        pareto_d = [float(np.linalg.norm(utopia - pts[i]))
                    for i in range(len(pts)) if is_pareto(i)]
        best_d = float(np.linalg.norm(utopia - best_vec))
        assert abs(best_d - min(pareto_d)) < 1e-9


class TestCompareRepeatsAndParallel:
    """n_repeats averaging is reproducible and n_jobs does not change it."""

    @pytest.fixture
    def sampler_numeric(self):
        space = mergen.ParameterSpace({
            'x': list(range(1, 9)),
            'y': list(range(1, 9)),
        })
        s = mergen.Sampler(space)
        s.set_design(n_samples=8, n_validation=0)
        s.set_optimizer('sa', n_restarts=1, max_iter=120)
        return s

    def test_repeats_reproducible(self, sampler_numeric, capsys):
        # Same base seed + n_repeats -> identical table, twice.
        kw = dict(criteria=['cd2', 'phi_p'], algorithms=['sa'],
                  seed=44, n_repeats=3, mc_samples=40, verbose=False)
        a = sampler_numeric.compare(**kw)
        b = sampler_numeric.compare(**kw)
        capsys.readouterr()
        assert a.best == b.best
        cols = ['min_distance', 'max_abs_correlation']
        assert a.table[cols].round(6).equals(b.table[cols].round(6))

    def test_njobs_matches_sequential(self, sampler_numeric, capsys):
        import os
        kw = dict(criteria=['cd2', 'phi_p'], algorithms=['sa'],
                  seed=44, n_repeats=2, mc_samples=40, verbose=False)
        seq = sampler_numeric.compare(n_jobs=1, **kw)
        capsys.readouterr()
        # Only exercise the parallel path when >1 CPU is available.
        if (os.cpu_count() or 1) > 1:
            par = sampler_numeric.compare(n_jobs=2, **kw)
            capsys.readouterr()
            assert seq.best == par.best
            cols = ['min_distance', 'max_abs_correlation']
            assert seq.table[cols].round(6).equals(par.table[cols].round(6))
