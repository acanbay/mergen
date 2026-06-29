"""Tests for mergen.output — plots and exports."""

import pandas as pd
import pytest


class TestPlots:

    def test_pairplot_saves_file(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.plot('pairplot', show=False, save=True)
        assert any(f.suffix == '.png' for f in tmp_path.iterdir())

    def test_1d_saves_file(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.plot('1d', show=False, save=True)
        assert any(f.suffix == '.png' for f in tmp_path.iterdir())

    def test_2d_saves_file(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.plot('2d', show=False, save=True)
        assert any(f.suffix == '.png' for f in tmp_path.iterdir())

    def test_distances_saves_file(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.plot('distances', show=False, save=True)
        assert any(f.suffix == '.png' for f in tmp_path.iterdir())

    def test_correlation_saves_file(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.plot('correlation', show=False, save=True)
        assert any(f.suffix == '.png' for f in tmp_path.iterdir())

    def test_unknown_kind_raises(self, basic_result):
        with pytest.raises(ValueError, match="Unknown plot kind"):
            basic_result.plot('nonexistent', show=False)

    def test_plot_no_show_no_save_runs(self, basic_result):
        """plot() with show=False, save=False should not raise."""
        basic_result.plot('pairplot', show=False, save=False)
        basic_result.plot('1d',       show=False, save=False)
        basic_result.plot('distances',show=False, save=False)


class TestExports:

    def test_to_csv_file_exists(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_csv('design.csv')
        assert (tmp_path / 'design.csv').exists()

    def test_to_csv_row_count(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_csv('design.csv')
        df       = pd.read_csv(tmp_path / 'design.csv')
        expected = len(basic_result.samples) + len(basic_result.validation)
        assert len(df) == expected

    def test_to_csv_has_point_type(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_csv('design.csv')
        df = pd.read_csv(tmp_path / 'design.csv')
        assert 'point_type' in df.columns

    def test_to_json_file_exists(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_json('design.json')
        assert (tmp_path / 'design.json').exists()

    def test_to_json_is_valid(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_json('design.json')
        df = pd.read_json(tmp_path / 'design.json')
        assert len(df) > 0

    def test_to_markdown_file_exists(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_markdown('design.md')
        assert (tmp_path / 'design.md').exists()

    def test_to_markdown_contains_banner(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_markdown('design.md')
        content = (tmp_path / 'design.md').read_text()
        assert 'mergen' in content.lower()

    def test_to_latex_file_exists(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_latex('design.tex')
        assert (tmp_path / 'design.tex').exists()

    def test_to_latex_contains_tabular(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_latex('design.tex')
        content = (tmp_path / 'design.tex').read_text()
        assert 'tabular' in content

    def test_to_html_file_exists(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_html('design.html')
        assert (tmp_path / 'design.html').exists()

    def test_to_html_is_valid(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.to_html('design.html')
        content = (tmp_path / 'design.html').read_text()
        assert '<html' in content
        assert '<table' in content

    def test_to_excel_file_exists(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        try:
            basic_result.to_excel('design.xlsx')
            assert (tmp_path / 'design.xlsx').exists()
        except ImportError:
            pytest.skip('openpyxl not installed')

    def test_to_excel_single_sheet(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        try:
            import openpyxl
            basic_result.to_excel('design.xlsx')
            wb = openpyxl.load_workbook(tmp_path / 'design.xlsx')
            assert 'Design' in wb.sheetnames
        except ImportError:
            pytest.skip('openpyxl not installed')


class TestPlotsExtra:

    def test_plot_quality_runs(self, basic_result):
        basic_result.plot('quality', show=False, save=False)

    def test_plot_quality_saves_file(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.plot('quality', show=False, save=True)
        assert any(f.suffix == '.png' for f in tmp_path.iterdir())

    def test_plot_all_runs(self, basic_result, tmp_path):
        basic_result.output_dir = str(tmp_path)
        basic_result.plot('all', show=False, save=True)
        pngs = list(tmp_path.glob('*.png'))
        assert len(pngs) >= 4   # pairplot, 1d, distances, correlation, quality