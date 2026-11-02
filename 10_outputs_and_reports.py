"""
Outputs and reports in one place
================================

Produce every plot type, every export and the quality report from a single finished design.

One finished design usually has to be communicated to several
audiences at once: slides want images, a lab notebook wants a readable
table, a paper wants LaTeX, and a downstream script wants raw data.
This example takes a single design and produces the full range of
Mergen's plots and export formats, so each audience gets the artefact
that suits it. It is the one example that deliberately generates
everything; every other example stays minimal.

Parameters
----------
- factor_a, factor_b, factor_c (0.0-1.0, continuous, 15-level grid,
  rounded to 3 decimals): three generic normalised inputs, enough
  dimensions for an interesting pairplot and correlation view. The
  moderate grid lets the optimiser reach a high-quality design at its
  full default effort while still running quickly.

What to look at
---------------
- The saved plots (one PNG per type): pairplot for coverage, 1d for
  per-factor spread, 2d for a single pair, distances for the pairwise
  profile, correlation for pairwise independence, and quality for the
  metric percentiles. Any of these can go straight onto a slide.
- ``quality_report()`` (printed): the numeric quality summary.
- The six export files, each for a different consumer: PNG for slides,
  Markdown and HTML for a lab notebook, LaTeX for a paper, CSV and JSON
  for downstream code.

Mergen features used
--------------------
- ``criteria='phi_p'``: a maximin criterion chosen here because it gives
  strong, balanced percentiles across all six quality metrics, so the
  quality plot reads well as a showcase.
- ``result.plot('all', save=True)``: render every plot type at once.
- Every export format: to_csv, to_json, to_markdown, to_latex,
  to_html, to_excel.
- ``result.quality_report()`` for the printed numeric summary.

Estimated runtime: a few seconds to a minute.
"""
# sphinx_gallery_thumbnail_number = 5
from mergen import ParameterSpace, Sampler

# 1. Define a three-factor space and build one design.
space = ParameterSpace({
    'factor_a': ('continuous', 0.0, 1.0, {'resolution': 15, 'round': 3}),
    'factor_b': ('continuous', 0.0, 1.0, {'resolution': 15, 'round': 3}),
    'factor_c': ('continuous', 0.0, 1.0, {'resolution': 15, 'round': 3}),
})
sampler = Sampler(space)
sampler.set_design(n_samples=30)
result = sampler.run(criteria='phi_p')

# 2. Printed numeric summary.
result.quality_report()

# 3. Every plot type, saved as PNG (slides).
result.plot('all', save=True)

# 4. Every export format, each for a different audience.
result.to_csv('design.csv')          # downstream code
result.to_json('design.json')        # downstream code
result.to_markdown('design.md')      # lab notebook
result.to_latex('design.tex')        # paper
result.to_html('design.html')        # lab notebook / web
result.to_excel('design.xlsx')       # spreadsheet users
