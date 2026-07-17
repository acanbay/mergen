# Export designs and reports

A finished design is only useful once it leaves Python. Mergen writes
the design table to six formats and prints a numeric quality summary,
all under a single output directory.

## The export formats

```python
result.to_csv('design.csv')          # downstream code, spreadsheets
result.to_json('design.json')        # downstream code, web
result.to_markdown('design.md')      # lab notebook
result.to_latex('design.tex')        # a paper's appendix
result.to_html('design.html')        # notebook or web
result.to_excel('design.xlsx')       # spreadsheet users
```

Each call writes under the result's output directory, creating it if
needed; the directory is reported in the run banner and available as
`result.output_dir`. The comparison object exposes the same interface,
so `comparison.to_markdown(...)` saves the ranked comparison table
itself.

## The printed quality summary

```python
result.quality_report()
```

This prints the metric table with its Monte Carlo baseline and
percentile ranks (see the explanation page on quality metrics for how
to read every row). It is a printed summary rather than a file, meant
for inspection while you decide whether the design is good enough.

## For a paper's methods section

The report plus `to_latex()` gives you everything an appendix needs:
the ranked evidence that the design is space-filling, and the design
table itself in LaTeX. A typical sentence, "a 30-run design was
generated with Mergen (uMaxPro criterion, simulated annealing); its
minimum inter-point distance ranked at the 90th percentile of 300
random reference designs", is assembled entirely from numbers the
report prints.
