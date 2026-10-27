# Using AI assistants with Mergen

Modern AI assistants are good at turning a plain description of an
experiment into a starting point: a parameter space, a sensible
criterion and algorithm, and a runnable script. But no assistant
knows Mergen from its training, so left to itself it will guess at
method names and get them wrong. The template below solves this by
being self-contained: it carries Mergen's real API and its own
decision rules into the conversation. Fill in the bracketed parts
and paste the whole thing into the assistant of your choice.

Treat the answer as a draft, not a verdict. An assistant cannot see
your system respond; when its suggestion matters, verify it the same
way this documentation recommends everywhere else: run
`Sampler.compare()` on the shortlisted options and let the comparison
decide (see {doc}`comparing designs
<how-to/compare_designs>`).

## The template

````text
You are helping me plan a space-filling Design of Experiments with
the Python package Mergen (import name `mergen`, distribution
`mergen-doe`). You do not know this package from training. Use ONLY
the API contract and the decision rules given below; do not invent
any method, argument or plot kind that is not listed here. Before
you answer, check every call in your script against the contract.

MY STUDY
- Goal of the experiment: [describe what you want to learn]
- Parameters and ranges: [name each setting; say whether it is a
  set of values, a continuous range (linear or log), an integer
  range, an ordered category or an unordered category]
- Total run budget: [how many experiments you can afford]
- Constraints: [combinations that are impossible or forbidden;
  single settings that must never be run; regions that deserve
  more runs; runs that already exist; an external test set - or
  "none"]
- What matters most to me: [for example even coverage, good 1D/2D
  projections for a surrogate model, uniformity, or handling
  categorical settings well]

API CONTRACT (use exactly this, nothing else)
- Space:
    space = mergen.ParameterSpace({
        'a': [v1, v2, ...],                        # explicit values
        'b': ('continuous', low, high),            # linear grid
        'c': ('continuous', low, high, 'log'),     # log grid, low > 0
        'd': ('continuous', low, high, {'resolution': 51}),  # grid size
        'e': ('integer', low, high),
        'f': ('nominal', ['A', 'B', 'C']),         # unordered category
        'g': ('ordinal', ['low', 'mid', 'high']),  # ordered category
    })
    space.add_constraint(lambda p: p['b'] + p['e'] <= 10)
        # p maps names to values; return True for allowed points
- Sampler setup (all optional except set_design):
    sampler = mergen.Sampler(space)
    sampler.set_design(n_samples=..., n_validation=...,
                       extra_sets={'test': 5})
    sampler.add_prescribed([[...], ...], in_design=True)
        # runs that must be in the design; in_design=False adds them
        # to the result on top of n_samples
    sampler.load_design(points)
        # take an existing design as is: run() then only adds the
        # validation and extra sets; cannot be combined with
        # add_prescribed, add_focus or add_exclusion
    sampler.add_focus(point, spread=1.0, n_samples=4)
        # extra points around one grid point; spread is the kernel
        # width in grid steps
    sampler.add_exclusion(point)
        # removes this ONE grid point only; forbid a region with
        # space.add_constraint instead
    sampler.add_set('external', points)
        # a test set: no design or validation point may reuse it,
        # and a prescribed point may not be one of its points
    sampler.set_optimizer('sa', n_restarts=5, max_iter=20000)
        # 'sa' and 'sce': n_restarts, max_iter;
        # 'ese': n_restarts, M, J, Q
        # leave these unset unless runtime matters: run() then
        # lengthens the search itself when results do not settle
  Points are lists of numbers in parameter order and must lie on the
  grid. Give a nominal or ordinal level as its position in the level
  list: 0 for the first level, 1 for the second, and so on.
- Run and result:
    result = sampler.run(criteria='umaxpro', algorithm='sa', seed=44)
        # n_repeats='auto' (default) repeats until the best result is
        # found five times or a work limit is reached (then it warns);
        # n_jobs=-1 (default) uses every core
    result.summary(); result.quality_report()
    result.plot(kind, save=True)  # kinds: 'pairplot', '1d', '2d',
        # 'distances', 'correlation', 'quality'
    result.to_csv('design.csv')   # also to_json, to_excel,
        # to_markdown, to_latex, to_html
    result.samples, result.validation, result.sets  # DataFrames
- Comparison sweep:
    comparison = sampler.compare(
        criteria=['umaxpro', 'phi_p'], algorithms=['sa', 'ese'],
        priority=('min_distance', 'max_abs_correlation'),
        seed=44)
    comparison.summary(); comparison.plot(save=True)
    comparison.best, comparison.best_result, comparison.table
    comparison.reselect(priority=..., selection=..., tol=...)
  priority may use only 'min_distance', 'minimax',
  'max_abs_correlation', 'projection_cd2' and 'cd2'.
  selection='pareto' (default) treats them as competing objectives
  and takes the design closest to all of them at their best;
  selection='lexicographic' applies them in order.
  tol={'min_distance': 0.01} makes differences below 0.01 a tie.

MERGEN'S DECISION RULES (follow these)
- Criteria: 'umaxpro' is the default for numeric spaces and
  optimises all lower-dimensional projections; 'maxpro' is its
  classical predecessor; 'phi_p' approximates maximin distance and
  gives the most even point separation; 'cd2' and 'stratified'
  target uniformity in the discrepancy sense. With any nominal
  parameter only 'maxproqq' and 'qqd' are allowed (run() refuses
  the others): 'maxproqq' for projection quality, 'qqd' for
  category-heavy spaces. Ordinal parameters are accepted by every
  criterion, but 'maxproqq' and 'qqd' remain the recommended
  choice for them too.
- Algorithms: 'sa' (simulated annealing with restarts) is the
  robust default; 'sce' and 'ese' are alternatives worth trying in
  a compare() sweep; 'ese' does not support one-parameter spaces.
- Sample size: about 10 runs per parameter is the accepted initial
  rule of thumb; fewer reduces quality, more helps until the budget
  runs out. By default Mergen adds a validation set of 20 percent of
  n_samples on top; these are extra runs, so set n_validation=0 if
  no surrogate model will be fitted.
- Fix the seed so the design can be reproduced.
- When two or more options seem plausible, do not guess: tell me to
  run Sampler.compare() on the shortlist.

WHAT TO PRODUCE
1. A ParameterSpace definition for my study, as Python code,
   including any constraints.
2. A recommended criterion and algorithm with a one-paragraph
   rationale tied to my goals, plus a suggested n_samples and
   n_validation that fit my budget.
3. A complete runnable script using only the API contract above:
   ParameterSpace, Sampler, set_design, run, then
   result.summary(), result.quality_report(),
   result.plot('pairplot', save=True), result.to_csv('design.csv').
4. If my constraints mention forbidden combinations, single
   forbidden settings, prior runs, a test set or focus areas, use
   add_constraint, add_exclusion, add_prescribed / load_design,
   add_set or add_focus accordingly.
5. If you shortlisted several criteria or algorithms, a compare()
   call with a priority that matches what matters most to me.
6. A short note on what to check in the quality report before
   trusting the design: the Rank column compares each metric with
   300 random designs from the same feasible space.
Ask me for any missing information before answering.
````

## Checking the answer

Run the script the assistant returns before you trust any of it. An
`AttributeError` or `TypeError` means it stepped outside the contract;
paste the error back and ask for a corrected script. Once it runs,
read `result.quality_report()`: ranks far below the 50th percentile on
the metrics you care about mean the design is no better than a random
one on those points, whatever the assistant said about the criterion.

## If your assistant can open links

Some assistants can open web links; most cannot, or only in certain
modes. The template assumes nothing and is complete without any
link. If your assistant can browse, append this line to the end of
the template:

```text
If you can open links, read the documentation at
https://mergen.readthedocs.io and prefer it over anything stated
above.
```

## Why the template is structured this way

The API contract exists because assistants do not know Mergen: it
replaces guessing with copying. The decision-rules block hands the
assistant the same guidance a careful reader would take from the
{doc}`criterion <explanation/choosing_criterion>` and
{doc}`algorithm <explanation/choosing_algorithm>` pages, so its
advice follows how Mergen behaves rather than how a similar library
might. The
sample-size rule it carries is the widely used ten-runs-per-parameter
guideline for initial computer experiments (Loeppky, Sacks &
Welch, 2009).

## Reference

Loeppky, J. L., Sacks, J., & Welch, W. J. (2009). Choosing the
sample size of a computer experiment: A practical guide.
*Technometrics*, 51(4), 366-376.
