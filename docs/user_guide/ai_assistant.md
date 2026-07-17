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
`Sampler.compare()` on the shortlisted options and let the ranked
table decide (see {doc}`comparing designs
<how-to/compare_designs>`).

## The template

````text
You are helping me plan a space-filling Design of Experiments with
the Python package Mergen (import name `mergen`, distribution
`mergen-doe`). You do not know this package from training. Use ONLY
the API contract and the decision rules given below; do not invent
any method, argument or plot kind that is not listed here.

MY STUDY
- Goal of the experiment: [describe what you want to learn]
- Parameters and ranges: [name each setting; say whether it is a
  set of values, a continuous range (linear or log), an integer
  range, or a category]
- Total run budget: [how many experiments you can afford]
- Constraints: [combinations that are impossible or forbidden;
  regions that matter more; runs that already exist - or "none"]
- What matters most to me: [for example even coverage, good 1D/2D
  projections for a surrogate model, uniformity, or handling
  categorical settings well]

API CONTRACT (use exactly this, nothing else)
- Space:
    space = mergen.ParameterSpace({
        'name': [v1, v2, ...],                  # discrete values
        'name': ('continuous', low, high),       # linear range
        'name': ('continuous', low, high, 'log'),# log-spaced range
        'name': ('integer', low, high),
        'name': ('nominal', 'a', 'b', ...),      # unordered category
        'name': ('ordinal', 'low', 'mid', ...),  # ordered category
    })
- Sampler setup (all optional except set_design):
    sampler = mergen.Sampler(space)
    sampler.set_design(n_samples=..., n_validation=...)
    sampler.add_prescribed(points)      # keep existing runs
    sampler.load_design(points)         # resume a full design
    sampler.add_focus(region, n=...)    # extra points in a region
    sampler.add_exclusion(region)       # forbid a region
    sampler.set_optimizer('sa', ...)    # tune budget if asked
- Run and result:
    result = sampler.run(criteria='umaxpro', algorithm='sa', seed=44)
    result.summary(); result.quality_report()
    result.plot(kind, save=True)  # kinds: 'pairplot','1d','2d',
        # 'distances','correlation','quality'
    result.to_csv()
- Comparison sweep:
    comparison = sampler.compare(criteria=[...], algorithms=[...])
    comparison.summary(); comparison.plot(save=True)

MERGEN'S DECISION RULES (follow these)
- Criteria: 'umaxpro' is the default for continuous or mixed
  numeric spaces and optimises all lower-dimensional projections;
  'maxpro' is its classical predecessor; 'phi_p' approximates
  maximin distance and gives the most even point separation; 'cd2'
  and 'stratified' target uniformity in the discrepancy sense;
  'maxproqq' extends projection quality to spaces with nominal or
  ordinal parameters; 'qqd' is designed for category-heavy spaces.
  If the space has nominal or ordinal parameters, only recommend
  criteria that support them ('maxproqq', 'qqd').
- Algorithms: 'sa' (simulated annealing with restarts) is the
  robust default; 'sce' and 'ese' are alternatives worth trying in
  a compare() sweep; 'ese' does not support one-dimensional spaces.
- Sample size: about 10 runs per parameter is the accepted initial
  rule of thumb; fewer reduces quality, more helps until the budget
  runs out. Mergen also reserves a validation set by default.
- When two or more options seem plausible, do not guess: tell me to
  run Sampler.compare() on the shortlist.

WHAT TO PRODUCE
1. A ParameterSpace definition for my study, as Python code.
2. A recommended criterion and algorithm with a one-paragraph
   rationale tied to my goals, plus a suggested n_samples.
3. A complete runnable script using only the API contract above:
   ParameterSpace, Sampler, set_design, run, then
   result.summary(), result.quality_report(),
   result.plot('pairplot', save=True), result.to_csv().
4. If my constraints mention forbidden regions, prior runs or
   focus areas, use add_exclusion, add_prescribed / load_design or
   add_focus accordingly.
5. A short note on what to check in the quality report before
   trusting the design.
Ask me for any missing information before answering.
````

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
advice cannot drift away from how Mergen actually behaves. The
sample-size rule it carries is the widely used ten-runs-per-parameter
guideline for initial computer experiments (Loeppky, Sacks &
Welch, 2009).

## Reference

Loeppky, J. L., Sacks, J., & Welch, W. J. (2009). Choosing the
sample size of a computer experiment: A practical guide.
*Technometrics*, 51(4), 366-376.
