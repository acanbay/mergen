# Design philosophy

Mergen makes a small number of structural decisions that shape
everything the user sees. This page states them and gives the
reasoning, so that the package's behaviour is predictable rather than
mysterious.

## Designs live on a discrete candidate grid

Every parameter, including continuous ones, is represented by a finite
set of levels; a continuous range is discretised into a fine grid at
definition time. The design is then a selection of points from the
Cartesian product of these level sets.

This choice buys three things. First, fidelity to reality: discrete
stocks (four catalyst concentrations, integer batch counts, named
instrument settings) are represented exactly, never rounded after the
fact. Second, exactness of the optimisation moves: exchanging one grid
level for another is a well-defined, reversible step, which is what
lets the optimisers explore systematically and lets incremental
criterion updates stay numerically consistent with full evaluations.
Third, reproducibility: a design is a set of exact grid coordinates,
identical on every machine.

The grid can be astronomically large; a candidate set is never
enumerated. Mergen indexes the grid through a mixed-radix bijection
between integers and coordinate tuples, so a candidate point is
materialised only when the optimiser looks at it, and memory use is
independent of grid size.

## The Latin hypercube backbone

Initial designs are Latin hypercube samples (McKay, Beckman & Conover,
1979): each parameter's levels are stratified so that every
one-dimensional projection of the starting design is already evenly
spread. The optimisers then improve the arrangement under the chosen
criterion. Starting from an LHS rather than from uniform noise means
the optimiser spends its budget on the hard part, joint structure,
rather than on repairing marginal clumping.

## Criteria and optimisers are orthogonal

Any criterion can be paired with any optimiser. Criteria implement a
single evaluation interface (with an incremental fast path for
single-point updates), and optimisers are written against that
interface, never against a specific criterion. This is what makes
`compare()` possible: the criterion by algorithm sweep is a genuine
Cartesian product, not a set of special cases. The repeated
optimisations inside a sweep are independent, so they parallelise
across cores (`n_jobs`) without changing the result.

## Design time only, surrogate free

Mergen deliberately stops where the simulation output begins. It looks
only at parameter-space geometry: no expected improvement, no Bayesian
optimisation, no adaptive sampling driven by observed responses. Those
methods belong to surrogate-modelling packages; keeping them out keeps
Mergen's guarantees checkable. The sequential utilities (extending a
design, sub-sampling a pool, run ordering, nested designs) follow the
same rule: they operate on geometry, so they compose with any
downstream modelling tool rather than competing with it.

## Evidence over assertion

A design is not done when the optimiser stops; it is done when its
quality is demonstrated. Every result carries a quality report ranked
against a Monte Carlo baseline drawn at runtime from the same feasible
space, so the evidence remains valid under constraints, exclusions,
and focus regions. The same philosophy extends to the package itself:
beyond unit tests, Mergen's validation suite checks the criterion
implementations against independent reference implementations,
published numerical values, and closed-form cases.

## References

McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of
three methods for selecting values of input variables in the analysis
of output from a computer code. *Technometrics*, 21(2), 239-245.
