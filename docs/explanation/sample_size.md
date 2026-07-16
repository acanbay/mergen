# How many runs?

The sample size is the single most consequential number you give
Mergen, and there is a well-tested starting point.

## The 10 x d guideline

For computer experiments intended for surrogate modelling, Loeppky,
Sacks & Welch (2009) examined prediction accuracy across a large body
of test problems and concluded that a sample size of about ten times
the number of input dimensions is a reasonable rule of thumb: it is
usually enough for a Gaussian-process surrogate to capture the main
structure, and when it is not, the shortfall is typically large enough
that a moderately bigger one-shot design would not have helped either.
Mergen adopts $n = 10 d$ as the reference point and will warn, not
refuse, when you configure a design far below it.

## When to go above it

Constraints and exclusions shrink the feasible region and can make it
geometrically awkward; representing an awkward region takes more
points than representing a box. Focus regions consume part of the
budget locally by design. Nominal factors multiply the work: each
level combination that matters to you must actually occur in the
design, so with many levels the qualitative structure alone can
dictate a floor on $n$. Finally, if the response is noisy and you plan
replication, the space-filling budget is only the unique-location part
of the total.

## When to go below it

Screening studies, where the goal is to rank the importance of inputs
rather than to model the response, tolerate smaller designs. Pilot
runs that will later be extended are also a legitimate reason: because
Mergen's `sequential.extend` preserves existing runs while adding
space-filling points, a small design now is not wasted budget later.

## The validation set

By default `set_design` reserves an additional 20 percent of
`n_samples` as validation points, chosen to be maximally distant from
the design proper. These are honest held-out locations for testing a
surrogate fit. They are extra runs, so account for them in the total
budget, or set `n_validation=0` when no surrogate is planned.

## Practical advice

Start at $10 d$, build the design, and read the quality report before
spending any laboratory or compute time. If the report is weak in the
aspects you care about, a larger $n$ is one lever, but often the
cheaper lever is a better-suited criterion. And because designs can be
extended without discarding completed runs, err on the side of
starting smaller and growing: the sequential how-to guide shows the
workflow.

## References

Loeppky, J. L., Sacks, J., & Welch, W. J. (2009). Choosing the sample
size of a computer experiment: A practical guide. *Technometrics*,
51(4), 366-376.
