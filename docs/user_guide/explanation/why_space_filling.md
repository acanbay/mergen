# Why space-filling designs?

Suppose you study a system with several adjustable settings: a
reactor with a temperature, a pressure, and a choice of catalyst, or
a machine learning model with a learning rate and a batch size.
Every combination of settings is one possible experiment, and
together the combinations form the *parameter space* of the study.
Trying them all is impossible; a realistic budget is a few dozen
runs against millions of combinations.

A few dozen can nevertheless be enough, because the goal is not to
test every combination but to learn how the system behaves. From the
results of well-placed runs one fits a model of the response,
whether a simple fit, a Gaussian process, or a machine learning
regressor, and that model then predicts the outcome at all the
combinations that were never run. How trustworthy those predictions
are is decided largely before any experiment happens: by where the
runs were placed. Runs that cluster teach the model the same thing
twice; regions with no runs are regions where the model has no
information.

Which runs, then? The intuitive answers all waste budget. Varying
one setting at a time never reveals how settings interact. A regular
grid spends most of its runs repeating the same few values of each
setting. Random picking clusters some runs together and leaves other
regions untouched. The established remedy is a *space-filling
design*: choose the runs so that they spread evenly through the
parameter space, every region is represented, and no two runs
duplicate each other's information. The idea goes back to Latin
hypercube sampling (McKay, Beckman & Conover, 1979) and is the
standard opening move in the design and analysis of computer
experiments (Santner, Williams & Notz, 2018).

Mergen builds such designs, and it builds them for the spaces real
studies actually have: with combinations that are infeasible or
forbidden, with runs that were already performed and must be kept,
with critical regions that deserve extra attention, and with
settings that are categories rather than numbers. And because a
design you cannot defend is not finished, every design leaves the
package with a statistical quality report: your experiment plan, and
the evidence that it was well chosen, as one object.

## References

McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison
of three methods for selecting values of input variables in the
analysis of output from a computer code. *Technometrics*, 21(2),
239-245.

Santner, T. J., Williams, B. J., & Notz, W. I. (2018). *The design
and analysis of computer experiments* (2nd ed.). Springer.
