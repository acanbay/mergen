# What the quality metrics mean

Every `quality_report()` evaluates the design on six geometric metrics
plus the optimised criterion, and ranks each against a Monte Carlo
baseline. This page defines the metrics precisely and explains what
each one can and cannot tell you. Coordinates are normalised to
$[0,1]^d$ throughout, and $d_{ij}$ denotes the Euclidean distance
between design points $x_i$ and $x_j$.

## The baseline and the rank

Raw metric values are scale-dependent and, on their own, impossible to
judge. The report therefore generates 300 random designs of the same
size from the same feasible space (respecting all constraints and
exclusions) and computes every metric for each. The *Rank* column is
direction-adjusted: it is the share of these random designs that your
design outperforms on that metric, so higher is always better. A
design at the 90th percentile on minimum distance beats 90 percent of
random attempts; one at the 6th percentile beats almost none.

Because the baseline is generated at runtime from your actual space,
the comparison remains honest under constraints: a heavily constrained
space is harder for the random designs too.

## Min distance ($d_{\min}$)

$$
\min_{i<j} d_{ij} \qquad \text{(higher is better)}
$$

The maximin quantity (Johnson, Moore & Ylvisaker, 1990): the distance
between the two closest runs. It answers "am I wasting budget on
near-duplicate experiments?" and is the metric most directly targeted
by `phi_p`.

## Minimax distance ($d_{\mathrm{mM}}$)

$$
\max_{x \in \mathcal{X}} \min_{i} \lVert x - x_i \rVert_2
\qquad \text{(lower is better)}
$$

The covering radius (Johnson, Moore & Ylvisaker, 1990): the distance
from the most neglected point of the space to its nearest design
point, estimated over a dense reference set. It answers "how far can
the truth hide from my nearest run?" Low minimax means no region is
left unwatched.

## Max correlation ($|\rho|_{\max}$)

$$
\max_{k \neq l} \; \bigl| \operatorname{corr}(X_{\cdot k},
X_{\cdot l}) \bigr| \qquad \text{(lower is better)}
$$

The largest absolute pairwise Pearson correlation between design
columns. Correlated columns make the effects of two inputs partially
indistinguishable in any subsequent model fit; near-zero correlation
keeps them separable.

## 2D projection discrepancy ($\mathrm{CD}_2^{\mathrm{proj}}$)

The centred $L_2$-discrepancy (Hickernell, 1998) averaged over all
$\binom{d}{2}$ two-dimensional projections of the design (lower is
better). Full-dimensional uniformity does not guarantee that pairs of
inputs look uniform together; this metric checks exactly that, and it
is the geometric face of what the projection-based criteria optimise.

## CV of distances ($\mathrm{CV}_d$)

$$
\operatorname{CV} = \frac{\operatorname{sd}(\{d_{ij}\})}
{\operatorname{mean}(\{d_{ij}\})}
\qquad \text{(lower is better)}
$$

The coefficient of variation of all pairwise distances. A low value
means the distance spectrum is narrow, the signature of a regular,
mesh-like arrangement; a high value indicates a mix of tight pairs and
long gaps.

## Mean distance ($\bar{d}$)

The average of all pairwise distances (higher is better). A coarse but
useful summary of overall spread. It should be read together with the
other metrics: a design can inflate mean distance by pushing points
into corners while leaving the interior empty.

## The criterion row

The final row reports the optimised criterion itself against the same
baseline. This is the most direct measurement of optimisation success,
and its scale is meaningful only relative to the baseline, which is
why both numbers are always printed. If this row ranks high while a
geometric metric you care about ranks low, the message is not that the
optimiser failed but that you optimised a criterion that does not
serve that metric; see the page on choosing a criterion.

## References

Hickernell, F. J. (1998). A generalized discrepancy and quadrature
error bound. *Mathematics of Computation*, 67(221), 299-322.

Johnson, M. E., Moore, L. M., & Ylvisaker, D. (1990). Minimax and
maximin distance designs. *Journal of Statistical Planning and
Inference*, 26(2), 131-148.
