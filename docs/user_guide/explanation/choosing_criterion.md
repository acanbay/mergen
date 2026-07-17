# Choosing a criterion

Every Mergen design is the answer to an optimisation problem, and the
criterion is the question. Seven criteria are available, and they do
not agree on what a good design is. This page explains what each one
optimises, gives its mathematical definition, and ends with practical
guidance. Throughout, the design is $D = \{x_1, \dots, x_n\} \subset
[0,1]^d$ in normalised coordinates, and every criterion is minimised.

## Three families

Space-filling criteria fall into three families, each protecting
against a different failure mode.

*Distance-based* criteria look at how close design points get to each
other. They protect against clustered runs that duplicate information.

*Uniformity (discrepancy) criteria* compare the empirical distribution
of the points with the uniform distribution. They protect against
systematic density imbalance: regions that are over- or
under-represented even when no two points are close.

*Projection-based* criteria demand that the design remain
space-filling not only in the full $d$-dimensional space but in every
lower-dimensional projection. They protect against the common
situation where only a subset of the inputs turns out to matter: a
design that is excellent in $d$ dimensions can collapse onto few
distinct values when projected onto the two inputs that drive the
response.

| Criterion | `criteria=` | Family | Qualitative factors | Reference |
|---|---|---|---|---|
| $\phi_p$ | `'phi_p'` | distance | no | Morris & Mitchell (1995) |
| $\mathrm{CD}_2$ | `'cd2'` | uniformity | no | Hickernell (1998) |
| MaxPro | `'maxpro'` | projection | no | Joseph, Gul & Ba (2015) |
| uMaxPro | `'umaxpro'` | projection | no | Vorechovsky & Masek (2026) |
| $\mathrm{SL}_2$ | `'stratified'` | uniformity | no | Tian & Xu (2025) |
| $\mathrm{MaxPro_{QQ}}$ | `'maxproqq'` | projection | yes | Joseph, Gul & Ba (2020) |
| QQD | `'qqd'` | uniformity | yes | Zhang, Yang & Zhou (2021) |

## $\phi_p$

$$
\phi_p(D) = \Big( \sum_{i<j} d_{ij}^{-p} \Big)^{1/p},
\qquad d_{ij} = \lVert x_i - x_j \rVert_2,
$$

with $p = 15$ by default (selected with `criteria='phi_p'`). As $p \to \infty$, minimising $\phi_p$
becomes equivalent to maximising the minimum inter-point distance, the
classical maximin objective; finite $p$ keeps the objective smooth
enough to optimise while concentrating almost all weight on the
closest pair (Morris & Mitchell, 1995).

Use it when physical spacing is the requirement: hardware limits on
how close two operating points may be, or a worst-case guarantee that
no two runs are near-duplicates. Its known weakness is projections: a
maximin-optimal design can still project poorly onto subsets of the
inputs.

## $\mathrm{CD}_2$

The centred $L_2$-discrepancy (Hickernell, 1998) measures the
worst-case mismatch, in an $L_2$ sense over all corner-anchored boxes,
between the fraction of design points in a box and its volume. It has
the closed form

$$
\mathrm{CD}^2(D) = \Big(\tfrac{13}{12}\Big)^{d}
- \frac{2}{n}\sum_{i=1}^{n}\prod_{k=1}^{d}
  \Big(1 + \tfrac12 |x_{ik}-\tfrac12| - \tfrac12 |x_{ik}-\tfrac12|^2\Big)
+ \frac{1}{n^2}\sum_{i,j=1}^{n}\prod_{k=1}^{d}
  \Big(1 + \tfrac12 |x_{ik}-\tfrac12| + \tfrac12 |x_{jk}-\tfrac12|
       - \tfrac12 |x_{ik}-x_{jk}|\Big),
$$

and Mergen reports $\mathrm{CD} = \sqrt{\mathrm{CD}^2}$
(selected with `criteria='cd2'`).

Use it when overall uniformity is the goal, when results must be
comparable with the uniform-design literature, or when integration-type
quantities will be estimated from the runs. It is also the criterion
whose implementation can be cross-checked against an independent one
in SciPy, which Mergen's validation suite does.

## MaxPro

$$
\psi(D) = \sum_{i<j} \prod_{l=1}^{d} (x_{il} - x_{jl})^{-2}.
$$

MaxPro (`criteria='maxpro'`): because the product runs over every
coordinate, a single pair of
points sharing a value in any one coordinate makes that pair's term
explode. The criterion therefore forces all $n$ points to remain
distinct in every one-dimensional projection and, as Joseph, Gul & Ba
(2015) show, promotes space-filling in all $\binom{d}{s}$ projections
simultaneously. Mergen reports the raw pairwise sum, a strictly
monotone transform of the published $\big(\psi / \binom{n}{2}
\big)^{1/d}$ form, so the optimum is identical.

Use it when a surrogate model will be fitted and you do not know in
advance which inputs matter: projection quality is then the property
that matters most. Its known artefact is a mild repulsion of points from the
region near the boundary.

## uMaxPro

uMaxPro (`criteria='umaxpro'`; Vorechovsky & Masek, 2026) keeps
the MaxPro construction but
replaces every squared coordinate difference by its periodic
counterpart,

$$
\delta_{ijl} = \min\big(|x_{il} - x_{jl}|,\; 1 - |x_{il} - x_{jl}|\big),
\qquad
\psi_u(D) = \sum_{i<j} \prod_{l=1}^{d} \delta_{ijl}^{-2},
$$

which treats each axis as a circle. This removes the boundary
repulsion of plain MaxPro and makes the criterion invariant under
cyclic shifts of the design, a property Mergen's validation suite
checks explicitly. It is the default criterion in `Sampler.run`.

Use it in the same situations as MaxPro; prefer it when uniform
treatment of the boundary matters, which is most of the time.

## $\mathrm{SL}_2$ (stratified)

The stratified $L_2$-discrepancy $\mathrm{SL}_2$
(`criteria='stratified'`; Tian & Xu, 2025) evaluates
uniformity across a nested hierarchy of grids: at depth $i$ each axis
is cut into $s^i$ equal strata, and the criterion aggregates, over
depths $i = 0, \dots, p$ and all axes, how evenly the points occupy
the strata:

$$
\mathrm{SD}^2(D) = -\Big(\sum_{i=0}^{p} w_i\, s^{-2i}\Big)^{d}
+ \frac{1}{n^2} \sum_{a,b=1}^{n} \prod_{j=1}^{d}
  \Big(\sum_{i=0}^{p} w_i\, s^{-i}\,
       \delta_i(x_{aj}, x_{bj})\Big),
$$

where $\delta_i(t, z) = \mathbf{1}\{\lfloor s^i t \rfloor = \lfloor
s^i z \rfloor\}$ indicates shared stratum membership. Defaults:
$s = 2$, depth $p = \lfloor \log_s n \rfloor$, and weights chosen
automatically (constant for moderate $d$, exponentially decaying for
$d \ge 8$).

Use it when multi-resolution balance is the goal: designs that must
look balanced both coarsely and finely, comparisons with stratified or
orthogonal-array based constructions, or grid-refinement studies.

## $\mathrm{MaxPro_{QQ}}$

$\mathrm{MaxPro_{QQ}}$ (`criteria='maxproqq'`; Joseph, Gul & Ba,
2020) extends MaxPro to designs that mix
continuous, discrete numeric, integer, ordinal, and nominal factors.
Continuous coordinates contribute $(x_{il} - x_{jl})^2$ as before;
a discrete numeric or ordinal factor with $m_k$ levels contributes
$(|u_{ik} - u_{jk}| + 1/m_k)^2$; a nominal factor with $L_h$ levels
contributes $(\mathbf{1}(v_{ih} \neq v_{jh}) + 1/L_h)^2$, so two runs
sharing a nominal level are penalised but the term cannot vanish. When
all factors are continuous, MaxProQQ reduces exactly to MaxPro, which
Mergen's test suite verifies.

Use it whenever the space contains nominal or ordinal parameters and
projection quality is the priority. It is one of the two criteria that
accept qualitative factors.

## QQD

The qualitative-quantitative discrepancy QQD (`criteria='qqd'`;
Zhang, Yang & Zhou, 2021) is
the uniformity counterpart for mixed spaces: quantitative columns are
scored with the wrap-around $L_2$ kernel and nominal columns with a
discrete kernel, calibrated so that neither factor type dominates.
When no qualitative factors are present, QQD reduces to the
wrap-around $L_2$-discrepancy of the quantitative sub-design, which
the test suite also verifies. Mergen reports $\mathrm{QQD}^2$, the
quantity used throughout the original paper.

Use it for mixed spaces when the goal is uniformity rather than
projections, for example when categorical treatment combinations must
be represented evenly.

## Two criteria, two signatures

The same space (two continuous parameters, 15 runs, identical seed)
optimised under two different criteria makes the family differences
concrete:

::::{grid} 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/img/contrast_phi_p.png
:alt: Design optimised under phi_p.

$\phi_p$: points pushed apart, near-perfect spacing.
```
:::

:::{grid-item}
```{figure} ../_static/img/contrast_umaxpro.png
:alt: Design optimised under uMaxPro.

uMaxPro: spacing traded for projection structure.
```
:::

::::

The quality reports carry the same signatures: each design ranks high
on the metrics its criterion serves and lower elsewhere.

::::{grid} 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/img/contrast_phi_p_quality.png
:alt: Quality chart of the phi_p design.

$\phi_p$: distance metrics ($d_{\min}$, $\mathrm{CV}_d$, $\bar d$)
dominate.
```
:::

:::{grid-item}
```{figure} ../_static/img/contrast_umaxpro_quality.png
:alt: Quality chart of the uMaxPro design.

uMaxPro: a different profile from the same space and budget.
```
:::

::::

Neither picture is "better"; they answer different questions. That is
the reason the criterion is a user decision rather than a constant.

## Practical guidance

Start from the structure of your space. If it contains nominal or
ordinal parameters, the choice is between $\mathrm{MaxPro_{QQ}}$
(projection quality, the usual choice for surrogate modelling) and
QQD (uniformity of coverage).

For purely numeric spaces: if you will fit a surrogate and do not know
which inputs matter, use uMaxPro, the default, or MaxPro if you
specifically want the published non-periodic form. If the requirement
is physical spacing or a worst-case distance guarantee, use $\phi_p$.
If the requirement is distributional uniformity or comparability with
the uniform-design literature, use $\mathrm{CD}_2$; if it is balance
across a hierarchy of resolutions, use $\mathrm{SL}_2$.

When requirements conflict or you are unsure, do not guess:
`mergen.compare()` runs the candidate criteria (and optimisers) on
your actual space and ranks the results on all quality metrics at
once. The how-to guide on comparing designs shows the workflow.

## References

Hickernell, F. J. (1998). A generalized discrepancy and quadrature
error bound. *Mathematics of Computation*, 67(221), 299-322.

Joseph, V. R., Gul, E., & Ba, S. (2015). Maximum projection designs
for computer experiments. *Biometrika*, 102(2), 371-380.

Joseph, V. R., Gul, E., & Ba, S. (2020). Designing computer
experiments with multiple types of factors: The MaxPro approach.
*Journal of Quality Technology*, 52(4), 343-354.

Morris, M. D., & Mitchell, T. J. (1995). Exploratory designs for
computational experiments. *Journal of Statistical Planning and
Inference*, 43(3), 381-402.

Tian, Y., & Xu, H. (2025). A stratified L2-discrepancy with
application to space-filling designs. *Journal of the Royal
Statistical Society, Series B*.

Vorechovsky, M., & Masek, J. (2026). Uniform maximum projection
designs for computer experiments. *Computers & Structures*.

Zhang, M., Yang, F., & Zhou, Y.-D. (2021). Uniformity criterion for
designs with both qualitative and quantitative factors. *Statistics*,
55(1), 90-109.
