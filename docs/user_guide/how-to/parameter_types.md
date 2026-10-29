# Declare different parameter types

`ParameterSpace` accepts five kinds of parameter. You can mix all of
them in one space, and the value you give each key decides its type.

```python
from mergen import ParameterSpace, Sampler

space = ParameterSpace({
    'flow_rate':   ('continuous', 0.1, 10.0, {'resolution': 25, 'round': 2}),
    'temperature': range(20, 101, 5),
    'n_stages':    ('integer', 1, 20),
    'catalyst':    ('nominal', ['A', 'B', 'C']),
    'grade':       ('ordinal', ['low', 'med', 'high']),
})
```

*Continuous* parameters take a `('continuous', low, high)` tuple. The
optional fourth element controls how the range is discretised onto the
candidate grid: `resolution` sets the number of levels and `round` the
decimal places. A finer grid gives the optimiser more freedom at the
cost of a larger candidate set.

*Discrete numeric* parameters are given as any explicit sequence of
values, such as a list or a `range`. Only these exact values appear in
the design, which is the right choice for settings your equipment
supports at fixed steps.

*Integer* parameters use an `('integer', low, high)` tuple and take
whole-number values across the interval.

*Nominal* parameters, `('nominal', [...])`, are unordered categories:
buffer types, algorithm names, material grades with no natural
ranking. *Ordinal* parameters, `('ordinal', [...])`, are categories
with a meaningful order, and Mergen preserves that order when scoring
distances.

The levels come out of Mergen as the labels you gave, numbers staying
numbers: in `result.samples`, in every export and report, on the plot
axes, and in what a constraint receives. Wherever you pass points to
Mergen (prescribed, focus and excluded points, sets, a loaded design,
the `sequential` functions), give the label, for example
`[300, 'adam']`; a label read back from a CSV file as text, such as
`'32'` for the label 32, is recognised too. A level's position in the
label list (0 for the first) is accepted as well, unless one of the
labels is itself a number, which would make it ambiguous. Two labels
that name the same number (`1` and `1.0`, or `'01'` and `'1'`) are
refused. A label that a CSV reader changes, such as empty or `'NA'`
text, or text with a leading zero like `'01'`, cannot be recognised
after a CSV round trip; give such levels other names.

## Criterion compatibility

Nominal and ordinal factors change how the design must be scored,
because distance between unordered labels is not Euclidean. If your
space contains any nominal factor, use a criterion that supports
qualitative factors, $\mathrm{MaxPro_{QQ}}$ (`'maxproqq'`) or QQD
(`'qqd'`); the other five criteria are
for purely numeric spaces. `mergen.criteria.nominal_supporting_criteria()`
lists the compatible ones, and `Sampler.run` raises an informative
error if the criterion and the space are incompatible. Without a
`criteria` argument, `Sampler.run` uses MaxPro$_{\mathrm{QQ}}$ on a
space with a nominal or ordinal parameter and uMaxPro otherwise.

For a logarithmically spaced parameter, such as a learning rate or a
concentration spanning orders of magnitude, append the `'log'` flag to
a continuous or integer specification and Mergen builds the grid on a
logarithmic ladder:

```python
'learning_rate': ('continuous', 1e-4, 1e-1, 'log'),
'batch_size':    ('integer', 8, 256, 'log'),
```

Alternatively, list explicit levels when you want exact control over
the sampled decades:

```python
'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
```
