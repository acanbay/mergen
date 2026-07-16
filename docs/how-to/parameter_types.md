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

## Criterion compatibility

Nominal and ordinal factors change how the design must be scored,
because distance between unordered labels is not Euclidean. If your
space contains any nominal factor, use a criterion that supports
qualitative factors, `maxproqq` or `qqd`; the other five criteria are
for purely numeric spaces. `mergen.nominal_supporting_criteria()`
lists the compatible ones, and `Sampler.run` raises an informative
error if the criterion and the space are incompatible.

For a logarithmically spaced parameter, such as a learning rate or a
concentration spanning orders of magnitude, list the levels directly
on the ladder you want:

```python
'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
```

This gives exact control over the sampled decades without a separate
parameter type.
