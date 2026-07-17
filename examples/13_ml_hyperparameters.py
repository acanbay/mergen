"""
Machine-learning hyperparameters
================================

Cover a hyperparameter space with far fewer runs than the full grid would need.

A hyperparameter design is built for training a model over four knobs:
learning rate, batch size, optimiser, and weight decay. Instead of a
full grid (which explodes combinatorially) or random search (which
clusters and leaves gaps), a space-filling design covers the
configuration space evenly with a modest number of runs. The optimiser
is a nominal factor, so the design is scored with maxproqq. A fixed
baseline configuration the team always wants to include is attached as
its own set.

Parameters
----------
- learning_rate (discrete decades: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
  sampled on a log-like ladder rather than linearly, as is standard for
  learning rates.
- batch_size (discrete powers of two: 16, 32, 64, 128, 256): the usual
  hardware-friendly choices.
- optimiser (nominal: 'adam', 'sgd', 'rmsprop'): unordered categorical.
- weight_decay (discrete: 0.0, 1e-4, 1e-3, 1e-2): common regularisation
  strengths.

What to look at
---------------
- ``summary()``: the space-filling configurations plus the always-included
  baseline set; note how few runs cover the space compared with a full
  grid (5 x 5 x 3 x 4 = 300 combinations).
- The saved pairplot: even coverage across the numeric knobs, with all
  three optimisers visited; the baseline configuration appears in its
  own colour.
- ``configs.json``: the design as JSON, the natural format when each row is
  a configuration consumed directly by a training script.

Mergen features used
--------------------
- Log-like discrete numeric factors alongside a nominal factor.
- criteria='``maxproqq``' as the correct choice for the nominal optimiser.
- ``Sampler.add_set()``: pin a fixed baseline configuration that must
  always be part of the design.
- to_json export for configurations consumed by code.

Estimated runtime: a few seconds to a minute.
"""
from mergen import ParameterSpace, Sampler

# 1. Define the hyperparameter space.
space = ParameterSpace({
    'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'batch_size':    [16, 32, 64, 128, 256],
    'optimiser':     ('nominal', ['adam', 'sgd', 'rmsprop']),
    'weight_decay':  [0.0, 1e-4, 1e-3, 1e-2],
})

# 2. Always include a known-good baseline configuration.
sampler = Sampler(space)
sampler.add_set('baseline',
                [[1e-3, 32, 'adam', 1e-4]],
                color='#ff8800')

# 3. Build the space-filling design with maxproqq (nominal optimiser).
sampler.set_design(n_samples=20, n_validation=4)
result = sampler.run(criteria='maxproqq')

# 4. Inspect and export the configurations for the training script.
result.summary()
result.plot('pairplot', save=True)
result.to_json('configs.json')
