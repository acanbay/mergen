# Worked domain examples

The repository's `examples/` directory contains complete, runnable
scripts that apply Mergen end to end in several fields. Each is
self-contained and produces a design, a quality report, and plots.
They are the fastest way to see the whole workflow on a realistic
problem close to your own.

## Wet-lab biology

An enzyme-assay design over pH, incubation temperature, buffer type,
and substrate concentration. Buffer type is a nominal factor, so the
design is scored with `maxproqq`; a held-out validation set is
reserved for checking the fitted response on unseen conditions. See
`examples/11_wetlab_biology.py`.

## CFD engineering

A computational-fluid-dynamics parameter study on a purely numeric,
continuous space, illustrating a projection-based criterion where any
subset of the inputs may drive the response. See
`examples/12_cfd_engineering.py`.

## Machine-learning hyperparameters

A hyperparameter design over learning rate, batch size, optimiser, and
weight decay. The learning rate is placed on a logarithmic ladder of
explicit levels and the optimiser is a nominal factor, so a QQ-type
criterion is used. See `examples/13_ml_hyperparameters.py`.

## High-energy physics phenomenology

A parameter scan of a beyond-the-Standard-Model two-Higgs-doublet
model, the kind of high-dimensional, constrained scan for which
space-filling coverage is essential. See
`examples/14_bsm_2hdm_phenomenology.py`.

## Accelerator beamline optics

A beamline-optics tuning design, showing Mergen on an
instrument-configuration problem. See
`examples/15_beamline_optics.py`.

## Running them

Each script runs directly:

```bash
python examples/11_wetlab_biology.py
```

The lower-numbered scripts (`01` through `10`) build up the same
features tutorial by tutorial, from the quickstart to full export and
reporting.
