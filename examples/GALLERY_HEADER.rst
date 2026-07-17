.. _examples_gallery:

Example gallery
===============

Fifteen complete, runnable studies, each built around a realistic
scenario from a different domain: materials processing, CFD, wet-lab
biology, machine-learning hyperparameters, particle phenomenology and
beamline optics, among others. Every page shows the full script, its
output, and download links for the ``.py`` source and a generated
Jupyter notebook.

The lighter examples are executed on every documentation build, so
their outputs are always produced by the code you see. The heavier
studies (04, 05, 12, 14) are rendered without execution to keep the
documentation build fast; the full set is executed regularly in
continuous integration.

To run any example locally::

    python examples/01_quickstart.py

To open one as a Jupyter notebook without downloading it from this
page::

    pip install jupytext
    jupytext --to notebook examples/01_quickstart.py
