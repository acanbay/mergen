:orphan:

.. _examples_gallery:

Examples
========

Fifteen complete, runnable studies, each built around a realistic
scenario from a different domain: materials processing, CFD, wet-lab
biology, machine-learning hyperparameters, particle phenomenology and
beamline optics, among others. Every example is executed on every
documentation build, so each page shows the full script together with
the exact output and figures that script produces, plus download links
for the ``.py`` source and a generated Jupyter notebook.

To run any example from a clone of the repository::

    python examples/01_quickstart.py


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a 30-run design for a thermal deposition study and read its quality evidence.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_01_quickstart_thumb.png
    :alt:

  :doc:`/auto_examples/01_quickstart`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quickstart: a first design</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Mix discrete, continuous, integer, nominal and ordinal factors under one constraint.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_02_parameter_types_thumb.png
    :alt:

  :doc:`/auto_examples/02_parameter_types`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mixed parameter types and a constraint</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compose a reactor design from prescribed runs, a focus region and an exclusion zone.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_03_advanced_design_thumb.png
    :alt:

  :doc:`/auto_examples/03_advanced_design`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Prescribed points, focus and exclusion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep five criteria against a shared Monte Carlo baseline and rank them by percentile.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_04_choosing_criteria_thumb.png
    :alt:

  :doc:`/auto_examples/04_choosing_criteria`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparing optimisation criteria</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run SA, SCE and ESE under a shared budget and weigh final score against elapsed time.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_05_choosing_algorithm_thumb.png
    :alt:

  :doc:`/auto_examples/05_choosing_algorithm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Choosing an optimisation algorithm</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Contrast a default-size and a doubled design of the same study to see what extra runs buy.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_06_sample_size_thumb.png
    :alt:

  :doc:`/auto_examples/06_sample_size`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">How large should a design be?</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Add a hand-picked test set alongside the optimised design and the automatic validation split.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_07_extra_sets_thumb.png
    :alt:

  :doc:`/auto_examples/07_extra_sets`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Extra sets: adding a test set</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Load points from a previous campaign and let run() build only the validation set around them.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_08_resume_existing_thumb.png
    :alt:

  :doc:`/auto_examples/08_resume_existing`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Resuming from an existing design</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Extend, reorder and subsample a base design as an experimental campaign grows in stages.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_09_sequential_workflow_thumb.png
    :alt:

  :doc:`/auto_examples/09_sequential_workflow`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A staged, sequential campaign</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Produce every plot type, every export and the quality report from a single finished design.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_10_outputs_and_reports_thumb.png
    :alt:

  :doc:`/auto_examples/10_outputs_and_reports`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Outputs and reports in one place</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Design an enzyme-activity assay whose buffer type is nominal, scored with the QQ-aware maxproqq.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_11_wetlab_biology_thumb.png
    :alt:

  :doc:`/auto_examples/11_wetlab_biology`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Wet-lab: an assay with a nominal factor</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Design an aerofoil CFD campaign where compare() restricts itself to nominal-capable criteria.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_12_cfd_engineering_thumb.png
    :alt:

  :doc:`/auto_examples/12_cfd_engineering`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">CFD: a sweep with a turbulence model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cover a hyperparameter space with far fewer runs than the full grid would need.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_13_ml_hyperparameters_thumb.png
    :alt:

  :doc:`/auto_examples/13_ml_hyperparameters`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Machine-learning hyperparameters</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cover a five-parameter 2HDM space with a few dozen well-separated benchmark points.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_14_bsm_2hdm_phenomenology_thumb.png
    :alt:

  :doc:`/auto_examples/14_bsm_2hdm_phenomenology`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Particle physics: 2HDM benchmarks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tune magnet and screen settings with a maximin design; no two setups are near-duplicates.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_15_beamline_optics_thumb.png
    :alt:

  :doc:`/auto_examples/15_beamline_optics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Beamline optics: costly settings</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/01_quickstart
   /auto_examples/02_parameter_types
   /auto_examples/03_advanced_design
   /auto_examples/04_choosing_criteria
   /auto_examples/05_choosing_algorithm
   /auto_examples/06_sample_size
   /auto_examples/07_extra_sets
   /auto_examples/08_resume_existing
   /auto_examples/09_sequential_workflow
   /auto_examples/10_outputs_and_reports
   /auto_examples/11_wetlab_biology
   /auto_examples/12_cfd_engineering
   /auto_examples/13_ml_hyperparameters
   /auto_examples/14_bsm_2hdm_phenomenology
   /auto_examples/15_beamline_optics


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
