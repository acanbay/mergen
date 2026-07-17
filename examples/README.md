# Examples

Fifteen complete, runnable studies, each built around a realistic
scenario from a different domain. Every script is self-contained:
run it from a clone of the repository and it prints its summary and
quality report and saves its figures and exports under `outputs/`.

```bash
python examples/01_quickstart.py
```

The documentation renders each example as an executed page, with the
full output and figures inline and downloadable notebook versions.

| Script | Study |
|---|---|
| `01_quickstart.py` | Build a 30-run design for a thermal deposition study and read its quality evidence. |
| `02_parameter_types.py` | Mix discrete, continuous, integer, nominal and ordinal factors under one constraint. |
| `03_advanced_design.py` | Compose a reactor design from prescribed runs, a focus region and an exclusion zone. |
| `04_choosing_criteria.py` | Sweep five criteria against a shared Monte Carlo baseline and rank them by percentile. |
| `05_choosing_algorithm.py` | Run SA, SCE and ESE under a shared budget and weigh final score against elapsed time. |
| `06_sample_size.py` | Contrast a default-size and a doubled design of the same study to see what extra runs buy. |
| `07_extra_sets.py` | Add a hand-picked test set alongside the optimised design and the automatic validation split. |
| `08_resume_existing.py` | Load points from a previous campaign and let `run()` build only the validation set around them. |
| `09_sequential_workflow.py` | Extend, reorder and subsample a base design as an experimental campaign grows in stages. |
| `10_outputs_and_reports.py` | Produce every plot type, every export and the quality report from a single finished design. |
| `11_wetlab_biology.py` | Design an enzyme-activity assay whose buffer type is nominal, scored with the QQ-aware `maxproqq`. |
| `12_cfd_engineering.py` | Design an aerofoil CFD campaign where `compare()` restricts itself to nominal-capable criteria. |
| `13_ml_hyperparameters.py` | Cover a hyperparameter space with far fewer runs than the full grid would need. |
| `14_bsm_2hdm_phenomenology.py` | Cover a five-parameter 2HDM space with a few dozen well-separated benchmark points. |
| `15_beamline_optics.py` | Tune magnet and screen settings with a maximin design; no two setups are near-duplicates. |
