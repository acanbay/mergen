"""Generate every figure used in the documentation.

Run from the repository root:

    python docs/generate_docs_images.py

Writes PNG files into docs/_static/img/. All runs are seeded, so the
figures are reproducible; total runtime is a few minutes.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("MERGEN_SILENT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import mergen

IMG = Path(__file__).resolve().parent / "_static" / "img"
IMG.mkdir(parents=True, exist_ok=True)


def _grab(result, kind: str, dest: str) -> None:
    """Save a plot and move the newest matching PNG into docs/_static/img."""
    result.plot(kind, save=True, show=False)
    outdir = Path(result.output_dir)
    newest = max(outdir.glob("*.png"), key=lambda p: p.stat().st_mtime)
    shutil.copy(newest, IMG / dest)
    print(f"  wrote {IMG / dest}")


def tutorial_design():
    """The design of the two tutorials (seed 44): pairplot, quality, 1d."""
    print("[1/3] tutorial design (umaxpro, sa, seed 44)")
    space = mergen.ParameterSpace({
        'temperature': ('continuous', 300, 500),
        'pressure':    ('continuous', 1.0, 5.0),
        'catalyst':    [0.1, 0.2, 0.5, 1.0],
    })
    sampler = mergen.Sampler(space)
    sampler.set_design(n_samples=30)
    result = sampler.run(criteria='umaxpro', algorithm='sa',
                         seed=44, verbose=False)
    _grab(result, 'pairplot', 'tutorial_pairplot.png')
    _grab(result, 'quality',  'tutorial_quality.png')
    _grab(result, '1d',       'tutorial_1d.png')


def criterion_contrast():
    """Same space, two criteria: phi_p vs umaxpro pairplots."""
    print("[2/3] criterion contrast (phi_p vs umaxpro)")
    for crit in ('phi_p', 'umaxpro'):
        space = mergen.ParameterSpace({
            'x1': ('continuous', 0.0, 1.0, {'resolution': 21, 'round': 3}),
            'x2': ('continuous', 0.0, 1.0, {'resolution': 21, 'round': 3}),
        })
        sampler = mergen.Sampler(space)
        sampler.set_design(n_samples=15)
        result = sampler.run(criteria=crit, algorithm='sa',
                             seed=44, verbose=False)
        _grab(result, 'pairplot', f'contrast_{crit}.png')
        _grab(result, 'quality',  f'contrast_{crit}_quality.png')


def algorithm_comparison():
    """One criterion, three optimisers: the comparison bar chart."""
    print("[3/3] algorithm comparison (phi_p; sa, sce, ese)")
    space = mergen.ParameterSpace({
        'x1': ('continuous', 0.0, 1.0, {'resolution': 21, 'round': 3}),
        'x2': ('continuous', 0.0, 1.0, {'resolution': 21, 'round': 3}),
    })
    sampler = mergen.Sampler(space)
    sampler.set_design(n_samples=15)
    result = sampler.run(criteria='phi_p', algorithm=['sa', 'sce', 'ese'],
                         seed=44, verbose=False)
    _grab(result, 'comparison', 'algorithm_comparison.png')


if __name__ == "__main__":
    tutorial_design()
    criterion_contrast()
    algorithm_comparison()
    print("done: 8 figures in", IMG)
