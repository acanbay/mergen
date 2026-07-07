"""
mergen.output
=============
Visualisation and export for SamplingResult.

Plot types
----------
    result.plot('pairplot')     — scatter matrix, all parameter pairs
    result.plot('1d')           — marginal KDE + strip per parameter
    result.plot('2d')           — scatter for selected pair(s)
    result.plot('distances')    — pairwise distance distribution
    result.plot('correlation')  — parameter correlation heatmap
    result.plot('quality')      — quality metrics bar chart
    result.plot('all')          — all of the above

Export formats
--------------
    result.to_csv('design.csv')
    result.to_json('design.json')
    result.to_excel('design.xlsx')
    result.to_markdown('report.md')
    result.to_latex('report.tex')
    result.to_html('report.html')

All text-based exports (markdown, latex, html) include:
    1. Package banner
    2. Design summary
    3. Parameter space description
    4. Quality metrics
    5. Design points table (per point type, separately)
    6. Validation set table
    7. Extra sets tables

All tabular exports (csv, excel) include all points in a single table
with a point_type column for filtering.
"""

from __future__ import annotations

import datetime
import os
import warnings
from itertools import combinations
from typing import List, Optional, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _MPL = True
except ImportError:
    _MPL = False

try:
    from scipy.stats import gaussian_kde
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ======================================================================
# Style constants
# ======================================================================

_STYLE = {
    "Optimised" : dict(color="#3a86ff", marker="o", ms=70, z=2, alpha=0.85, lw=0.5, ec="white"),
    "Focus"     : dict(color="#2a9d8f", marker="o", ms=70, z=4, alpha=0.90, lw=0.5, ec="white"),
    "Prescribed": dict(color="#e07b39", marker="o", ms=70, z=5, alpha=0.95, lw=0.5, ec="white"),
    "Validation": dict(color="#e63946", marker="o", ms=70, z=6, alpha=0.90, lw=0.5, ec="white"),
}

_DRAW   = ["Optimised", "Validation", "Focus", "Prescribed"]
_LEGEND = ["Prescribed", "Focus", "Optimised", "Validation"]

# Layout constants (inches)
_PS     = 3.0    # base panel size
_LGND   = 1.5    # legend column width
_LPAD   = 0.65
_RPAD   = 0.20
_TPAD_T = 0.45
_TPAD_N = 0.10
_BPAD   = 0.55
_HGAP   = 0.45
_VGAP   = 0.45
_MAX_W  = 10.0
_MAX_H  = 9.5
_MIN_PS = 0.8

_POOL_MS         = 25
_POOL_MAX_LEVELS = 50

# Candidate-pool alpha is scaled by the number of unique projected
# points: alpha = clip(K / n_unique, MIN, MAX). Sparse grids stay
# solid; dense grids fade so the design markers remain visible.
_POOL_ALPHA_K    = 120.0
_POOL_ALPHA_MIN  = 0.05
_POOL_ALPHA_MAX  = 0.50

_cnt: dict = {}


# ======================================================================
# Utilities
# ======================================================================

def _require_mpl() -> None:
    if not _MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def _outdir(result) -> str:
    return getattr(result, 'output_dir', 'outputs')


def _resolve_filename(kind: str, filename: Optional[str], fmt: Optional[str]) -> str:
    if filename:
        return filename
    ext = fmt.lstrip('.').lower() if fmt else 'png'
    _cnt[kind] = _cnt.get(kind, 0) + 1
    return f"{kind}_{_cnt[kind]}.{ext}"


def _savefig(fig, result, filename: str, dpi: int) -> None:
    d = _outdir(result)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, filename)
    fig.savefig(p, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {p}")


def _collect(result) -> pd.DataFrame:
    """Combine all sets into a single DataFrame."""
    frames = [result.samples]
    if len(result.validation):
        frames.append(result.validation)
    for df in getattr(result, 'sets', {}).values():
        if len(df):
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Focus points are part of the optimised design; plots show them with
    # the same colour and legend entry. Exports keep the Focus label.
    out.loc[out["point_type"] == "Focus", "point_type"] = "Optimised"
    return out


def _present(df: pd.DataFrame) -> list:
    types_in = df["point_type"].unique()
    base  = [t for t in _DRAW if t in types_in]
    extra = [t for t in types_in if t not in _DRAW]
    return base + extra


def _extra_colors(result) -> dict:
    return {
        name: sdf["color"].iloc[0]
        for name, sdf in getattr(result, 'sets', {}).items()
        if len(sdf) and "color" in sdf.columns
    }


def _pool_visible(space, xc: str, yc: str) -> bool:
    """Pool is always visible — large spaces are sampled."""
    return xc in space._parameters and yc in space._parameters


# ======================================================================
# Figure builder
# ======================================================================

def _legend_handles(present: list, ec: dict, show_pool: bool = False):
    handles, lbls = [], []
    if show_pool:
        handles.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='#aaaaaa',
                               markeredgecolor='none',
                               markersize=np.sqrt(_POOL_MS) * 0.9,
                               linestyle='none'))
        lbls.append("Candidate pool")
    for t in _LEGEND:
        if t not in present:
            continue
        st = _STYLE[t]
        handles.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=st['color'],
                               markeredgecolor=st['ec'],
                               markeredgewidth=st['lw'],
                               markersize=np.sqrt(st['ms']) * 0.9,
                               linestyle='none'))
        lbls.append(t)
    for t, color in ec.items():
        if t not in present:
            continue
        handles.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=color,
                               markeredgecolor='white',
                               markeredgewidth=0.5,
                               markersize=np.sqrt(70) * 0.9,
                               linestyle='none'))
        lbls.append(t)
    return handles, lbls


def _make_fig(nrows: int, ncols: int, title: str,
              present: list, ec: dict, show_pool: bool = False):
    tpad  = _TPAD_T if title else _TPAD_N
    ps_w  = (_MAX_W - _LPAD - (ncols - 1) * _HGAP - _LGND - _RPAD) / max(ncols, 1)
    ps_h  = (_MAX_H - tpad  - (nrows - 1) * _VGAP - _BPAD)          / max(nrows, 1)
    ps    = min(_PS, ps_w, ps_h)

    if ps < _MIN_PS:
        warnings.warn(
            f"Plot panels are very small ({ps:.2f} in). "
            f"Consider using plot('2d', params=[...]) for specific pairs.",
            UserWarning, stacklevel=5
        )

    fig_w = _LPAD + ncols * ps + (ncols - 1) * _HGAP + _LGND + _RPAD
    fig_h = tpad  + nrows * ps + (nrows - 1) * _VGAP + _BPAD
    fig   = plt.figure(figsize=(fig_w, fig_h))

    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            left   = (_LPAD + c * (ps + _HGAP)) / fig_w
            bottom = (_BPAD + (nrows - 1 - r) * (ps + _VGAP)) / fig_h
            axes[r, c] = fig.add_axes([left, bottom, ps / fig_w, ps / fig_h])

    if title:
        cx  = (_LPAD + (ncols * ps + (ncols - 1) * _HGAP) / 2) / fig_w
        top = (_BPAD + nrows * ps + (nrows - 1) * _VGAP) / fig_h
        fig.text(cx, top + 0.015, title, ha='center', va='bottom', fontsize=11)

    lgnd_left = (_LPAD + ncols * (ps + _HGAP) - _HGAP + 0.15) / fig_w
    lgnd_mid  = (_BPAD + (nrows * ps + (nrows - 1) * _VGAP) / 2) / fig_h
    handles, lbls = _legend_handles(present, ec, show_pool)
    fig.legend(handles, lbls,
               title="Point type", loc='center left',
               bbox_to_anchor=(lgnd_left, lgnd_mid),
               fontsize=8, title_fontsize=8,
               framealpha=0.92, borderpad=0.7,
               handlelength=1.5, handletextpad=0.5)

    return fig, axes


# ======================================================================
# Core drawing helpers
# ======================================================================

def _scatter(ax, df, xc: str, yc: str, present: list,
             space, show_pool: bool, ec: dict) -> None:
    if show_pool and _pool_visible(space, xc, yc):
        pool  = space.candidate_pool
        names = space.names
        xi, yi = names.index(xc), names.index(yc)
        # Project onto the (xc, yc) plane and keep each grid location
        # once. A d-dimensional grid projects many rows onto the same
        # 2D point; drawing them all would stack translucent markers
        # and make some cells look darker purely because of overlap,
        # not because they carry more information. De-duplicating gives
        # every visible candidate an identical, honest weight.
        proj = np.unique(pool[:, [xi, yi]], axis=0)
        # Scale alpha to the number of *unique* projected points so a
        # dense grid stays legible without washing out the design
        # markers. This is a display heuristic, not a statistical
        # claim: alpha decays inversely with point count, clipped to a
        # readable band.
        alpha = float(np.clip(_POOL_ALPHA_K / max(len(proj), 1),
                              _POOL_ALPHA_MIN, _POOL_ALPHA_MAX))
        ax.scatter(proj[:, 0], proj[:, 1],
                   facecolors='#aaaaaa', edgecolors='none',
                   s=_POOL_MS, alpha=alpha, zorder=1, linewidths=0)

    for t in _DRAW:
        if t not in present:
            continue
        sub = df[df["point_type"] == t]
        if sub.empty:
            continue
        st = _STYLE[t]
        ax.scatter(sub[xc], sub[yc],
                   s=st['ms'], c=st['color'], marker=st['marker'],
                   edgecolors=st['ec'], linewidths=st['lw'],
                   alpha=st['alpha'], zorder=st['z'])

    for t, color in ec.items():
        if t not in present:
            continue
        sub = df[df["point_type"] == t]
        if sub.empty:
            continue
        ax.scatter(sub[xc], sub[yc],
                   s=70, c=color, marker='o',
                   edgecolors='white', linewidths=0.5,
                   alpha=0.90, zorder=8)

    for dim, setter in [(xc, ax.set_xlim), (yc, ax.set_ylim)]:
        if dim in space._parameters:
            v   = space._parameters[dim]
            pad = (v.max() - v.min()) * 0.07 or 0.5
            setter(float(v.min()) - pad, float(v.max()) + pad)

    ax.set_xlabel(xc, fontsize=9)
    ax.set_ylabel(yc, fontsize=9)
    ax.tick_params(labelsize=7)


def _kde_strip(ax, df, param: str, present: list, space, ec: dict) -> None:
    if param not in space._parameters:
        ax.set_visible(False)
        return

    v   = space._parameters[param]
    xlo = float(v.min())
    xhi = float(v.max())
    xg  = np.linspace(xlo, xhi, 400)

    all_types = list(present)

    def _color(t):
        if t in _STYLE:
            return _STYLE[t]['color']
        return ec.get(t, '#888888')

    kdes = {}
    if _SCIPY:
        for t in all_types:
            s = df[df["point_type"] == t][param].dropna().values
            if len(s) < 2:
                continue
            try:
                kdes[t] = gaussian_kde(s, bw_method='scott')(xg)
            except Exception:
                pass

    gmax = max((y.max() for y in kdes.values()), default=1.0) or 1.0

    for t in all_types:
        c = _color(t)
        if t in kdes:
            yn = kdes[t] / gmax
            ax.fill_between(xg, 0, yn, alpha=0.18, color=c)
            ax.plot(xg, yn, color=c, lw=1.4, alpha=0.82)
        else:
            for uv in np.unique(df[df["point_type"] == t][param].dropna().values):
                ax.axvline(uv, color=c, lw=1.4, alpha=0.82, linestyle='--')

    rng = np.random.default_rng(seed=44)
    for t in all_types:
        vals = df[df["point_type"] == t][param].dropna().values
        if len(vals) == 0:
            continue
        c   = _color(t)
        st  = _STYLE.get(t)
        # Jitter strip-plot points vertically to reduce overplotting.
        ms  = (st['ms']    if st else 70) * 0.55
        lw  = (st['lw']    if st else 0.5)
        eca = (st['ec']    if st else 'white')
        alp = (st['alpha'] if st else 0.90)
        z   = (st['z']     if st else 3)
        jit = rng.uniform(-0.025, 0.025, len(vals))
        ax.scatter(vals, np.full(len(vals), -0.07) + jit,
                   c=c, marker='o', s=ms, alpha=alp,
                   edgecolors=eca, linewidths=lw, zorder=z, clip_on=False)

    ax.axhline(0, color='#bbbbbb', lw=0.5, ls='--')
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.18, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1'], fontsize=7)
    ax.set_xlabel(param, fontsize=9)
    ax.tick_params(labelsize=7)


def _apply_log(ax, xc: str, yc: str, log_params: list) -> set:
    if not log_params:
        return set()
    skipped = set()
    for p in log_params:
        if p == xc:
            ax.set_xscale('log')
        elif p == yc:
            ax.set_yscale('log')
        else:
            skipped.add(p)
    return skipped


# ======================================================================
# Plot functions
# ======================================================================

def plot_pairplot(result, show_pool: bool = False, title: bool = True,
                  log: Optional[list] = None,
                  show: bool = True, save: bool = False,
                  filename: Optional[str] = None,
                  fmt: Optional[str] = None, dpi: int = 150) -> None:
    """
    Scatter matrix of all parameter pairs.

    Parameters
    ----------
    show_pool : bool — overlay the candidate pool (default False).
                When True, the grid is projected onto each panel and
                de-duplicated so every candidate is shown once with an
                equal, density-scaled opacity.
    title     : bool — show figure title
    log       : list of parameter names to display on log scale
    show, save, filename, fmt, dpi — output options
    """
    _require_mpl()
    space   = result.space
    names   = space.names
    nd      = len(names)
    df      = _collect(result)
    present = _present(df)
    ec      = _extra_colors(result)

    if nd > 8:
        raise ValueError(
            f"pairplot: {nd} parameters exceeds the maximum (8). "
            f"Use plot('2d', params=[...]) to show specific pairs."
        )
    if nd >= 6:
        warnings.warn(
            f"pairplot: {nd} parameters — panels will be small. "
            f"Consider plot('2d', params=[...]) for specific pairs.",
            UserWarning, stacklevel=2
        )

    tstr       = "Pairplot" if title else ""
    fig, axes  = _make_fig(nd, nd, tstr, present, ec, show_pool)
    rev_names  = list(reversed(names))

    all_skipped: set = set()
    for i, yc in enumerate(rev_names):
        for j, xc in enumerate(names):
            ax      = axes[i, j]
            is_self = (xc == yc)
            _scatter(ax, df, xc, yc, present, space,
                     show_pool and not is_self, ec)
            ax.set_xlabel(xc if i == nd - 1 else "", fontsize=9)
            ax.set_ylabel(yc if j == 0 else "", fontsize=9)
            if is_self:
                if xc in space._parameters:
                    v   = space._parameters[xc]
                    pad = (v.max() - v.min()) * 0.07 or 0.5
                    lim = (float(v.min()) - pad, float(v.max()) + pad)
                    ax.set_xlim(lim)
                    ax.set_ylim(lim)
                    ax.plot(lim, lim, color='#cccccc', lw=0.8, ls='--', zorder=0)
            if log:
                all_skipped |= _apply_log(ax, xc, yc, log)

    if log:
        skipped = all_skipped - set(names)
        if skipped:
            warnings.warn(f"plot('pairplot'): log scale skipped for {sorted(skipped)}",
                          UserWarning, stacklevel=2)

    fn = _resolve_filename("pairplot", filename, fmt)
    if save:
        _savefig(fig, result, fn, dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_1d(result, params: Optional[Union[str, list]] = None,
            title: bool = True, log: Optional[list] = None,
            show: bool = True, save: bool = False,
            filename: Optional[str] = None,
            fmt: Optional[str] = None, dpi: int = 150) -> None:
    """
    Marginal KDE + strip chart for each parameter.

    Parameters
    ----------
    params : str, list, or None — parameters to plot (default: all)
    """
    _require_mpl()
    space   = result.space
    df      = _collect(result)
    present = _present(df)
    ec      = _extra_colors(result)

    selected = (space.names     if params is None else
                [params]        if isinstance(params, str) else
                list(params))
    n    = len(selected)
    tstr = "1D Distribution" if title else ""

    fig, axes = _make_fig(1, n, tstr, present, ec)
    for j, param in enumerate(selected):
        _kde_strip(axes[0, j], df, param, present, space, ec)
        if j > 0:
            axes[0, j].set_ylabel("")
        if log and param in log:
            axes[0, j].set_xscale('log')

    fn = _resolve_filename("1d", filename, fmt)
    if save:
        _savefig(fig, result, fn, dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_2d(result, params: Optional[Union[list, List[list]]] = None,
            show_pool: bool = False, title: bool = True,
            log: Optional[list] = None,
            show: bool = True, save: bool = False,
            filename: Optional[str] = None,
            fmt: Optional[str] = None, dpi: int = 150) -> None:
    """
    Scatter plot for selected parameter pair(s).

    Parameters
    ----------
    params : [x, y] or [[x1,y1], [x2,y2], ...] — pairs to plot
    """
    _require_mpl()
    space   = result.space
    names   = space.names
    df      = _collect(result)
    present = _present(df)
    ec      = _extra_colors(result)

    if params is None:
        pairs = list(combinations(names, 2))
    elif (isinstance(params, list) and len(params) == 2
          and isinstance(params[0], str)):
        pairs = [tuple(params)]
    else:
        pairs = [tuple(p) for p in params]

    n     = len(pairs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    tstr  = "2D Scatter" if title else ""

    fig, axes = _make_fig(nrows, ncols, tstr, present, ec, show_pool)

    plotted: set = set()
    for idx, (xc, yc) in enumerate(pairs):
        ax = axes[idx // ncols, idx % ncols]
        _scatter(ax, df, xc, yc, present, space, show_pool, ec)
        ax.set_xlabel(xc, fontsize=9)
        ax.set_ylabel(yc, fontsize=9)
        if log:
            _apply_log(ax, xc, yc, log)
            plotted |= {xc, yc}

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fn = _resolve_filename("2d", filename, fmt)
    if save:
        _savefig(fig, result, fn, dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_distances(result, title: bool = True,
                   show: bool = True, save: bool = False,
                   filename: Optional[str] = None,
                   fmt: Optional[str] = None, dpi: int = 150) -> None:
    """Pairwise distance distribution for all sets."""
    _require_mpl()
    space   = result.space
    gmins   = space.gmins
    granges = space.granges

    def _d(pts):
        if len(pts) < 2:
            return np.array([])
        norm = (pts - gmins) / granges
        out  = []
        for i in range(len(norm) - 1):
            out.extend(np.linalg.norm(norm[i + 1:] - norm[i], axis=1).tolist())
        return np.array(out)

    sets = [("Design", result.samples[space.names].values, "#3a86ff")]
    if len(result.validation):
        sets.append(("Validation", result.validation[space.names].values, "#e63946"))
    for name, sdf in getattr(result, 'sets', {}).items():
        if len(sdf):
            c = sdf["color"].iloc[0] if "color" in sdf.columns else "#888888"
            sets.append((name, sdf[space.names].values, c))

    tpad = _TPAD_T if title else _TPAD_N
    pw   = min(_PS * 1.6, _MAX_W - _LPAD - _LGND - _RPAD)
    ph   = min(_PS, _MAX_H - tpad - _BPAD)
    fw   = _LPAD + pw + _LGND + _RPAD
    fh   = tpad + ph + _BPAD
    fig  = plt.figure(figsize=(fw, fh))
    ax   = fig.add_axes([_LPAD / fw, _BPAD / fh, pw / fw, ph / fh])

    all_d = [d for _, pts, _ in sets if len(d := _d(pts)) > 0]
    if not all_d:
        plt.close(fig)
        return

    xmin = min(d.min() for d in all_d)
    xmax = max(d.max() for d in all_d)
    xg   = np.linspace(xmin, xmax, 300)

    handles, lbls = [], []
    for name, pts, color in sets:
        dists = _d(pts)
        if len(dists) == 0:
            continue
        if _SCIPY and len(dists) >= 3:
            try:
                y = gaussian_kde(dists, bw_method='scott')(xg)
                ax.fill_between(xg, y, alpha=0.20, color=color)
                ax.plot(xg, y, color=color, lw=1.8)
            except Exception:
                ax.hist(dists, bins=20, alpha=0.35, color=color, density=True)
        else:
            ax.hist(dists, bins=20, alpha=0.35, color=color, density=True)
        handles.append(Line2D([0], [0], color=color, lw=2))
        lbls.append(name)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Normalised pairwise distance", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.tick_params(labelsize=8)

    if title:
        cx  = (_LPAD + pw / 2) / fw
        top = (_BPAD + ph) / fh
        fig.text(cx, top + 0.015, "Pairwise Distance Distribution",
                 ha='center', va='bottom', fontsize=11)

    lgnd_left = (_LPAD + pw + 0.15) / fw
    lgnd_mid  = (_BPAD + ph / 2) / fh
    fig.legend(handles, lbls, title="Set", loc='center left',
               bbox_to_anchor=(lgnd_left, lgnd_mid),
               fontsize=8, title_fontsize=8,
               framealpha=0.92, borderpad=0.7,
               handlelength=2.0, handletextpad=0.5)

    fn = _resolve_filename("distances", filename, fmt)
    if save:
        _savefig(fig, result, fn, dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_correlation(result, title: bool = True,
                     show: bool = True, save: bool = False,
                     filename: Optional[str] = None,
                     fmt: Optional[str] = None, dpi: int = 150) -> None:
    """
    Parameter correlation heatmap for the design points.

    Shows absolute pairwise correlations between all parameters.
    Values close to 0 indicate good orthogonality.
    """
    _require_mpl()
    space   = result.space
    names   = space.names
    gmins   = space.gmins
    granges = space.granges
    d       = len(names)

    pts    = result.samples[names].values.astype(float)
    X_norm = (pts - gmins) / granges

    # Correlation matrix — diagonal set to 1.0 (self-correlation)
    Xc    = X_norm - X_norm.mean(axis=0)
    norms = np.sqrt((Xc ** 2).sum(axis=0))
    norms[norms < 1e-10] = 1.0
    Xn    = Xc / norms
    corr  = np.abs(Xn.T @ Xn)
    np.fill_diagonal(corr, 1.0)

    _MAX_SIZE = 10.0
    size     = min(_MAX_SIZE, max(3.5, d * 0.7))
    cell_fs  = max(5, int(9 - d * 0.3))   # shrink font as d grows
    tick_fs  = max(6, int(9 - d * 0.2))

    fig, ax = plt.subplots(figsize=(size, size))

    ax.imshow(corr, vmin=0, vmax=1, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(d))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=tick_fs)
    ax.set_yticks(range(d))
    ax.set_yticklabels(names, fontsize=tick_fs)

    for i in range(d):
        for j in range(d):
            val = corr[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha='center', va='center', fontsize=cell_fs,
                    color='white' if val > 0.5 else 'black')

    if title:
        ax.set_title("Parameter Correlation", fontsize=11, pad=10)

    fig.tight_layout()
    fn = _resolve_filename("correlation", filename, fmt)
    if save:
        _savefig(fig, result, fn, dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_quality(result, title: bool = True,
                 show: bool = True, save: bool = False,
                 filename: Optional[str] = None,
                 fmt: Optional[str] = None, dpi: int = 150) -> None:
    """
    Quality metrics bar chart.

    Computes all default metrics and displays them as horizontal bars.
    Bars are coloured green (good) to red (poor) relative to a random baseline.
    """
    _require_mpl()
    from . import metrics as _metrics

    stats = result.quality_report(mc_samples=200, verbose=False)

    metric_names = _metrics._DEFAULT_METRICS
    labels = [_metrics._METRIC_LABELS.get(m, m) for m in metric_names]
    values = [stats.get(m, np.nan) for m in metric_names]
    ranks  = [stats.get(f'{m}_percentile_rank', None) for m in metric_names]

    # Normalise values to [0, 1] for bar length
    # For higher-is-better: value as-is; for lower-is-better: 1 - normalised
    baselines = [stats.get(f'{m}_baseline_median', None) for m in metric_names]

    # Figure: wider to accommodate labels + legend outside
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(metric_names) * 0.7)))

    colors = []
    for r in ranks:
        if r is None or np.isnan(r):
            colors.append('#aaaaaa')
        elif r >= 75:
            colors.append('#2a9d8f')
        elif r >= 50:
            colors.append('#e9c46a')
        else:
            colors.append('#e63946')

    y_pos = range(len(metric_names))
    ax.barh(list(y_pos), values, color=colors, alpha=0.85, height=0.6)

    # Baseline markers
    # Draw a vertical dashed line for each baseline value directly on the
    # data coordinates. This is more robust to y-axis inversions than
    # using axes coordinates with ymin/ymax. The line's height is set
    # slightly larger than the bar height for better visibility.
    bar_height = 0.6
    marker_height = bar_height * 1.1  # 110% of the bar height
    for i, (bl, v) in enumerate(zip(baselines, values)):
        if bl is not None and not np.isnan(bl):
            ax.plot([bl, bl], [i - marker_height / 2, i + marker_height / 2],
                       color='#333333', lw=1.2, ls='--', alpha=0.7)

    # Value labels inside bars (avoid overflow)
    x_max = max((v for v in values if not np.isnan(v)), default=1.0)
    ax.set_xlim(0, x_max * 1.35)
    for i, (v, r) in enumerate(zip(values, ranks)):
        if not np.isnan(v):
            label = f"{v:.4f}"
            if r is not None and not np.isnan(r):
                label += f"  ({r:.0f}th pct)"
            ax.text(v + x_max * 0.01, i, label, va='center', fontsize=8)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Metric value", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.invert_yaxis()

    if title:
        ax.set_title("Design Quality Metrics", fontsize=11, pad=10)

    # Legend outside plot area
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2a9d8f', label='>= 75th percentile'),
        Patch(facecolor='#e9c46a', label='50-75th percentile'),
        Patch(facecolor='#e63946', label='< 50th percentile'),
        Line2D([0], [0], color='#333333', lw=1.2, ls='--', label='Baseline'),
    ]
    ax.legend(handles=legend_elements, fontsize=8,
              loc='upper left', bbox_to_anchor=(1.01, 1.0),
              framealpha=0.9, borderpad=0.6)

    fig.tight_layout()
    fn = _resolve_filename("quality", filename, fmt)
    if save:
        _savefig(fig, result, fn, dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison(result, title: bool = True,
                    show: bool = True, save: bool = False,
                    filename: Optional[str] = None,
                    fmt: Optional[str] = None, dpi: int = 150) -> None:
    """
    Bar chart comparing the criterion score of each optimisation algorithm.

    Only meaningful when :meth:`Sampler.run` was called with
    ``algorithm=[...]``. The best (lowest-scoring) algorithm is
    highlighted; bar heights show the raw criterion value, lower is
    better. Elapsed time is annotated above each bar.

    Parameters
    ----------
    result : SamplingResult
        Must contain at least one entry in ``result.algorithm_results``.
    title, show, save, filename, fmt, dpi
        Standard plotting controls (see other ``plot_*`` functions).

    Raises
    ------
    ValueError
        If no algorithm results are present.
    """
    _require_mpl()
    alg_results = getattr(result, 'algorithm_results', {}) or {}
    if not alg_results:
        raise ValueError(
            "plot_comparison requires result.algorithm_results to be "
            "populated. Call Sampler.run(algorithm=[...]) with multiple "
            "algorithms to compare them."
        )

    # Sort by score (best first)
    ranked = sorted(alg_results.items(), key=lambda kv: kv[1].score)
    names  = [name for name, _ in ranked]
    scores = [res.score for _, res in ranked]
    times  = [res.elapsed for _, res in ranked]
    best   = getattr(result, 'best_algorithm', names[0])

    # Dynamic figure width: scales with number of algorithms, capped to
    # a sensible upper bound to keep label spacing comfortable.
    n_alg = len(names)
    fig_w = min(14.0, max(5.0, 1.1 * n_alg + 3.5))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    colors = ['#2a9d8f' if name == best else '#aaaaaa' for name in names]
    bars = ax.bar(names, scores, color=colors, alpha=0.9, width=0.6,
                  edgecolor='#333333', linewidth=0.6)

    # Score + elapsed time labels above each bar
    s_max = max(scores) if scores else 1.0
    ax.set_ylim(0, s_max * 1.18)
    for bar, score, t in zip(bars, scores, times):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + s_max * 0.02,
                f"{score:.4g}\n({t:.2f}s)",
                ha='center', va='bottom', fontsize=8.5,
                fontweight='normal')

    crit_name = result._meta.get('criteria', 'criterion')
    ax.set_ylabel(f"{crit_name} score  (lower is better)", fontsize=9)
    # x label intentionally omitted — algorithm names speak for themselves
    ax.tick_params(labelsize=9)
    ax.grid(axis='y', alpha=0.25, linestyle=':')
    ax.set_axisbelow(True)

    if title:
        ax.set_title(f"Algorithm comparison  --  best = {best}",
                     fontsize=11, pad=10)

    fig.tight_layout()
    fn = _resolve_filename("comparison", filename, fmt)
    if save:
        _savefig(fig, result, fn, dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_all(result, show_pool: bool = False, title: bool = True,
             log: Optional[list] = None,
             show: bool = True, save: bool = False,
             fmt: Optional[str] = None, dpi: int = 150) -> None:
    """Produce all standard plots."""
    names = result.space.names
    if len(names) <= 8:
        plot_pairplot(result, show_pool=show_pool, title=title, log=log,
                      show=show, save=save, fmt=fmt, dpi=dpi)
    for p in names:
        plot_1d(result, params=[p], title=title, log=log,
                show=show, save=save, fmt=fmt, dpi=dpi)
    for pair in combinations(names, 2):
        plot_2d(result, params=[list(pair)], show_pool=show_pool,
                title=title, log=log, show=show, save=save, fmt=fmt, dpi=dpi)
    plot_distances(result, title=title, show=show, save=save, fmt=fmt, dpi=dpi)
    plot_correlation(result, title=title, show=show, save=save, fmt=fmt, dpi=dpi)
    plot_quality(result, title=title, show=show, save=save, fmt=fmt, dpi=dpi)

    # Multi-algorithm bar chart — only when more than one was run
    if len(getattr(result, 'algorithm_results', {}) or {}) > 1:
        plot_comparison(result, title=title, show=show, save=save,
                        fmt=fmt, dpi=dpi)


# ── Dispatcher ────────────────────────────────────────────────────────────

def plot(result, kind: str = 'pairplot', **kwargs) -> None:
    """
    Visualise the design.

    Parameters
    ----------
    kind : str
        'pairplot'    — scatter matrix, all parameter pairs
        '1d'          — marginal KDE + strip per parameter
        '2d'          — scatter for selected pair(s)
        'distances'   — pairwise distance distribution
        'correlation' — parameter correlation heatmap
        'quality'     — quality metrics bar chart
        'comparison'  — bar chart of per-algorithm scores
                        (only when multiple algorithms were run)
        'all'         — all of the above
    **kwargs
        show_pool, title, log, params, show, save, filename, fmt, dpi
    """
    dispatch = {
        'pairplot'   : plot_pairplot,
        '1d'         : plot_1d,
        '2d'         : plot_2d,
        'distances'  : plot_distances,
        'correlation': plot_correlation,
        'quality'    : plot_quality,
        'comparison' : plot_comparison,
        'all'        : plot_all,
    }
    if kind not in dispatch:
        raise ValueError(
            f"Unknown plot kind '{kind}'. "
            f"Available: {list(dispatch.keys())}"
        )
    dispatch[kind](result, **kwargs)


# ======================================================================
# Report helpers
# ======================================================================

def _report_header(result) -> str:
    """Package banner + design summary for text-based exports."""
    import mergen
    space = result.space
    meta  = result._meta

    lines = [
        mergen._banner(),
        "",
        f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Parameter Space",
        "─" * 40,
    ]
    for name, vals in zip(space.names, space.values):
        ptype = space._param_types.get(name, 'discrete')
        lines.append(
            f"  {name:<24} {ptype:<12} "
            f"{len(vals):>5} levels  "
            f"[{float(vals.min()):.4g}, {float(vals.max()):.4g}]"
        )
    lines += [
        f"  {'Candidates':<24} {space.n_candidates:>5,}",
        "",
        "Design Summary",
        "─" * 40,
    ]
    vc = result.samples["point_type"].value_counts()
    for lbl in ("Prescribed", "Focus", "Optimised"):
        n = vc.get(lbl, 0)
        if n:
            lines.append(f"  {lbl:<24} {n:>5}")
    lines.append(f"  {'Total design':<24} {len(result.samples):>5}")
    if len(result.validation):
        lines.append(f"  {'Validation':<24} {len(result.validation):>5}")
    for sname, sdf in getattr(result, 'sets', {}).items():
        lines.append(f"  {sname:<24} {len(sdf):>5}")
    lines += [
        "",
        "Run Settings",
        "─" * 40,
        f"  {'Criterion':<24} {meta.get('criteria', '?')}",
        f"  {'Seed':<24} {meta.get('seed', '?')}",
    ]

    # Algorithm reporting: single or multi
    alg = meta.get('algorithm', None)
    if isinstance(alg, list):
        lines.append(f"  {'Algorithms':<24} {', '.join(alg)}")
        if meta.get('best_algorithm'):
            lines.append(f"  {'Best algorithm':<24} {meta['best_algorithm']}")
    elif alg is not None:
        lines.append(f"  {'Algorithm':<24} {alg}")

    # Per-algorithm scores table when multiple were run
    alg_results = getattr(result, 'algorithm_results', {}) or {}
    if len(alg_results) > 1:
        lines += [
            "",
            "Per-algorithm Scores",
            "─" * 40,
        ]
        ranked = sorted(alg_results.items(), key=lambda kv: kv[1].score)
        for name, res in ranked:
            mark = "*" if name == getattr(result, 'best_algorithm', None) else " "
            lines.append(
                f"  {mark} {name:<8} score={res.score:<12.6g} "
                f"elapsed={res.elapsed:.2f}s  n_iter={res.n_iter}"
            )

    # Quality metrics — full table format (same as terminal)
    try:
        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        result.quality_report(mc_samples=200, verbose=True)
        sys.stdout = old_stdout
        table_str = buf.getvalue()
        # Strip ANSI colour codes for text-based exports
        import re
        table_str = re.sub(r'\033\[[0-9;]*m', '', table_str)
        if table_str.strip():
            lines += ["", table_str.rstrip()]
    except Exception:
        pass

    return "\n".join(lines)


# ======================================================================
# Export functions
# ======================================================================

def export_csv(result, filename: str = 'design.csv') -> None:
    """Save all points to CSV (single table, point_type column)."""
    d  = _outdir(result)
    os.makedirs(d, exist_ok=True)
    p  = os.path.join(d, filename)
    df = _full_df(result)
    df.to_csv(p, index=True, float_format='%.10g')
    print(f"  Saved: {p}  ({len(df)} rows)")


def export_json(result, filename: str = 'design.json') -> None:
    """Save all points to JSON (single table)."""
    d  = _outdir(result)
    os.makedirs(d, exist_ok=True)
    p  = os.path.join(d, filename)
    df = _full_df(result)
    df.to_json(p, orient='records', indent=2, double_precision=10)
    print(f"  Saved: {p}  ({len(df)} rows)")


def export_excel(result, filename: str = 'design.xlsx') -> None:
    """
    Save to Excel — single sheet with all points and point_type column.
    Columns are auto-sized and header row is formatted.
    """
    try:
        import openpyxl  # noqa: F401  (availability check)
    except ImportError:
        raise ImportError(
            "openpyxl is required for Excel export. "
            "Install it with: pip install openpyxl"
        )
    d  = _outdir(result)
    os.makedirs(d, exist_ok=True)
    p  = os.path.join(d, filename)
    df = _full_df(result)

    with pd.ExcelWriter(p, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Design', index=True, float_format='%.10g')
        ws = writer.sheets['Design']
        # Auto-size columns
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)

    print(f"  Saved: {p}  ({len(df)} rows)")


def export_markdown(result, filename: str = 'report.md') -> None:
    """
    Save a full Markdown report:
    banner → metadata → parameter space → design summary →
    run settings → quality metrics → design tables.
    """
    import mergen
    import re
    import io
    import sys as _sys
    d = _outdir(result)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, filename)

    space = result.space
    names = space.names
    meta  = result._meta

    with open(p, 'w', encoding='utf-8') as f:

        # ── Banner ────────────────────────────────────────────────────
        f.write("# Mergen Design Report\n\n")
        banner_lines = mergen._banner().split('\n')
        for bl in banner_lines:
            f.write(f"> {bl}  \n")
        f.write(f"> Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write("\n---\n\n")

        # ── Parameter Space ───────────────────────────────────────────
        f.write("## Parameter Space\n\n")
        f.write("| Parameter | Type | Levels | Range |\n")
        f.write("|-----------|------|-------:|-------|\n")
        for name, vals in zip(space.names, space.values):
            ptype = space._param_types.get(name, 'discrete')
            f.write(f"| {name} | {ptype} | {len(vals)} | [{float(vals.min()):.4g}, {float(vals.max()):.4g}] |\n")
        f.write(f"| **Feasible candidates** | | {space.n_candidates} | |\n")
        f.write("\n---\n\n")

        # ── Design Summary ────────────────────────────────────────────
        f.write("## Design Summary\n\n")
        vc = result.samples["point_type"].value_counts()
        for lbl in ("Prescribed", "Focus", "Optimised"):
            n = vc.get(lbl, 0)
            if n:
                f.write(f"- **{lbl}**: {n}\n")
        f.write(f"- **Total design**: {len(result.samples)}\n")
        if len(result.validation):
            f.write(f"- **Validation**: {len(result.validation)}\n")
        for sname, sdf in getattr(result, 'sets', {}).items():
            f.write(f"- **{sname}**: {len(sdf)}\n")
        f.write("\n")

        # ── Run Settings ──────────────────────────────────────────────
        f.write("## Run Settings\n\n")
        f.write(f"- **Criterion**: {meta.get('criteria', '?')}\n")
        f.write(f"- **Restarts**: {meta.get('n_restarts', '?')}\n")
        f.write(f"- **Seed**: {meta.get('seed', '?')}\n")
        f.write("\n---\n\n")

        # ── Quality Metrics — same as terminal ────────────────────────
        f.write("## Quality Metrics\n\n")
        try:
            buf = io.StringIO()
            old_stdout = _sys.stdout
            _sys.stdout = buf
            result.quality_report(mc_samples=200, verbose=True)
            _sys.stdout = old_stdout
            table_str = buf.getvalue()
            # Strip ANSI colour codes
            table_str = re.sub(r'\033\[[0-9;]*m', '', table_str)
            # Strip [METRICS] progress lines
            lines_clean = [l for l in table_str.split('\n')
                          if '[METRICS]' not in l]
            table_str = '\n'.join(lines_clean).strip()
            if table_str:
                f.write("```\n" + table_str + "\n```\n\n")
        except Exception:
            pass
        f.write("---\n\n")

        # ── Design Points ─────────────────────────────────────────────
        f.write("## Design Points\n\n")
        for lbl in ("Prescribed", "Focus", "Optimised"):
            sub = result.samples[result.samples["point_type"] == lbl]
            if sub.empty:
                continue
            f.write(f"### {lbl} ({len(sub)} points)\n\n")
            try:
                f.write(sub[names].to_markdown(index=True, floatfmt=".10g") + "\n\n")
            except Exception:
                f.write(sub[names].to_string(index=True, float_format=lambda x: f"{x:.10g}") + "\n\n")

        # ── Validation ────────────────────────────────────────────────
        if len(result.validation):
            f.write(f"## Validation Set ({len(result.validation)} points)\n\n")
            try:
                f.write(result.validation[names].to_markdown(index=True, floatfmt='.10g') + "\n\n")
            except Exception:
                f.write(result.validation[names].to_string(index=True, float_format=lambda x: f"{x:.10g}") + "\n\n")

        # ── Extra sets ────────────────────────────────────────────────
        for sname, sdf in getattr(result, 'sets', {}).items():
            if len(sdf):
                f.write(f"## {sname} ({len(sdf)} points)\n\n")
                try:
                    f.write(sdf[names].to_markdown(index=True, floatfmt='.10g') + "\n\n")
                except Exception:
                    f.write(sdf[names].to_string(index=True, float_format=lambda x: f"{x:.10g}") + "\n\n")

    print(f"  Saved: {p}")


def export_latex(result, filename: str = 'report.tex') -> None:
    """
    Save a LaTeX report: banner → parameter space → design summary →
    run settings → quality metrics → design tables per point type.
    """
    import mergen
    import re
    import io
    import sys as _sys
    d = _outdir(result)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, filename)

    space = result.space
    names = space.names
    meta  = result._meta

    with open(p, 'w', encoding='utf-8') as f:
        # Banner as comment
        for line in mergen._banner().split('\n'):
            f.write(f"% {line}\n")
        f.write(f"% Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("\\section*{Mergen Design Report}\n\n")

        # Parameter Space
        f.write("\\subsection*{Parameter Space}\n\n")
        f.write("\\begin{tabular}{llrl}\n")
        f.write("\\hline\nParameter & Type & Levels & Range \\\\\n\\hline\n")
        for name, vals in zip(space.names, space.values):
            ptype = space._param_types.get(name, 'discrete')
            f.write(f"{name} & {ptype} & {len(vals)} & "
                    f"[{float(vals.min()):.4g}, {float(vals.max()):.4g}] \\\\\n")
        f.write(f"\\hline\nFeasible candidates & & {space.n_candidates} & \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\n")

        # Design Summary
        f.write("\\subsection*{Design Summary}\n\n")
        vc = result.samples["point_type"].value_counts()
        f.write("\\begin{tabular}{lr}\n\\hline\n")
        for lbl in ("Prescribed", "Focus", "Optimised"):
            n = vc.get(lbl, 0)
            if n:
                f.write(f"{lbl} & {n} \\\\\n")
        f.write(f"Total design & {len(result.samples)} \\\\\n")
        if len(result.validation):
            f.write(f"Validation & {len(result.validation)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\n")

        # Run Settings
        f.write("\\subsection*{Run Settings}\n\n")
        f.write("\\begin{tabular}{ll}\n\\hline\n")
        f.write(f"Criterion & {meta.get('criteria', '?')} \\\\\n")
        f.write(f"Restarts & {meta.get('n_restarts', '?')} \\\\\n")
        f.write(f"Seed & {meta.get('seed', '?')} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\n")

        # Quality Metrics
        f.write("\\subsection*{Quality Metrics}\n\n")
        try:
            buf = io.StringIO()
            old_stdout = _sys.stdout
            _sys.stdout = buf
            result.quality_report(mc_samples=200, verbose=True)
            _sys.stdout = old_stdout
            table_str = buf.getvalue()
            table_str = re.sub(r'\033\[[0-9;]*m', '', table_str)
            lines_clean = [l for l in table_str.split('\n') if '[METRICS]' not in l]
            f.write("\\begin{verbatim}\n")
            f.write('\n'.join(lines_clean).strip())
            f.write("\n\\end{verbatim}\n\n")
        except Exception:
            pass

        # Design points per type
        for lbl in ("Prescribed", "Focus", "Optimised"):
            sub = result.samples[result.samples["point_type"] == lbl]
            if sub.empty:
                continue
            f.write(f"\\subsection*{{{lbl} ({len(sub)} points)}}\n\n")
            f.write(sub[names].to_latex(index=True, float_format='%.10g') + "\n")

        # Validation
        if len(result.validation):
            f.write(f"\\subsection*{{Validation Set ({len(result.validation)} points)}}\n\n")
            f.write(result.validation[names].to_latex(index=True, float_format='%.10g') + "\n")

        # Extra sets
        for sname, sdf in getattr(result, 'sets', {}).items():
            if len(sdf):
                f.write(f"\\subsection*{{{sname} ({len(sdf)} points)}}\n\n")
                f.write(sdf[names].to_latex(index=True, float_format='%.10g') + "\n")

    print(f"  Saved: {p}")


def export_html(result, filename: str = 'report.html') -> None:
    """
    Save a self-contained HTML report with styled tables.
    """
    import mergen
    import re
    import io
    import sys as _sys
    d = _outdir(result)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, filename)

    space = result.space
    names = space.names
    meta  = result._meta

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
             max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #222; }
      h1   { color: #1a1a2e; border-bottom: 2px solid #3a86ff; padding-bottom: 0.3rem; }
      h2   { color: #16213e; margin-top: 2rem; }
      h3   { color: #0f3460; }
      blockquote { border-left: 4px solid #3a86ff; margin: 0; padding: 0.5rem 1rem;
                   background: #f8f9ff; color: #444; }
      pre  { background: #f4f4f8; padding: 1rem; border-radius: 6px;
             font-size: 0.85rem; overflow-x: auto; }
      table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
      th    { background: #3a86ff; color: white; padding: 0.5rem 0.8rem; text-align: left; }
      td    { padding: 0.4rem 0.8rem; border-bottom: 1px solid #e0e0e0; }
      tr:nth-child(even) td { background: #f8f9ff; }
      hr    { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
    </style>
    """

    def _df_to_html(df):
        return df.to_html(index=True, border=0, classes='', float_format='%.10g')

    def _get_metrics_text():
        try:
            buf = io.StringIO()
            old_stdout = _sys.stdout
            _sys.stdout = buf
            result.quality_report(mc_samples=200, verbose=True)
            _sys.stdout = old_stdout
            table_str = buf.getvalue()
            table_str = re.sub(r'\033\[[0-9;]*m', '', table_str)
            lines_clean = [l for l in table_str.split('\n') if '[METRICS]' not in l]
            return '\n'.join(lines_clean).strip()
        except Exception:
            return ''

    with open(p, 'w', encoding='utf-8') as f:
        f.write(f"<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
                f"<meta charset='UTF-8'>\n"
                f"<title>Mergen Design Report</title>\n{css}\n</head>\n<body>\n")

        # Banner
        f.write("<h1>Mergen Design Report</h1>\n")
        banner_lines = mergen._banner().split('\n')
        f.write("<blockquote>\n")
        for bl in banner_lines:
            f.write(f"  {bl}<br>\n")
        f.write(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("</blockquote>\n<hr>\n")

        # Parameter space
        f.write("<h2>Parameter Space</h2>\n")
        f.write("<table><tr><th>Parameter</th><th>Type</th><th>Levels</th><th>Range</th></tr>\n")
        for name, vals in zip(space.names, space.values):
            ptype = space._param_types.get(name, 'discrete')
            f.write(f"<tr><td>{name}</td><td>{ptype}</td><td>{len(vals)}</td>"
                    f"<td>[{float(vals.min()):.4g}, {float(vals.max()):.4g}]</td></tr>\n")
        f.write(f"<tr><td><strong>Feasible candidates</strong></td><td></td>"
                f"<td>{space.n_candidates}</td><td></td></tr>\n")
        f.write("</table>\n<hr>\n")

        # Design summary
        f.write("<h2>Design Summary</h2>\n<ul>\n")
        vc = result.samples["point_type"].value_counts()
        for lbl in ("Prescribed", "Focus", "Optimised"):
            n = vc.get(lbl, 0)
            if n:
                f.write(f"  <li><strong>{lbl}</strong>: {n}</li>\n")
        f.write(f"  <li><strong>Total design</strong>: {len(result.samples)}</li>\n")
        if len(result.validation):
            f.write(f"  <li><strong>Validation</strong>: {len(result.validation)}</li>\n")
        for sname, sdf in getattr(result, 'sets', {}).items():
            f.write(f"  <li><strong>{sname}</strong>: {len(sdf)}</li>\n")
        f.write("</ul>\n")

        # Run settings
        f.write("<h2>Run Settings</h2>\n<ul>\n")
        f.write(f"  <li><strong>Criterion</strong>: {meta.get('criteria', '?')}</li>\n")
        f.write(f"  <li><strong>Restarts</strong>: {meta.get('n_restarts', '?')}</li>\n")
        f.write(f"  <li><strong>Seed</strong>: {meta.get('seed', '?')}</li>\n")
        f.write("</ul>\n<hr>\n")

        # Quality metrics
        f.write("<h2>Quality Metrics</h2>\n")
        metrics_text = _get_metrics_text()
        if metrics_text:
            f.write(f"<pre>{metrics_text}</pre>\n")
        f.write("<hr>\n")

        # Design points
        f.write("<h2>Design Points</h2>\n")
        for lbl in ("Prescribed", "Focus", "Optimised"):
            sub = result.samples[result.samples["point_type"] == lbl]
            if sub.empty:
                continue
            f.write(f"<h3>{lbl} ({len(sub)} points)</h3>\n")
            f.write(_df_to_html(sub[names]) + "\n")

        if len(result.validation):
            f.write(f"<h2>Validation Set ({len(result.validation)} points)</h2>\n")
            f.write(_df_to_html(result.validation[names]) + "\n")

        for sname, sdf in getattr(result, 'sets', {}).items():
            if len(sdf):
                f.write(f"<h2>{sname} ({len(sdf)} points)</h2>\n")
                f.write(_df_to_html(sdf[names]) + "\n")

        f.write("</body>\n</html>\n")

    print(f"  Saved: {p}")


# ── Internal: full DataFrame ───────────────────────────────────────────────

def _full_df(result) -> pd.DataFrame:
    """All points in a single DataFrame with point_type column."""
    frames = [result.samples]
    if len(result.validation):
        df = result.validation.copy()
        frames.append(df)
    for sdf in getattr(result, 'sets', {}).values():
        if len(sdf):
            frames.append(sdf.copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
