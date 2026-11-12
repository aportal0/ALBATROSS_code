#!/usr/bin/env python3
"""
Plotting for the deterministic SPEI forecast skill scores.

Separate from the metrics script, so the scoring can run headless
(no cartopy, no matplotlib) and the figures can be redrawn from the
saved NetCDF at any time.

Public entry point:

    plot_scores(out, ratio, n_eff, r_crit, y0, y1,
                fig_dir=..., fig_stem=...)

Style follows the ET0 / SPEI plotting script: Cartopy PlateCarree maps,
50m coastlines, labelled gridlines, 4.4 x 6 inch figures at 300 dpi.
"""

import calendar
import os

import numpy as np

MONTHS = list(calendar.month_abbr)[1:]

# ----------------------------------------------------------------------
# CONFIG - used by the __main__ entry point
# ----------------------------------------------------------------------
INIT_MONTH   = 10
METHOD_LABEL = "Mod-Hargreaves"
YEAR_RANGE = [1993, 2025]

NC_FILE      = ("/home/alice/Desktop/UniBo/data/C3S_seasonal/ecmwf51/monthly/"
                f"init_{INIT_MONTH}/SPEI/SPEI_{METHOD_LABEL}_deterministic_scores_"
                f"lead123_init{INIT_MONTH}_{YEAR_RANGE[0]}-{YEAR_RANGE[1]}_Madagascar.nc")
FIG_DIR      = "/home/alice/Desktop/UniBo/figures/SPEI"
FIG_STEM     = (f"SPEI_{METHOD_LABEL}_"
                "{score}_lead123_"
                f"init{INIT_MONTH}_{YEAR_RANGE[0]}-{YEAR_RANGE[1]}_Madagascar")


# =============================================================================
# Style
# =============================================================================
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt


def format_map(ax, title):
    """Match the ET0 / SPEI plotting script: coastlines + labelled gridlines."""
    ax.set_title(title)
    ax.coastlines(resolution="50m", linewidth=0.6)
    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="0.5",
        alpha=0.5,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False


def get_lon_lat_vals(da):
    """Coordinate names, matching the ET0 / SPEI plotting script."""
    lon_name = "lon" if "lon" in da.coords else "longitude"
    lat_name = "lat" if "lat" in da.coords else "latitude"

    lon = np.asarray(da[lon_name].values)
    lat = np.asarray(da[lat_name].values)
    vals = np.squeeze(np.asarray(da.values))

    return lon, lat, vals


def _map(ax, da, title, cmap, label, vmin=None, vmax=None,
         diverging=False, extend=None):
    """Draw one gridded field as a Cartopy map with a colourbar."""
    import cartopy.crs as ccrs

    lon, lat, vals = get_lon_lat_vals(da)
    kw = {}
    if diverging:
        lim = np.nanmax(np.abs(vals))
        kw = dict(vmin=-lim, vmax=lim)
    elif vmin is not None:
        kw = dict(vmin=vmin, vmax=vmax)

    m = ax.pcolormesh(lon, lat, vals, transform=ccrs.PlateCarree(),
                      cmap=cmap, **kw)
    format_map(ax, title)
    ax.figure.colorbar(m, ax=ax, label=label, shrink=0.8, extend=extend)
    return m


def _new_map_fig(plt):
    import cartopy.crs as ccrs
    return plt.subplots(figsize=(4.4, 6),
                        subplot_kw={"projection": ccrs.PlateCarree()})


# =============================================================================
# Figures
# =============================================================================
def plot_scores(out, ratio, n_eff, r_crit, y0, y1,
                fig_dir="figures", fig_stem=None,
                method_label="Modified Hargreaves", init_month=10,
                src_fcst="", src_obs=""):
    """Write the score maps and summary panels to `fig_dir`.

    Parameters
    ----------
    out : xarray.Dataset
        Score dataset (rmse, corr, spread, spread_rmse_ratio, ...).
    ratio : xarray.DataArray
        spread / RMSE per grid point.
    n_eff : int
        Number of verification cases.
    r_crit : float
        Critical Pearson r at p=0.05 for `n_eff`.
    y0, y1 : int
        First and last init year scored.
    fig_dir : str
        Output directory (created if missing).
    fig_stem : str
        Filename stem with a `{score}` placeholder, e.g.
        "SPEI_Mod-Hargreaves_{score}_lead123_init10_1993-2025_Madagascar".
    """
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D

    plt = setup_mpl()
    os.makedirs(fig_dir, exist_ok=True)

    if fig_stem is None:
        fig_stem = ("SPEI_Mod-Hargreaves_{score}_lead123_init"
                    f"{init_month}_{y0}-{y1}_Madagascar")

    def path(score):
        return os.path.join(fig_dir, fig_stem.format(score=score) + ".png")

    period = f"{y0}-{y1} ({n_eff} init years)"
    init_label = MONTHS[init_month - 1]

    # ---------------- RMSE map -----------------------------------------
    fig, ax = _new_map_fig(plt)
    r = float(np.nanmean(out["rmse"].values))
    _map(ax, out["rmse"],
         f"SPEI forecast RMSE - {method_label}\n"
         f"leads 1-3, init {init_label}, {period}",
         "binary", "RMSE (SPEI units)", extend="max")
    fig.tight_layout()
    fig.savefig(path("rmse"), dpi=300)
    plt.close(fig)
    print(f"  rmse map: domain mean {r:.2f} (no-skill reference 1.0)"
          + (f", source {src_fcst}" if src_fcst else ""))

    # ---------------- correlation map ----------------------------------
    # Cells above the critical r are marked on the same map, so
    # significance is visible without a second plot.
    fig, ax = _new_map_fig(plt)
    corr = out["corr"]
    sig = (corr > r_crit)
    frac = float(np.nanmean(sig.values))
    _map(ax, corr,
         f"SPEI forecast correlation - {method_label}\n"
         f"leads 1-3, init {init_label}, {period}",
         "RdBu_r", "Pearson r", diverging=True, extend="both")

    # Outline the significant region along the exact cell edges.
    #
    # contour() is not used: it interpolates between sample points, so a
    # boundary crossing the grid diagonally is chamfered and rectangles
    # come out with rounded corners. Instead each side of the region is
    # emitted as a straight segment of exactly one cell width, so every
    # corner is a true right angle at a cell corner.
    lon = np.asarray(corr.lon.values)
    lat = np.asarray(corr.lat.values)
    msk = np.isfinite(corr.values) & sig.values
    print(f"  corr map: {int(msk.sum())} of {msk.size} cells significant "
          f"({msk.mean():.1%})")

    if msk.any():
        xs, ys = region_edges(msk, lon, lat)
        ax.plot(xs, ys, color="green", linewidth=0.7,
                solid_capstyle="butt", solid_joinstyle="miter",
                transform=ccrs.PlateCarree(), zorder=5)
        print(f"  corr map: outline drawn as {len(xs) // 3} edge "
              f"segments")

    ax.legend(handles=[Line2D([], [], color="green", linewidth=1,
                              label=f"significant at p<0.05 (r > {r_crit:.2f})")],
              loc="lower left", fontsize=8, frameon=True, framealpha=0.9,
              borderpad=0.4)
    fig.tight_layout()
    fig.savefig(path("corr"), dpi=300)
    plt.close(fig)
    print(f"  corr map: {frac:.0%} of grid points significant at p<0.05 "
          f"(r > {r_crit:.2f})"
          + (f", source {src_obs}" if src_obs else ""))

    # ---------------- spread / RMSE map --------------------------------
    fig, ax = _new_map_fig(plt)
    rr = float(np.nanmean(ratio.values))
    _map(ax, ratio,
         f"SPEI forecast spread / RMSE - {method_label}\n"
         f"leads 1-3, init {init_label}, {period}",
         "PuOr_r", "spread / RMSE (1 = calibrated)", vmin=0.8, vmax=1.2,
         extend="both")
    fig.tight_layout()
    fig.savefig(path("spread_rmse"), dpi=300)
    plt.close(fig)
    print(f"  spread/rmse map: domain mean {rr:.2f}"
          + (f", source {src_fcst}" if src_fcst else ""))

    # ---------------- distributions ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    for ax, values, name, colour, xlabel in (
        (axes[0], out["rmse"], "rmse", "#2F5D9E", "RMSE (SPEI units)"),
        (axes[1], out["corr"], "corr", "#E07B39", "Pearson r"),
        (axes[2], ratio, "spread_rmse_ratio", "#3A9D5D", "spread / RMSE"),
    ):
        v = np.asarray(values.values).ravel()
        v = v[np.isfinite(v)]
        if name == "spread_rmse_ratio":
            v = v[v > 0]
        if not v.size:
            ax.set_visible(False)
            continue
        ax.hist(v, bins=30, color=colour, edgecolor="white", linewidth=0.4)
        ax.axvline(float(np.nanmedian(v)), color="#222", lw=1.4, ls="--",
                   label=f"median {np.nanmedian(v):.2f}")
        if name == "rmse":
            ax.axvline(1.0, color="#B33", lw=1.4, ls=":", label="no-skill 1.0")
        if name == "corr":
            ax.axvline(r_crit, color="#B33", lw=1.4, ls=":",
                       label=f"p=0.05 r = {r_crit:.2f}")
        if name == "spread_rmse_ratio":
            ax.axvline(1.0, color="#B33", lw=1.4, ls=":", label="calibrated 1.0")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Grid points")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Distribution of SPEI skill scores - {method_label}\n"
                 f"{out['rmse'].size} grid points, init {init_label}, "
                 f"{period}")
    fig.tight_layout()
    fig.savefig(path("distributions"), dpi=300)
    plt.close(fig)

    print(f"\nwrote 4 figures to {fig_dir}/")
    for s in ("rmse", "corr", "spread_rmse", "distributions"):
        print(f"  {fig_stem.format(score=s)}.png")


# =============================================================================
# Replot from a saved score file
# =============================================================================
def region_edges(mask, lon, lat):
    """Segments tracing the boundary of a boolean mask, along cell edges.

    Each side of the region is emitted as a straight segment exactly one
    cell wide, so corners land on true cell corners with right angles -
    unlike contour(), which interpolates and rounds them off.

    A side is emitted only where a True cell meets a False or
    out-of-bounds neighbour, so interior edges are skipped and adjacent
    cells merge into one connected region. Segments are separated by NaN
    so separate patches are not joined by a stray line.

    Neighbours are taken in ARRAY order, and the emitted y coordinate is
    the coordinate value for that row, so a latitude axis running either
    north-to-south or south-to-north is handled correctly.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    mask = np.asarray(mask, dtype=bool)

    dlon = float(np.median(np.diff(lon))) if lon.size > 1 else 0.1
    dlat = float(np.median(np.diff(lat))) if lat.size > 1 else 0.1
    hx, hy = dlon / 2.0, dlat / 2.0

    xs, ys = [], []
    ny, nx = mask.shape
    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                continue
            x, y = lon[i], lat[j]
            # west side: neighbour is the previous column
            if i == 0 or not mask[j, i - 1]:
                xs += [x - hx, x - hx, np.nan]
                ys += [y - hy, y + hy, np.nan]
            # east side
            if i == nx - 1 or not mask[j, i + 1]:
                xs += [x + hx, x + hx, np.nan]
                ys += [y - hy, y + hy, np.nan]
            # one side per row neighbour, in array order
            if j == 0 or not mask[j - 1, i]:
                xs += [x - hx, x + hx, np.nan]
                ys += [y - hy, y - hy, np.nan]
            if j == ny - 1 or not mask[j + 1, i]:
                xs += [x - hx, x + hx, np.nan]
                ys += [y + hy, y + hy, np.nan]

    return np.asarray(xs), np.asarray(ys)


def _get_scalar(ds, name):
    """Read a scalar written either as a global attribute or a coordinate.

    The metrics script stores the scalars with assign_coords, so they
    live in ds.coords rather than ds.attrs. Look in both, and in the
    variable metadata as a last resort. Returns nan when absent.
    """
    v = ds.attrs.get(name)
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass

    if name in ds.coords:
        try:
            return float(np.asarray(ds[name].values).ravel()[0])
        except (TypeError, ValueError, IndexError):
            pass

    if name in ds.variables:
        try:
            arr = np.asarray(ds[name].values).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return float(arr[0])
        except (TypeError, ValueError, IndexError):
            pass

    return float("nan")



def plot_from_file(nc_path, fig_dir="figures", fig_stem=None,
                   method_label="Modified Hargreaves", init_month=10):
    """Redraw the figures from a score NetCDF written by the metrics script."""
    import xarray as xr

    out = xr.open_dataset(nc_path)

    y0 = YEAR_RANGE[0]
    y1 = YEAR_RANGE[1]

    # n_eff and r_crit are read from the score file, not recomputed:
    # both are written by the metrics script as scalar coordinates, so
    # the figures use exactly the values the scoring used.
    n_eff = int(_get_scalar(out, "n_cases"))
    if not np.isfinite(n_eff) or n_eff < 3:
        n_eff = y1 - y0 + 1

    r_crit = _get_scalar(out, "r_crit_p05")
    if not np.isfinite(r_crit):
        # older score files: the same threshold is also written as a
        # gridded copy, so any finite value of it will do
        if "corr_r_crit_p05" in out:
            v = np.asarray(out["corr_r_crit_p05"].values).ravel()
            v = v[np.isfinite(v)]
            if v.size:
                r_crit = float(v[0])
    print(f"  read from file: n_eff = {n_eff}, r_crit_p05 = {r_crit:.3f}")

    ratio = out["spread_rmse_ratio"]

    plot_scores(out, ratio, n_eff, r_crit, y0, y1,
                fig_dir=fig_dir, fig_stem=fig_stem,
                method_label=method_label, init_month=init_month,
                src_fcst=out.attrs.get("forecast", ""),
                src_obs=out.attrs.get("observation", ""))


if __name__ == "__main__":
    plot_from_file(NC_FILE,
                   fig_dir=FIG_DIR,
                   fig_stem=FIG_STEM,
                   method_label=METHOD_LABEL,
                   init_month=INIT_MONTH)

