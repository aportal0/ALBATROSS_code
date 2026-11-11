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
NC_FILE      = ("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/monthly/"
                "init_10/SPEI/SPEI_Mod-Hargreaves_deterministic_scores_"
                "lead123_init10_1993-2025_Madagascar.nc")
FIG_DIR      = "/ec/res4/scratch/ecme4047/figures/SPEI"
FIG_STEM     = ("SPEI_Mod-Hargreaves_{score}_lead123_init10"
                "_1993-2025_Madagascar")
INIT_MONTH   = 10
METHOD_LABEL = "Modified Hargreaves"


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
                    f"{init_month}_1993-2025_Madagascar")

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
         "viridis", "RMSE (SPEI units)", extend="max")
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

    # Stippled mask rather than hatching: the cells are ~0.1 deg, far
    # below one pixel at this figure size, so hatch lines collapse.
    lon2d, lat2d = np.meshgrid(corr.lon.values, corr.lat.values)
    msk = np.isfinite(corr.values) & sig.values
    ax.scatter(lon2d[msk], lat2d[msk], s=1.6, c="#101010",
               marker="o", linewidths=0, zorder=4,
               transform=ccrs.PlateCarree())
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none",
                              markersize=3.5, color="#101010",
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
         "PuOr_r", "spread / RMSE (1 = calibrated)", vmin=0, vmax=2,
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

    # ---------------- spread vs RMSE -----------------------------------
    fig, ax = plt.subplots(figsize=(4.4, 6))
    xv = np.asarray(out["rmse"].values).ravel()
    yv = np.asarray(out["spread"].values).ravel()
    m = np.isfinite(xv) & np.isfinite(yv)
    if m.sum():
        ax.scatter(xv[m], yv[m], s=9, alpha=0.45, color="#2F5D9E",
                   edgecolors="none")
        lim = [0, float(np.nanpercentile(
            np.concatenate([xv[m], yv[m]]), 99.5))]
        if lim[1] > 0:
            ax.plot(lim, lim, color="#B33", lw=1.3, ls="--",
                    label="spread = RMSE (calibrated)")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        above = float(np.nanmean(yv[m] > xv[m]))
    else:
        above = float("nan")
    ax.set_xlabel("RMSE (SPEI units)")
    ax.set_ylabel("Ensemble spread (SPEI units)")
    ax.set_title(f"Ensemble spread against RMSE - {method_label}\n"
                 f"one point per grid cell, init {init_label}, {period}")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path("spread_vs_rmse"), dpi=300)
    plt.close(fig)
    print(f"  spread-vs-rmse: spread exceeds RMSE at {above:.0%} of grid "
          f"points (over-dispersive)")

    print(f"\nwrote 5 figures to {fig_dir}/")
    for s in ("rmse", "corr", "spread_rmse", "distributions",
              "spread_vs_rmse"):
        print(f"  {fig_stem.format(score=s)}.png")


# =============================================================================
# Replot from a saved score file
# =============================================================================
def plot_from_file(nc_path, fig_dir="figures", fig_stem=None,
                   method_label="Modified Hargreaves", init_month=10):
    """Redraw the figures from a score NetCDF written by the metrics script."""
    import xarray as xr

    out = xr.open_dataset(nc_path)

    n_eff = int(out.attrs.get("n_cases", out["n_cases"].size))
    r_crit = float(out.attrs.get("r_crit_p05", float("nan")))
    y0 = int(out.attrs.get("first_init_year", 0))
    y1 = int(out.attrs.get("last_init_year", 0))

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

