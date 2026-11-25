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

# fixed colourbar boundaries, one list per score
TICKS = {
    "corr":  np.round(np.arange(-0.55, 0.5501, 0.10), 2),   # -0.55 .. 0.55
    "crps":  np.round(np.arange(0.30, 0.8001, 0.05), 2),    #  0.30 .. 0.80
    "crpss": np.round(np.arange(-0.375, 0.3751, 0.05), 3),  # -0.375 .. 0.375
    "rmse":  np.round(np.arange(0.70, 1.2001, 0.05), 2),    #  0.70 .. 1.20
    "rps":   np.round(np.arange(1.50, 2.0001, 0.05), 2),    #  1.50 .. 2.00
    "rpss":  np.round(np.arange(-0.225, 0.2251, 0.03), 3),  # -0.225 .. 0.225
    "spread_rmse": np.round(np.arange(0.75, 1.2501, 0.05), 2),  #  0.75 .. 1.25
    "bs":    np.round(np.arange(0.40, 1.0001, 0.05), 2),    #  0.40 .. 1.00
    "bss":   np.round(np.arange(-0.30, 0.3001, 0.05), 3),   # -0.30 .. 0.30
}

# ----------------------------------------------------------------------
# CONFIG - used by the __main__ entry point
# ----------------------------------------------------------------------
INIT_MONTH   = 10
COUNTRY = "Madagascar"
METHOD_LABEL = "Mod-Hargreaves"
YEAR_RANGE = [1993, 2025]
LEAD_STR = "123"

_DIR_SCORES = ("/home/alice/Desktop/UniBo/data/C3S_seasonal/ecmwf51/monthly/"
               f"init_{INIT_MONTH}/SPEI/")

NC_FILE       = (_DIR_SCORES
                 + f"SPEI_{METHOD_LABEL}_deterministic_scores_lead{LEAD_STR}"
                 f"_init{INIT_MONTH}_{YEAR_RANGE[0]}-{YEAR_RANGE[1]}_{COUNTRY}.nc")
NC_FILE_PROB  = (_DIR_SCORES
                 + f"SPEI_{METHOD_LABEL}_probabilistic_scores_lead{LEAD_STR}"
                 f"_init{INIT_MONTH}_{YEAR_RANGE[0]}-{YEAR_RANGE[1]}_{COUNTRY}.nc")
# the mask is read from the forecast file: its NaN footprint is the
# land/sea mask for this grid, and it is the reference the user trusts
NC_FILE_MASK = (_DIR_SCORES
                + f"SPEI_{METHOD_LABEL}_lead{LEAD_STR}_init{INIT_MONTH}"
                f"_1993-2025_{COUNTRY}_res_ERA5-Land_qdm_ERA5-Land_MSWEP.nc")

FIG_DIR      = "/home/alice/Desktop/UniBo/figures/SPEI"
FIG_STEM     = (f"SPEI_{METHOD_LABEL}_"
                "{score}_"
                f"lead{LEAD_STR}_init{INIT_MONTH}_{YEAR_RANGE[0]}-{YEAR_RANGE[1]}_{COUNTRY}")


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
         diverging=False, extend=None, ticks=None,
         under_shade=0.45, over_shade=0.45, lighten_under=False):
    """Draw one gridded field as a Cartopy map with a colourbar.

    `ticks` gives the colourbar boundaries explicitly; the field is
    drawn with a BoundaryNorm so the colours step at those values and
    the bar is labelled with the same list.

    `under_shade` / `over_shade` darken the out-of-range triangles (the
    two ends shown when extend="both") to 45% brightness by default.
    Taking a darker shade of each end colour keeps the sense of which
    direction is which while making clear the value lies beyond the
    scale - a flat grey would lose that, and the plain end colour would
    read as in-range. Set either to None to leave that end as it is.

    A colormap whose ends are white and black cannot show this: there is
    no darker shade of white, and black has nowhere to go. Use a
    sequential colormap with real hue at both ends instead.

    `lighten_under` flips the UNDER arrow to a lighter shade instead,
    for a scale where the low end is the pale one and going lighter is
    what plainly says "beyond the scale". Use it where the low end is
    already dark.
    """
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm

    lon, lat, vals = get_lon_lat_vals(da)
    kw = {}
    if ticks is not None:
        levels = np.asarray(ticks, dtype=float)
        kw = dict(norm=BoundaryNorm(levels, ncolors=256, clip=False))
    elif diverging:
        lim = np.nanmax(np.abs(vals))
        kw = dict(vmin=-lim, vmax=lim)
    elif vmin is not None:
        kw = dict(vmin=vmin, vmax=vmax)

    # The over / under colours are set on the colormap BEFORE the
    # colourbar is built: the bar takes its own copy at construction, so
    # changing them afterwards has no effect on what is drawn. A copy is
    # used so the map's own colours are untouched.
    import matplotlib.pyplot as _plt

    cmap_obj = _plt.colormaps[cmap] if isinstance(cmap, str) else cmap
    cmap_obj = cmap_obj.copy()

    def _shade(rgba, f, lighter):
        r, g, b, a = rgba
        if lighter:
            # move toward white rather than black
            return (r + (1.0 - r) * (1.0 - f),
                    g + (1.0 - g) * (1.0 - f),
                    b + (1.0 - b) * (1.0 - f), a)
        return (r * f, g * f, b * f, a)

    if under_shade is not None:
        cmap_obj.set_under(_shade(cmap_obj(0.0), under_shade, lighten_under))
    if over_shade is not None:
        cmap_obj.set_over(_shade(cmap_obj(1.0), over_shade, False))

    m = ax.pcolormesh(lon, lat, vals, transform=ccrs.PlateCarree(),
                      cmap=cmap_obj, **kw)
    format_map(ax, title)
    cb = ax.figure.colorbar(m, ax=ax, label=label, shrink=0.8,
                            extend=extend, ticks=ticks)

    if ticks is not None:
        cb.ax.tick_params(labelsize=8)
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
        "SPEI_Mod-Hargreaves_{score}_lead123_init10_1993-2025_{COUNTRY}".
    """
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D

    plt = setup_mpl()
    os.makedirs(fig_dir, exist_ok=True)

    if fig_stem is None:
        fig_stem = ("SPEI_{METHOD_LABEL}_{score}_lead{LEAD_STR}_init"
                    f"{init_month}_{y0}-{y1}_{COUNTRY}")

    def path(score):
        return os.path.join(fig_dir, fig_stem.format(score=score) + ".png")

    period = f"{y0}-{y1} ({n_eff} init years)"
    init_label = MONTHS[init_month - 1]

    # ---------------- RMSE map -----------------------------------------
    fig, ax = _new_map_fig(plt)
    r = float(np.nanmean(out["rmse"].values))
    _map(ax, out["rmse"],
         f"SPEI forecast RMSE - {method_label}\n"
         f"leads {LEAD_STR}, init {init_label}, {period}",
         "YlOrRd", "RMSE (SPEI units)", extend="both", ticks=TICKS["rmse"],
         under_shade=0.5, lighten_under=True)
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
         f"leads {LEAD_STR}, init {init_label}, {period}",
         "RdBu_r", "Pearson r", diverging=True, extend="both",
         ticks=TICKS["corr"])

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
         f"leads {LEAD_STR}, init {init_label}, {period}",
         "PuOr_r", "spread / RMSE (1 = calibrated)", extend="both",
         ticks=TICKS["spread_rmse"])
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



def plot_probabilistic_from_file(nc_path, fig_dir="figures", fig_stem=None,
                                 method_label="Modified Hargreaves",
                                 init_month=10):
    """Draw the probabilistic score maps from a score NetCDF.

    Reads the file written by prob_metrics.py. Only the variables that
    file actually contains are drawn, so this is safe to point at a
    deterministic score file too (it will simply draw nothing).
    """
    import xarray as xr

    out = xr.open_dataset(nc_path)

    have = set(out.data_vars)
    if not (have & {"crps", "crpss", "rps", "rpss"}):
        print(f"  {nc_path}: no probabilistic variables found "
              f"(has {sorted(have)})")
        return

    y0 = YEAR_RANGE[0]
    y1 = YEAR_RANGE[1]
    n_eff = int(_get_scalar(out, "n_cases"))
    if not np.isfinite(n_eff) or n_eff < 3:
        n_eff = y1 - y0 + 1

    plot_probabilistic(out, n_eff, y0, y1,
                       fig_dir=fig_dir, fig_stem=fig_stem,
                       method_label=method_label, init_month=init_month,
                       src_fcst=out.attrs.get("forecast", ""),
                       src_obs=out.attrs.get("observation", ""))


def _land_mask(out, mask_path=None, var="SPEI"):
    """Boolean land mask: True where the forecast SPEI is not NaN.

    Read from the forecast file (NC_FILE_MASK), whose NaN footprint is
    the land/sea mask for this grid. A grid point where the forecast is
    masked is water, so every score plotted there is blanked.

    If the file cannot be read or has no such variable, fall back to the
    mask implied by the score file itself rather than blanking the whole
    map, and say so.
    """
    import xarray as xr

    if mask_path:
        try:
            ds = xr.open_dataset(mask_path)
            if var in ds:
                da = ds[var]
                # collapse any non-spatial dims (member, init_year) to one
                # 2-D field: a point is land if any member/year has data
                if da.ndim > 2:
                    da = da.isel({d: 0 for d in da.dims
                                  if d not in ("lat", "lon")}, drop=True)
                v = np.asarray(da.values, dtype=float)
                return xr.DataArray(np.isfinite(v), coords=da.coords,
                                    dims=da.dims)
            print(f"  ! {var} not found in {mask_path}")
        except Exception as e:                            # noqa: BLE001
            print(f"  ! could not read the mask from {mask_path}: {e}")

    # fallback: any score field that is already masked
    for name in ("crps", "rps", "bs", "p_below"):
        if name in out:
            v = np.asarray(out[name].values, dtype=float)
            if np.isnan(v).any():
                print(f"  ! land mask taken from {name} instead "
                      f"(score file), not from the forecast file")
                return xr.DataArray(np.isfinite(v), coords=out[name].coords,
                                    dims=out[name].dims)

    ref = next(iter(out.data_vars.values()))
    return xr.DataArray(np.ones(ref.shape, dtype=bool),
                        coords=ref.coords, dims=ref.dims)


def _masked(da, mask):
    """Set a field to NaN where the mask is False, for plotting only."""
    return da.where(mask)


def plot_probabilistic(out, n_eff, y0, y1,
                       fig_dir="figures", fig_stem=None,
                       method_label="Modified Hargreaves", init_month=10,
                       src_fcst="", src_obs=""):
    """Score maps and summary panels for the probabilistic metrics.

    out : xarray.Dataset with crps, crpss, rps, rpss, bs, bss,
          p_below, p_normal, p_above.
    """
    plt = setup_mpl()
    os.makedirs(fig_dir, exist_ok=True)

    if fig_stem is None:
        fig_stem = (f"SPEI_{method_label}_{{score}}_lead{LEAD_STR}_init"
                    f"{init_month}_{y0}-{y1}_{COUNTRY}")

    def path(score):
        return os.path.join(fig_dir, fig_stem.format(score=score) + ".png")

    period = f"{y0}-{y1} ({n_eff} init years)"
    init_label = MONTHS[init_month - 1]

    # Ocean points carry a score of NaN in the error fields already; this
    # makes the skill scores match, so the three of them read the same
    # way. The file is not modified - the mask is applied to the copy
    # used for plotting.
    mask = _land_mask(out, mask_path=NC_FILE_MASK)
    n_land = int(mask.values.sum())
    n_all = int(mask.size)
    print(f"  land mask: {n_land} of {n_all} grid points "
          f"({n_land / max(n_all, 1):.1%})")

    # ---------------- CRPS map ------------------------------------------
    # proper score, so lower is better; like RMSE it has no natural upper
    # bound, so the colour scale is taken from the data itself
    if "crps" in out:
        fig, ax = _new_map_fig(plt)
        c = float(np.nanmean(out["crps"].values))
        _map(ax, _masked(out["crps"], mask),
             f"SPEI forecast CRPS - {method_label}\n"
             f"leads {LEAD_STR}, init {init_label}, {period}",
             "viridis", "CRPS (SPEI units)", extend="both",
             ticks=TICKS["crps"])
        fig.tight_layout()
        fig.savefig(path("crps"), dpi=300)
        plt.close(fig)
        print(f"  crps map: domain mean {c:.3f}"
              + (f", source {src_fcst}" if src_fcst else ""))

    # ---------------- CRPSS map -----------------------------------------
    # skill score: 0 is no skill, so the scale is centred on zero and the
    # colour range is symmetric so positive and negative read alike
    if "crpss" in out:
        fig, ax = _new_map_fig(plt)
        v = out["crpss"].values
        frac = float(np.nanmean(v > 0))
        _map(ax, _masked(out["crpss"], mask),
             f"SPEI forecast CRPSS - {method_label}\n"
             f"leads {LEAD_STR}, init {init_label}, {period}",
             "RdBu_r", "CRPSS (-)", diverging=True, extend="both",
             ticks=TICKS["crpss"])
        fig.tight_layout()
        fig.savefig(path("crpss"), dpi=300)
        plt.close(fig)
        print(f"  crpss map: skill > 0 at {frac:.0%} of grid points"
              + (f", source {src_fcst}" if src_fcst else ""))

    # ---------------- RPS map -------------------------------------------
    if "rps" in out:
        fig, ax = _new_map_fig(plt)
        r = float(np.nanmean(out["rps"].values))
        _map(ax, _masked(out["rps"], mask),
             f"SPEI forecast RPS over terciles - {method_label}\n"
             f"leads {LEAD_STR}, init {init_label}, {period}",
             "viridis", "RPS (-)", extend="both", ticks=TICKS["rps"])
        fig.tight_layout()
        fig.savefig(path("rps"), dpi=300)
        plt.close(fig)
        print(f"  rps map: domain mean {r:.3f}")

    # ---------------- RPSS map ------------------------------------------
    if "rpss" in out:
        fig, ax = _new_map_fig(plt)
        v = out["rpss"].values
        frac = float(np.nanmean(v > 0))
        _map(ax, _masked(out["rpss"], mask),
             f"SPEI forecast RPSS over terciles - {method_label}\n"
             f"leads {LEAD_STR}, init {init_label}, {period}",
             "RdBu_r", "RPSS (-)", diverging=True, extend="both",
             ticks=TICKS["rpss"])
        fig.tight_layout()
        fig.savefig(path("rpss"), dpi=300)
        plt.close(fig)
        print(f"  rpss map: skill > 0 at {frac:.0%} of grid points")

    # ---------------- BS map --------------------------------------------
    # like RPS, an error not a skill score: lower is better. The scale
    # runs 0 to 2 over three categories but most of the range is empty,
    # so the ticks start at 0.40 rather than at 0.
    if "bs" in out:
        fig, ax = _new_map_fig(plt)
        b = float(np.nanmean(out["bs"].values))
        _map(ax, _masked(out["bs"], mask),
             f"SPEI forecast Brier score over terciles - {method_label}\n"
             f"leads {LEAD_STR}, init {init_label}, {period}",
             "viridis", "BS (-)", extend="both", ticks=TICKS["bs"])
        fig.tight_layout()
        fig.savefig(path("bs"), dpi=300)
        plt.close(fig)
        print(f"  bs map: domain mean {b:.3f}")

    # ---------------- BSS map -------------------------------------------
    # skill score: 0 is no skill, centred on zero and symmetric
    if "bss" in out:
        fig, ax = _new_map_fig(plt)
        v = out["bss"].values
        frac = float(np.nanmean(v > 0))
        _map(ax, _masked(out["bss"], mask),
             f"SPEI forecast Brier skill score - {method_label}\n"
             f"leads {LEAD_STR}, init {init_label}, {period}",
             "RdBu_r", "BSS (-)", diverging=True, extend="both",
             ticks=TICKS["bss"])
        fig.tight_layout()
        fig.savefig(path("bss"), dpi=300)
        plt.close(fig)
        print(f"  bss map: skill > 0 at {frac:.0%} of grid points")

    # ---------------- distributions -------------------------------------
    panels = [(n, "#2F5D9E", u) for n, u in (
        ("crps", "CRPS (SPEI units)"),
        ("crpss", "CRPSS (-)"),
        ("rpss", "RPSS (-)"),
        ("bs", "BS (-)"),
        ("bss", "BSS (-)"),
    ) if n in out]

    if panels:
        ncol = len(panels)
        # six panels at 4.4 in each would be unwieldy, so wrap to two rows
        nrow = 1 if ncol <= 3 else 2
        ncol_grid = (ncol + nrow - 1) // nrow
        fig, axes = plt.subplots(nrow, ncol_grid,
                                 figsize=(4.4 * ncol_grid, 4.5 * nrow),
                                 squeeze=False)
        axes = axes.ravel()
        colours = ["#2F5D9E", "#E07B39", "#3A9D5D", "#8E5EA2"]
        # hide any grid cell left over when panels do not fill the grid
        for ax in axes[len(panels):]:
            ax.set_visible(False)
        for ax, (name, _c, xlabel), colour in zip(axes, panels,
                                                  colours * 2):
            vals = np.asarray(out[name].values).ravel()
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                ax.set_visible(False)
                continue
            ax.hist(vals, bins=30, color=colour, edgecolor="white",
                    linewidth=0.4)
            ax.axvline(float(np.median(vals)), color="#222", lw=1.4,
                       ls="--", label=f"median {np.median(vals):.2f}")
            if name in ("crpss", "rpss", "bss"):
                ax.axvline(0.0, color="#B33", lw=1.4, ls=":",
                           label="no skill 0")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Grid points")
            ax.legend(fontsize=8, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Distribution of probabilistic SPEI scores - "
                     f"{method_label}\n{out['crps'].size if 'crps' in out else ''}"
                     f" grid points, init {init_label}, {period}")
        fig.tight_layout()
        fig.savefig(path("prob_distributions"), dpi=300)
        plt.close(fig)

    # ---------------- tercile probabilities -----------------------------
    if {"p_below", "p_normal", "p_above"} <= set(out.data_vars):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
        lo, hi = _get_scalar(out, "tercile_edge_low"), \
            _get_scalar(out, "tercile_edge_high")
        for ax, name, title, cmap in (
            (axes[0], "p_below", f"P(SPEI < {lo:.2f})", "Reds"),
            (axes[1], "p_normal", f"P({lo:.2f} to {hi:.2f})", "Greens"),
            (axes[2], "p_above", f"P(SPEI > {hi:.2f})", "Blues"),
        ):
            lon, lat, vals = get_lon_lat_vals(out[name])
            m = ax.pcolormesh(lon, lat, vals, cmap=cmap,
                              vmin=0.0, vmax=1.0)
            ax.set_title(title)
            ax.set_xlabel("Longitude (degrees east)")
            ax.set_ylabel("Latitude (degrees north)")
            ax.set_aspect("equal")
            ax.spines[["top", "right"]].set_visible(False)
            ax.figure.colorbar(m, ax=ax, label="probability", shrink=0.8)
        fig.suptitle(f"Mean forecast tercile probabilities - {method_label}\n"
                     f"leads {LEAD_STR}, init {init_label}, {period}")
        fig.tight_layout()
        fig.savefig(path("tercile_probs"), dpi=300)
        plt.close(fig)

    print(f"\nwrote probabilistic figures to {fig_dir}/")


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

    plot_probabilistic_from_file(NC_FILE_PROB,
                                 fig_dir=FIG_DIR,
                                 fig_stem=FIG_STEM,
                                 method_label=METHOD_LABEL,
                                 init_month=INIT_MONTH)

