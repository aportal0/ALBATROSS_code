import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# -----------------------------
# Parameters
# -----------------------------
YEAR_RANGE = [1993, 2022]

MONTH_INIT = 10
MONTH_LEADS = [0, 1, 2]

COUNTRY = "Madagascar"

VARS = ["ecmwf51", "mswep", "bias"]
THR_LABEL = "thr4"

OUT_DIR = "/ec/res4/scratch/ecme4047/figures/model_calibration"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Fixed colour scales (built once, shared by every figure)
# -----------------------------
# percentages: 0-100 in steps of 10 -> 10 bins, 11 boundaries
pct_bounds = np.arange(0, 101, 10)
pct_cmap = plt.get_cmap("viridis", len(pct_bounds) - 1)
pct_norm = BoundaryNorm(pct_bounds, ncolors=pct_cmap.N, clip=True)

# bias: -100 to +100 in steps of 20 -> 10 bins, 11 boundaries
bias_bounds = np.arange(-100, 101, 20)
bias_cmap = plt.get_cmap("RdBu_r", len(bias_bounds) - 1)
bias_norm = BoundaryNorm(bias_bounds, ncolors=bias_cmap.N, clip=True)


# -----------------------------
# Helpers
# -----------------------------
def get_threshold_mm(da, unit_scale):
    """
    Read the dry-day threshold attribute and normalise it to mm/day.
    ecmwf51 reports metres -> scale 1000; MSWEP reports mm -> scale 1.
    """
    thr = da.attrs.get("threshold")
    if thr is None:
        return None
    return float(thr) * unit_scale


def build_ncfile(month_init, lead, country):
    month = month_init + lead
    return (
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{month_init}/"
        f"tp/calibrated_MSWEP/"
        f"drydays_pct_mon{month}_init{month_init}_{YEAR_RANGE[0]}-{YEAR_RANGE[1]}_{country}_{THR_LABEL}.nc"
    )


def plot_one(ncfile, month_init, lead, country):
    ds = xr.open_dataset(ncfile)
    da_fc, da_obs, da_bias = (ds[v] for v in VARS)

    lat = ds["lat"].values
    lon = ds["lon"].values

    thr_fc_mm  = get_threshold_mm(da_fc, 1000.0)   # m  -> mm
    thr_obs_mm = get_threshold_mm(da_obs, 1.0)     # mm -> mm

    # -----------------------------
    # Figure
    # -----------------------------
    fig, axes = plt.subplots(
        1, 3,
        figsize=(9, 4.6),
        constrained_layout=True,
        sharey=True,
    )

    im0 = axes[0].pcolormesh(lon, lat, da_fc.values,
                             cmap=pct_cmap, norm=pct_norm, shading="auto")
    im1 = axes[1].pcolormesh(lon, lat, da_obs.values,
                             cmap=pct_cmap, norm=pct_norm, shading="auto")
    im2 = axes[2].pcolormesh(lon, lat, da_bias.values,
                             cmap=bias_cmap, norm=bias_norm, shading="auto")

    # -----------------------------
    # Labels, titles, colourbars
    # -----------------------------
    titles = [
        da_fc.attrs.get("long_name", VARS[0]),
        da_obs.attrs.get("long_name", VARS[1]),
        da_bias.attrs.get("long_name", VARS[2]),
    ]
    if thr_fc_mm is not None:
        titles[0] = f"{titles[0]}\n(thr = {thr_fc_mm:g} mm/day)"
    if thr_obs_mm is not None:
        titles[1] = f"{titles[1]}\n(thr = {thr_obs_mm:g} mm/day)"

    for ax, da, title in zip(axes, (da_fc, da_obs, da_bias), titles):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal")

    axes[0].set_ylabel("Latitude")

    cb0 = fig.colorbar(im0, ax=axes[:2], location="right", shrink=0.85, pad=0.02,
                       ticks=pct_bounds, spacing="uniform")
    cb0.set_label(da_fc.attrs.get("units", "%"))

    cb2 = fig.colorbar(im2, ax=axes[2], location="right", shrink=0.85, pad=0.02,
                       ticks=bias_bounds, spacing="uniform")
    cb2.set_label(da_bias.attrs.get("units", "%"))

    month = month_init + lead
    fig.suptitle(
        f"Dry-day percentage and bias - init {month_init:02d}, "
        f"month {month:02d} (lead {lead}) - {country}"
    )

    out = (f"{OUT_DIR}/dry_day_bias_{VARS[0]}_mon{month}_init{month_init}_"
           f"{YEAR_RANGE[0]}-{YEAR_RANGE[1]}_{country}_{THR_LABEL}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)                      # release memory across iterations
    print(f"Saved {out}")

    ds.close()                          # close the file handle each iteration


# -----------------------------
# Main
# -----------------------------
for lead in MONTH_LEADS:
    ncfile = build_ncfile(MONTH_INIT, lead, COUNTRY)

    if not os.path.exists(ncfile):
        print(f"Missing input, skipping: {ncfile}")
        continue

    plot_one(ncfile, MONTH_INIT, lead, COUNTRY)

