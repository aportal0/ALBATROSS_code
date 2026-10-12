#!/usr/bin/env python3
"""
Side-by-side plot: RAW SEAS5 forecast  vs  QDM-corrected field, SAME day.

    left  panel : tp_24h_<version>_init<YYYYMM>_<region>_res_<resolution>_patch.nc
                  raw forecast, units "m", forecast_period in HOURS
    right panel : tp_24h_mon<MM>_init<MM>_2023-2025_<region>_res_<resolution>_qdm_MSWEP.nc
                  QDM-corrected, units "mm/day", forecast_period in DAYS

HOW THE TWO ARE ALIGNED -- from ncdump -c of both files:

    raw      forecast_reference_time = 1696118400  (= 2023-10-01 00 UTC)
             forecast_period = 24, 48, 72, ...      hours since init
             tp units = "m"

    corrected  year = 0, 1, 2
               forecast_period = 10, 11, 12, ..., 30  DAYS since init
               tp_qdm units = "mm/day"

    In BOTH files forecast_period is the LEAD -- hours in the raw file, days in
    the corrected one. The same lead is therefore selected by

        raw      : forecast_period = LEAD_DAYS * 24   (hours)
        corrected: forecast_period = LEAD_DAYS        (days)

    With an October init and SKIP_LEAD_DAYS = 10, the corrected mon10 file starts
    at lead 10 (11 Oct) and ends at lead 30 (31 Oct); the raw file covers the
    whole window from lead 1. The calendar date of a lead is

        init date (1 Oct of the verification year) + LEAD_DAYS

Run:
    python plot_raw_vs_qdm.py
"""

from pathlib import Path
from datetime import date, timedelta
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")          # no display on a compute node
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# ----------------------------------------------------------------------
# configuration -- edit these
# ----------------------------------------------------------------------
INIT_YEAR  = 2025          # initialization year of the RAW file
INIT_MONTH = 10            # initialization month (October)

LEAD_DAYS  = 35            # LEAD in days since the initialization:
                           #   raw       -> forecast_period = LEAD_DAYS * 24 h
                           #   corrected -> forecast_period = LEAD_DAYS days

TARGET_MONTH = 11          # corrected file to open: mon10, mon11 or mon12
MEMBER       = 5           # ensemble member index

YEAR_INDEX = INIT_YEAR - 2023            # corrected file: 0, 1, 2 -> verification years

REGION     = "Madagascar"
VERSION    = "ecmwf51"
RESOLUTION = "ERA5-Land"
VERIF_YEARS = [2023, 2024, 2025]

VMAX = None                # shared colour max in mm/day; None -> 99th pct
CMAP = "YlGnBu"

# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
RAW_DIR = Path(f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}/tp/"
               "regridded_ERA5-Land")
QDM_DIR = Path(f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}/tp/"
               "calibrated_MSWEP")

RAW_VAR = "tp"
QDM_VAR = "tp_qdm"

FC_TIME_DIM = "forecast_period"
FC_SELECT   = "valid_time"       # coordinate on forecast_period (raw file only)
MEMBER_DIM  = "number"
LAT, LON    = "lat", "lon"

RAW_TO_MM = 1000.0               # raw tp in metres -> mm/day
QDM_TO_MM = 1.0                  # corrected file already mm/day
PLOT_UNITS = "mm/day"

OUT_DIR = Path("/ec/res4/scratch/ecme4047/figures/model_calibration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / (f"tp_24h_{VERSION}_init{INIT_YEAR:04d}{INIT_MONTH:02d}_"
                      f"{REGION}_res_{RESOLUTION}_patch.nc")
QDM_FILE = QDM_DIR / (f"tp_24h_mon{TARGET_MONTH:02d}_init{INIT_MONTH:02d}_"
                      f"{VERIF_YEARS[0]}-{VERIF_YEARS[-1]}_{REGION}_"
                      f"res_{RESOLUTION}_qdm_MSWEP.nc")


def load_raw(lead_days=LEAD_DAYS, member=MEMBER, path=RAW_FILE):
    """
    RAW forecast field at the given LEAD (in days since init).

    forecast_period in the raw file is in HOURS, so the lead is converted with
    * 24. Selection is by the dimension value directly -- no date arithmetic and
    no valid_time matching, which is what made the earlier version pick the
    wrong day.
    Returns (field, valid_time string, lat, lon).
    """
    if not path.exists():
        raise FileNotFoundError(f"raw file not found: {path}")
    ds = xr.open_dataset(path)

    target_hours = int(lead_days) * 24
    fp_hours = ds[FC_TIME_DIM].values / np.timedelta64(1, "h")
    hit = np.where(fp_hours == target_hours)[0]
    if hit.size == 0:
        raise ValueError(
            f"raw file has no step with forecast_period = {target_hours} h "
            f"(lead {lead_days} d). Available hours: "
            f"{fp_hours.astype(int)[:5].tolist()} ... "
            f"{fp_hours.astype(int)[-3:].tolist()}"
        )

    da = ds[RAW_VAR].isel({FC_TIME_DIM: int(hit[0])})
    da = da.squeeze(drop=True)                 # drop forecast_reference_time etc.

    n_mem = ds.sizes[MEMBER_DIM]
    if not (0 <= member < n_mem):
        raise ValueError(f"member {member} out of range 0..{n_mem - 1}")
    da = da.isel({MEMBER_DIM: member}).squeeze(drop=True)

    if da.ndim != 2:
        raise ValueError(f"unexpected raw field dims {da.dims} shape {da.shape}")

    field = np.asarray(da.values, dtype=np.float64) * RAW_TO_MM
    vt_str = (str(ds[FC_SELECT].isel({FC_TIME_DIM: int(hit[0])}).values)[:16]
              if FC_SELECT in ds.coords else "")
    lat, lon = ds[LAT].values, ds[LON].values
    ds.close()
    return field, vt_str, lat, lon


def load_qdm(lead_days=LEAD_DAYS, member=MEMBER, year_index=YEAR_INDEX,
             path=QDM_FILE):
    """
    QDM-corrected field at the given LEAD.

    forecast_period in the corrected file is ALREADY the lead in days (its
    values run 10..30 for mon10), so the selection is a direct match on the
    dimension value.
    Returns (field, lead, year value, lat, lon).
    """
    if not path.exists():
        raise FileNotFoundError(f"corrected file not found: {path}")
    ds = xr.open_dataset(path)

    fp_days = ds[FC_TIME_DIM].values / np.timedelta64(1, "D")
    hit = np.where(fp_days == int(lead_days))[0]
    if hit.size == 0:
        raise ValueError(
            f"corrected file has no lead {lead_days} d; "
            f"forecast_period runs {fp_days.astype(int)[0]}..{fp_days.astype(int)[-1]}. "
            f"Note: the initialization month loses its first days to the "
            f"spin-up cut"
        )

    n_mem = ds.sizes[MEMBER_DIM]
    if not (0 <= member < n_mem):
        raise ValueError(f"member {member} out of range 0..{n_mem - 1}")

    sub = ds[QDM_VAR].isel(year=year_index, **{FC_TIME_DIM: int(hit[0])},
                           **{MEMBER_DIM: member}).squeeze(drop=True)
    field = np.asarray(sub.values, dtype=np.float64) * QDM_TO_MM
    year_val = int(ds["year"].isel(year=year_index).values)
    lat, lon = ds[LAT].values, ds[LON].values
    ds.close()
    return field, int(lead_days), year_val, lat, lon


def main():
    # calendar date the lead corresponds to: init on 1 Oct of the raw file's year
    init_d = date(INIT_YEAR, INIT_MONTH, 1)
    lead_date = init_d + timedelta(days=int(LEAD_DAYS))

    print(f"raw file       : {RAW_FILE}")
    print(f"corrected file : {QDM_FILE}")
    print(f"lead {LEAD_DAYS} d -> raw forecast_period {LEAD_DAYS * 24} h, "
          f"corrected forecast_period {LEAD_DAYS} d")
    print(f"calendar date  : {lead_date.isoformat()}\n")

    raw, raw_vt, lat_r, lon_r = load_raw()
    print(f"raw  : valid_time {raw_vt}  member {MEMBER}  "
          f"min={np.nanmin(raw):.2f} max={np.nanmax(raw):.2f} "
          f"mean={np.nanmean(raw):.2f} mm/day")

    qdm, lead, year_val, lat_q, lon_q = load_qdm()
    print(f"qdm  : lead {lead} d  year {year_val}  member {MEMBER}  "
          f"min={np.nanmin(qdm):.2f} max={np.nanmax(qdm):.2f} "
          f"mean={np.nanmean(qdm):.2f} mm/day\n")

    if raw.shape != qdm.shape:
        raise ValueError(f"grid mismatch: raw {raw.shape} vs qdm {qdm.shape}")

    vmax = (float(np.nanpercentile(np.concatenate([raw.ravel(), qdm.ravel()]), 99))
            or 1.0) if VMAX is None else float(VMAX)
    print(f"shared colour scale: 0 .. {vmax:.2f} mm/day\n")

    fig, axes = plt.subplots(1, 2, figsize=(7, 6.3), dpi=140,
            subplot_kw={"projection": ccrs.PlateCarree()})

    for ax, field, lat, lon, title in [
        (axes[0], raw, lat_r, lon_r,
         f"RAW SEAS5 tp - init {INIT_YEAR:04d}-{INIT_MONTH:02d}\n"
         f"lead {LEAD_DAYS} d ({lead_date.isoformat()}) - member {MEMBER}\n"
         f"mean {np.nanmean(raw):.2f} mm/day"),
        (axes[1], qdm, lat_q, lon_q,
         f"QDM-corrected vs MSWEP - init {INIT_MONTH:02d}, mon{TARGET_MONTH:02d}\n"
         f"lead {LEAD_DAYS} d ({lead_date.isoformat()}) - "
         f"member {MEMBER} (year {year_val})\n"
         f"mean {np.nanmean(qdm):.2f} mm/day"),
    ]:
        im = ax.pcolormesh(lon, lat, field, cmap=CMAP,
                           vmin=0, vmax=vmax, shading="auto",
                           transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
                crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("longitude (degrees_east)")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("latitude (degrees_north)")
    
    # dedicate axes for the colorbar, placed in the figure coordinates
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])   # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(f"tp ({PLOT_UNITS})")

    fig.suptitle(f"Daily precipitation at lead {LEAD_DAYS} d "
                 f"({lead_date.isoformat()}) - raw vs QDM-corrected "
                 f"(member {MEMBER})", fontsize=11)

    out = OUT_DIR / (f"{RAW_VAR}_raw_vs_qdm_lead{LEAD_DAYS}d_mon{TARGET_MONTH:02d}_"
                     f"init{INIT_YEAR}{INIT_MONTH:02d}_mem{MEMBER}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

