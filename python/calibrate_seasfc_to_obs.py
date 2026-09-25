#!/usr/bin/env python3
"""
Dry-day frequency over Madagascar: ECMWF SEAS5 (24h tp) vs MSWEP, per month, per grid point.

Run on the HPC where the data lives.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
FC_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/regridded_ERA5-Land")
OB_DIR = Path("/ec/res4/scratch/ecme4047/MSWEP/MSWEP_V316_test/Past/Daily/regridded_ERA5-Land")
OUT_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/calibrated_MSWEP")

REGION       = "Madagascar"
YEARS        = range(1993, 2023)     # 1993..2023 inclusive
INIT_MONTH   = 10                    # starting month of the forecast
N_MONTHS     = 3                     # how many months each forecast file spans

FC_VAR       = "tp"                  # forecast variable name
OB_VAR       = "precipitation"              # MSWEP variable name

FC_THRESH_M  = 0.0002                # dry-day threshold in METRES (forecast)
OB_THRESH_MM = 0.1                   # dry-day threshold in MILLIMETERS (MSWEP)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# months covered by the forecast, with the year offset when it rolls over
TARGET = [
    (((INIT_MONTH - 1 + k) % 12) + 1, (INIT_MONTH - 1 + k) // 12)
    for k in range(N_MONTHS)
]


def dry_fraction(da: xr.DataArray, thresh: float) -> xr.DataArray:
    """Percentage of days below `thresh`, averaged over every non-spatial axis."""
    spatial = [d for d in ("lat", "latitude", "lon", "longitude", "x", "y") if d in da.dims]
    reduce_dims = [d for d in da.dims if d not in spatial]
    valid = da.notnull()
    dry   = (da < thresh) & valid
    return (dry.sum(dim=reduce_dims) / valid.sum(dim=reduce_dims) * 100.0)


# ----------------------------------------------------------------------
# 1. forecast — open all inits, split by calendar month, pool over years
# ----------------------------------------------------------------------
fc_files = []
for y in YEARS:
    f = FC_DIR / f"tp_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}_res_ERA5-Land_patch.nc"
    if f.exists():
        fc_files.append(f)
    else:
        print(f"missing forecast: {f}")

fc_monthly = {}
for month, _ in TARGET:
    parts = []
    for f in fc_files:
        with xr.open_dataset(f) as ds:
            sub = ds[FC_VAR].sel(valid_time=ds["valid_time.month"] == month)
            parts.append(sub.load())
    fc_monthly[month] = xr.concat(parts, dim="forecast_reference_time")
    print(f"forecast {month:02d}: {fc_monthly[month].sizes}")


# ----------------------------------------------------------------------
# 2. MSWEP — same months, same years
# ----------------------------------------------------------------------
ob_monthly = {}
for month, yoff in TARGET:
    parts = []
    for y in YEARS:
        f = OB_DIR / f"precip_daily_{y + yoff}{month:02d}_{REGION}_res_ERA5-Land.nc"
        if not f.exists():
            print(f"missing MSWEP: {f}")
            continue
        with xr.open_dataset(f) as ds:
            parts.append(ds[OB_VAR].load())
    if not parts:
        print(f"no MSWEP data for month {month:02d}")
        continue
    ob_monthly[month] = xr.concat(parts, dim="time")
    print(f"MSWEP {month:02d}: {ob_monthly[month].sizes}")


# ----------------------------------------------------------------------
# 3. dry-day percentage per grid point, per month
# ----------------------------------------------------------------------
results = {}
for month, _ in TARGET:
    fields = {}
    if month in fc_monthly:
        fields["ecmwf51"] = dry_fraction(fc_monthly[month], FC_THRESH_M)
    if month in ob_monthly:
        fields["mswep"] = dry_fraction(ob_monthly[month], OB_THRESH_MM)
    if not fields:
        continue

    out = xr.Dataset(
        {name: da.rename(f"drydays_pct_{name}") for name, da in fields.items()}
    )
    for name in out.data_vars:
        out[name].attrs.update(
            units="%",
            long_name=f"percentage of dry days ({name})",
            threshold=str(FC_THRESH_M if name == "ecmwf51" else OB_THRESH_MM),
        )
    if "ecmwf51" in out and "mswep" in out:
        out["bias"] = out["ecmwf51"] - out["mswep"]
        land_mask = out["ecmwf51"].notnull()
        out["mswep"] = out["mswep"].where(land_mask) 
        out["bias"].attrs.update(units="%", long_name="dry-day bias, forecast minus MSWEP")

    path = OUT_DIR / f"drydays_pct_mon{month:02d}_init{INIT_MONTH:02d}_{REGION}.nc"
    out.to_netcdf(path)
    results[month] = out
    print(f"wrote {path}")

print("done")

