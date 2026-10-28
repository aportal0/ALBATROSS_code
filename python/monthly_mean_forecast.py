#!/usr/bin/env python3
"""
Monthly mean of QDM-adjusted daily tmin forecasts.

Edit the parameters below, then run:  python monthly_mean.py
"""

import os

import numpy as np
import xarray as xr

# ---------------------------------------------------------------- parameters
month_init = 10                     # target init month, e.g. 11, 12, 1
country    = "Madagascar"           # region tag used in the filename
year_range = [1993, 2025]           # [y0, yN] inclusive
variable   = "mn2t24"               # variable name; also names the data folder
# ---------------------------------------------------------------------------

VAR = variable + "_qdm"             # array name inside the file
ROOT = os.path.join("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{month_init}/", variable)   # path/<variable>/


def build_path(month, country, y0, y1):
    if variable == "mn2t24" or variable == "mx2t24":
        suffix = "res_ERA5-Land_qdm_ERA5-Land"
    elif variable == "tp":
        suffix = "res_ERA5-Land_qdm_MSWEP"
    name = (f"{variable}_24h_mon{month:02d}_init{month_init:02}_{y0}-{y1}_"
            f"{country}_{suffix}.nc")
    return os.path.join(ROOT, name)


def load_one(path):
    ds = xr.open_dataset(path)
    if "forecast_period" in ds.dims:
        ds = ds.rename({"forecast_period": "lead"})
    return ds


def monthly_mean(ds, varname=VAR, weights=None):
    """
    Mean over every dimension except lat/lon.

    Dims (year, lead, number, lat, lon) -> a single (lat, lon) climatology.
    Pass `weights` if the lead days are not equally weighted.
    """
    dims = [d for d in ds[varname].dims if d not in ("lat", "lon")]
    if weights is not None:
        return (ds[varname] * weights).sum(dims) / weights.sum(dims)
    return ds[varname].mean(dims, skipna=True)


def main():
    y0, y1 = year_range

    path = build_path(month_init, country, y0, y1)
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")

    ds = load_one(path)

    # the data's own year axis, checked against the range in the filename
    if "year" in ds.coords:
        ys = ds["year"].values
        inside = (ys >= y0) & (ys <= y1)
        print(f"year axis: {ys.min()}-{ys.max()}, n={ys.size}")
        if not inside.all():
            print(f"  outside [{y0}, {y1}]: {ys[~inside]}")
    else:
        print("no 'year' coordinate present")

    # mean over year and number only, keeping the lead dimension:
    # out = ds[VAR].mean(["year", "number"], skipna=True)
    # one map per year:
    # out = ds[VAR].mean(["lead", "number"], skipna=True)

    out = monthly_mean(ds)
    out.name = VAR
    out.attrs.update({
        "long_name": f"monthly mean of {VAR}, init month {month_init}",
        "units": ds[VAR].attrs.get("units", "K"),
        "variable": variable,
        "month_init": month_init,
        "country": country,
        "year_start": y0,
        "year_end": y1,
    })

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join(
        "outputs",
        f"monthly_mean_{variable}_mon{month_init:02d}_{country}_{y0}-{y1}.nc",
    )
    out.to_dataset().to_netcdf(out_path)
    print(out)


if __name__ == "__main__":
    main()

