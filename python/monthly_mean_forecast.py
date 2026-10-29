#!/usr/bin/env python3
"""
Monthly mean of QDM-adjusted daily forecasts, merged over init+1 .. init+3.

Edit the parameters below, then run:  python monthly_mean.py

Each target month (init+1, init+2, init+3) is a separate file, and the three
files have different lengths along the day axis (28/29/30/31). Every file is
reduced over its OWN day axis first, so the ragged length never has to be
concatenated away. 'lead' is the offset in months from the init month and is
the dimension of the output series.
"""

import calendar
import os

import numpy as np
import xarray as xr

# ---------------------------------------------------------------- parameters
month_init = 10                     # target init month, 1..12
country    = "Madagascar"           # region tag used in the filename
year_range = [1993, 2025]           # [y0, yN] inclusive
variable   = "mx2t24"                   # variable name ("mn2t24", "mx2t24", "tp"); also names the data folder
# ---------------------------------------------------------------------------

VAR = variable + "_qdm"             # array name inside the file
INIT_TAG = f"init{month_init}"      # init tag as it appears in the filenames

QDM_REF = {
    "mn2t24": "ERA5-Land",
    "mx2t24": "ERA5-Land",
    "tp":     "MSWEP",
}


def target_months(month_init, n=3):
    """Init month plus 1..n, wrapped into 1..12."""
    return [(month_init + k - 1) % 12 + 1 for k in range(1, n + 1)]


def days_in_month(month, year):
    """Days in a calendar month; leap years handled via calendar."""
    return calendar.monthrange(year, month)[1]


def build_root(month_init):
    ref = QDM_REF[variable]
    return os.path.join(
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{month_init}",
        variable,
        f"calibrated_{ref}"
    )


def build_path(month, country, y0, y1):
    ref = QDM_REF[variable]
    name = (f"{variable}_24h_mon{month:02d}_{INIT_TAG}_{y0}-{y1}_"
            f"{country}_res_ERA5-Land_qdm_{ref}.nc")
    return os.path.join(build_root(month_init), name)


def reduce_one(path, month,  month_offset):
    """
    Open one target-month file, reduce over its own day axis, and tag the
    result with its month offset.

    Returns a dataset whose dims are (year, number, lat, lon) plus a scalar
    'lead' coordinate equal to month_offset.
    """
    ds = xr.open_dataset(path)
    if "forecast_period" in ds.dims:
        ds = ds.rename({"forecast_period": "days"})

    out = ds[VAR].mean("days", skipna=True)

    out = out.expand_dims(lead=[month_offset])
    out.attrs["target_month"] = month
    return out


def main():
    y0, y1 = year_range
    months = target_months(month_init, 3)
    print(f"init month {month_init} -> target months {months}")
    
    parts = []
    for offset, month in enumerate(months, start=1):
        path = build_path(month, country, y0, y1)
        if not os.path.exists(path):
            raise SystemExit(f"not found: {path}")
        part = reduce_one(path, month, offset)
        ys = part["init_year"].values
        inside = (ys >= y0) & (ys <= y1)
        print(f"  +{offset} mon{month:02d}: dims {dict(part.sizes)}, lead={offset}")
        if not inside.all():
            print(f"     outside [{y0}, {y1}]: {ys[~inside]}")
        parts.append(part)

    merged = xr.concat(parts, dim="lead", combine_attrs="override")
    merged = merged.sortby("lead")

    out = merged.rename(VAR)
    out.name = VAR
    out.attrs.update({
        "long_name": (f"monthly mean of {VAR}, init month {month_init}, "
                      f"target months {months}"),
        "units": merged.attrs.get("units", "K"),
        "variable": variable,
        "month_init": month_init,
        "target_months": ",".join(str(m) for m in months),
        "lead_units": "months since init",
        "year_start": y0,
        "year_end": y1,
    })
    
    out_path = os.path.join(
        build_root(month_init),
        f"{variable}_monthly_lead1-3_init{month_init:02d}_{y0}-{y1}"
        f"_{country}_res_ERA5-Land_qdm_{QDM_REF[variable]}.nc",
    )
    out.to_dataset().to_netcdf(out_path)
    print("Monthly file saved as:\n", out_path)


if __name__ == "__main__":
    main()

