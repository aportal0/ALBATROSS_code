#!/usr/bin/env python3
"""
Plot ET0 from monthly NetCDF files (Hargreaves or Modified Hargreaves).
"""

import calendar
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


# =============================================================================
# Configuration
# =============================================================================
VAR = "ET0"
METHOD = "MH"          # "H" or "MH"
MONTH = 12
WINDOW = 1
YEAR = 2023
COUNTRY = "Madagascar"

PATH_IN = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/ET0/monthly/")
PATH_OUT = Path("/home/alice/Desktop/UniBo/figures/ET0/")


# =============================================================================
# Constants
# =============================================================================
MONTHS = list(calendar.month_abbr)[1:]

METHOD_INFO = {
    "H": {
        "label": "Hargreaves",
        "filename": "ET0_Hargreaves_monthly_1993-2024_{country}.nc",
    },
    "MH": {
        "label": "Modified Hargreaves",
        "filename": "ET0_Mod-Hargreaves_monthly_1993-2024_{country}.nc",
    },
}


def build_input_file(country, method):
    if method not in METHOD_INFO:
        raise ValueError(f"Unknown method: {method}. Use one of {list(METHOD_INFO)}")
    return PATH_IN / METHOD_INFO[method]["filename"].format(country=country)


def build_output_file(country, method, window, year, month):
    return PATH_OUT / f"ET0-{method}_{window}m_{year}{month:02d}_{country}.png"


def load_var(var, country, method):
    file_path = build_input_file(country, method)
    ds = xr.open_dataset(file_path)
    return ds[var]


def window_mean_ending_month(da, month, window, year):
    if window < 1:
        raise ValueError("window must be >= 1")
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    if window == 1:
        selected = da.where(da.time.dt.month == month, drop=True)
    else:
        rolling_mean = da.rolling(time=window, center=False, min_periods=window).mean()
        selected = rolling_mean.where(rolling_mean.time.dt.month == month, drop=True)

    selected = selected.where(selected.time.dt.year == year, drop=True)

    if selected.sizes["time"] == 0:
        raise ValueError(
            f"No data with a full {window}-month window ending in "
            f"{MONTHS[month - 1]} {year}."
        )

    return selected.isel(time=0), selected.time[0].values


def build_title(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = f"{MONTHS[month - 1]} {year}"
    window_label = "single month" if window == 1 else f"{window}-month mean"
    return f"ET0 - {method_label}\n{window_label}, {period}"


def get_color_settings():
    vmin, vmax = 0, 7
    levels = MaxNLocator(nbins=vmax * 2).tick_values(vmin, vmax)

    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_over("purple")

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


def plot_spatial(da, method, month, window, year, outpath):
    title = build_title(method, month, window, year)

    lon = np.asarray(da["longitude"].values)
    lat = np.asarray(da["latitude"].values)
    vals = np.squeeze(np.asarray(da.values))

    cmap, norm = get_color_settings()

    fig, ax = plt.subplots(
        figsize=(4.4, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im = ax.pcolormesh(
        lon,
        lat,
        vals,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )

    ax.set_title(title)
    ax.coastlines(resolution="50m", linewidth=0.6)

    fig.colorbar(
        im,
        ax=ax,
        label="ET0 (mm/day)",
        shrink=0.8,
        extend="max",
    )

    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="0.5",
        alpha=0.5,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def print_summary(file_out, method, month, window, time_stamp, result):
    print(f"Wrote {file_out}")
    print(f"  method : {METHOD_INFO[method]['label']}")
    print(f"  window : {window} month(s) ending in {MONTHS[month - 1]}")
    print(f"  ending : {np.datetime_as_string(time_stamp, unit='D')}")
    print(
        f"  shape  : "
        f"lat={result.sizes['latitude']}, lon={result.sizes['longitude']}"
    )
    print(
        f"  range  : "
        f"{float(np.nanmin(result.values)):.2f} .. "
        f"{float(np.nanmax(result.values)):.2f} mm/day"
    )


def main():
    PATH_OUT.mkdir(parents=True, exist_ok=True)

    file_out = build_output_file(COUNTRY, METHOD, WINDOW, YEAR, MONTH)

    da = load_var(VAR, COUNTRY, METHOD)
    result, time_stamp = window_mean_ending_month(da, MONTH, WINDOW, YEAR)
    plot_spatial(result, METHOD, MONTH, WINDOW, YEAR, file_out)

    print_summary(file_out, METHOD, MONTH, WINDOW, time_stamp, result)


if __name__ == "__main__":
    main()

