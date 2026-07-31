#!/usr/bin/env python3
"""
Plot ET0 and SPEI from monthly NetCDF files.

- ET0 is plotted for the selected month/year.
- SPEI is assumed to be already accumulated on WINDOW months.
- The script writes:
    1) ET0 figure
    2) SPEI figure
    3) combined ET0 + SPEI figure
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
VAR_ET0 = "ET0"
VAR_SPEI = "SPEI"      # change if your variable is named differently

METHOD = "MH"             # "H" or "MH"
MONTH = 12
WINDOW = 3
YEAR = 2023
COUNTRY = "Madagascar"

PATH_ET0 = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/ET0/monthly/")
PATH_SPEI = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/SPEI/monthly/")
PATH_OUT = Path("/home/alice/Desktop/UniBo/figures/SPEI/")


# =============================================================================
# Constants
# =============================================================================
MONTHS = list(calendar.month_abbr)[1:]

METHOD_INFO = {
    "H": {
        "label": "Hargreaves",
        "file_ET0": "ET0_Hargreaves_monthly_1993-2024_{country}.nc",
        "file_SPEI": "SPEI_Hargreaves_{window}m_1993-2024_{country}.nc",
    },
    "MH": {
        "label": "Modified Hargreaves",
        "file_ET0": "ET0_Mod-Hargreaves_monthly_1993-2024_{country}.nc",
        "file_SPEI": "SPEI_Mod-Hargreaves_{window}m_1993-2024_{country}.nc",
    },
}


# =============================================================================
# File builders
# =============================================================================
def build_input_file(var, country, method, window=None):
    if method not in METHOD_INFO:
        raise ValueError(f"Unknown method: {method}. Use one of {list(METHOD_INFO)}")
    if var==VAR_ET0:
        return PATH_ET0 / METHOD_INFO[method][f"file_{var}"].format(country=country, window=window)
    elif var==VAR_SPEI:
        return PATH_SPEI / METHOD_INFO[method][f"file_{var}"].format(country=country, window=window)


def build_output_file(var, country, method, window, year, month):
    return PATH_OUT / f"{var}_{method}_{window}m_{year}{month:02d}_{country}.png"


def build_output_file_combo(country, method, window, year, month):
    return PATH_OUT / f"ET0-SPEI_{method}_{window}m_{year}{month:02d}_{country}.png"


# =============================================================================
# Loaders
# =============================================================================
def load_var(var, country, method, window):
    file_path = build_input_file(var, country, method, window)
    ds = xr.open_dataset(file_path)
    return ds[var]

# =============================================================================
# Time selectors
# =============================================================================
def select_month_year(da, month, year):
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    selected = da.where(da.time.dt.month == month, drop=True)
    selected = selected.where(selected.time.dt.year == year, drop=True)

    if selected.sizes["time"] == 0:
        raise ValueError(f"No data found for {MONTHS[month - 1]} {year}.")

    return selected.isel(time=0), selected.time[0].values


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


# =============================================================================
# Titles
# =============================================================================
def build_title_et0(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = f"{MONTHS[month - 1]} {year}"
    window_label = "single month" if window == 1 else f"{window}-month mean"
    return f"ET0 - {method_label}\n{window_label}, {period}"


def build_title_spei(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = f"{MONTHS[month - 1]} {year}"
    return f"SPEI-{window} - {method_label}\n{period}"


def build_title_combo(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = f"{MONTHS[month - 1]} {year}"
    return f"ET0 and SPEI-{window} - {method_label}\n{period}"


# =============================================================================
# Color settings
# =============================================================================
def get_color_settings_et0():
    vmin, vmax = 0, 7
    levels = MaxNLocator(nbins=vmax * 2).tick_values(vmin, vmax)

    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_over("purple")

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


def get_color_settings_spei():
    levels = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5])

    cmap = plt.colormaps["RdBu"].copy()
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


# =============================================================================
# Plot helpers
# =============================================================================
def _get_lon_lat_vals(da):
    lon_name = "lon" if "lon" in da.coords else "longitude"
    lat_name = "lat" if "lat" in da.coords else "latitude"

    lon = np.asarray(da[lon_name].values)
    lat = np.asarray(da[lat_name].values)
    vals = np.squeeze(np.asarray(da.values))

    return lon, lat, vals


def _format_map(ax, title):
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


# =============================================================================
# Plot single-variable figures
# =============================================================================
def plot_et0(da, method, month, window, year, outpath):
    title = build_title_et0(method, month, window, year)

    lon, lat, vals = _get_lon_lat_vals(da)
    cmap, norm = get_color_settings_et0()

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

    _format_map(ax, title)

    fig.colorbar(
        im,
        ax=ax,
        label="ET0 (mm/day)",
        shrink=0.8,
        extend="max",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_spei(da, method, month, window, year, outpath):
    title = build_title_spei(method, month, window, year)

    lon, lat, vals = _get_lon_lat_vals(da)
    cmap, norm = get_color_settings_spei()

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

    _format_map(ax, title)

    fig.colorbar(
        im,
        ax=ax,
        label=f"SPEI-{window} (-)",
        shrink=0.8,
        extend="both",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# =============================================================================
# Plot combined figure
# =============================================================================
def plot_et0_spei(et0, spei, method, month, window, year, outpath):
    title = build_title_combo(method, month, window, year)

    lon_et0, lat_et0, vals_et0 = _get_lon_lat_vals(et0)
    lon_spei, lat_spei, vals_spei = _get_lon_lat_vals(spei)

    cmap_et0, norm_et0 = get_color_settings_et0()
    cmap_spei, norm_spei = get_color_settings_spei()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9.5, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im0 = axes[0].pcolormesh(
        lon_et0,
        lat_et0,
        vals_et0,
        transform=ccrs.PlateCarree(),
        cmap=cmap_et0,
        norm=norm_et0,
    )
    _format_map(axes[0], "ET0")

    im1 = axes[1].pcolormesh(
        lon_spei,
        lat_spei,
        vals_spei,
        transform=ccrs.PlateCarree(),
        cmap=cmap_spei,
        norm=norm_spei,
    )
    _format_map(axes[1], f"SPEI-{window}")

    fig.suptitle(title)

    fig.colorbar(
        im0,
        ax=axes[0],
        label="ET0 (mm/day)",
        shrink=0.8,
        extend="max",
    )
    fig.colorbar(
        im1,
        ax=axes[1],
        label=f"SPEI-{window} (-)",
        shrink=0.8,
        extend="both",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# =============================================================================
# Summary
# =============================================================================
def print_summary(file_out, label, time_stamp, result):
    lon_name = "lon" if "lon" in result.coords else "longitude"
    lat_name = "lat" if "lat" in result.coords else "latitude"

    print(f"Wrote {file_out}")
    print(f"  field  : {label}")
    print(f"  ending : {np.datetime_as_string(time_stamp, unit='D')}")
    print(
        f"  shape  : "
        f"{lat_name}={result.sizes[lat_name]}, {lon_name}={result.sizes[lon_name]}"
    )
    print(
        f"  range  : "
        f"{float(np.nanmin(result.values)):.2f} .. "
        f"{float(np.nanmax(result.values)):.2f}"
    )


# =============================================================================
# Main
# =============================================================================
def main():
    PATH_OUT.mkdir(parents=True, exist_ok=True)

    file_out_et0 = build_output_file(VAR_ET0, COUNTRY, METHOD, WINDOW, YEAR, MONTH)
    file_out_spei = build_output_file(VAR_SPEI, COUNTRY, METHOD, WINDOW, YEAR, MONTH)
    file_out_combo = build_output_file_combo(COUNTRY, METHOD, WINDOW, YEAR, MONTH)

    da_et0 = load_var(VAR_ET0, COUNTRY, METHOD, WINDOW)
    da_spei = load_var(VAR_SPEI, COUNTRY, METHOD, WINDOW)

    result_et0, time_et0 = window_mean_ending_month(da_et0, MONTH, WINDOW, YEAR)
    result_spei, time_spei = select_month_year(da_spei, MONTH, YEAR)

    plot_et0(result_et0, METHOD, MONTH, WINDOW, YEAR, file_out_et0)
    plot_spei(result_spei, METHOD, MONTH, WINDOW, YEAR, file_out_spei)
    plot_et0_spei(result_et0, result_spei, METHOD, MONTH, WINDOW, YEAR, file_out_combo)

    print_summary(file_out_et0, "ET0", time_et0, result_et0)
    print_summary(file_out_spei, f"SPEI-{WINDOW}", time_spei, result_spei)
    print(f"Wrote {file_out_combo}")


if __name__ == "__main__":
    main()
