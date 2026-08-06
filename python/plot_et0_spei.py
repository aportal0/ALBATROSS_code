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
VAR_PR = "precipitation"
VAR_WB = "balance"

METHOD = "H"             # "H" or "MH"
MONTH = 11
WINDOW = 3
YEAR = "clim"               # year (int) or "clim" (str)
COUNTRY = "Madagascar"

PATH_ET0 = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/ET0/monthly/")
PATH_SPEI = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/SPEI/monthly/")
PATH_PR = PATH_ET0
PATH_WB = PATH_SPEI
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
        "file_precipitation": "precip_monthly_1993-2024_{country}.nc",
        "file_balance": "water-balance_Hargreaves_monthly_1993-2024_{country}.nc",
    },
    "MH": {
        "label": "Modified Hargreaves",
        "file_ET0": "ET0_Mod-Hargreaves_monthly_1993-2024_{country}.nc",
        "file_SPEI": "SPEI_Mod-Hargreaves_{window}m_1993-2024_{country}.nc",
        "file_precipitation": "precip_monthly_1993-2024_{country}.nc",
        "file_balance": "water-balance_Mod-Hargreaves_monthly_1993-2024_{country}.nc",
    },
}


# =============================================================================
# File builders
# =============================================================================
def build_input_file(var, country, method, window=None):
    if method not in METHOD_INFO:
        raise ValueError(f"Unknown method: {method}. Use one of {list(METHOD_INFO)}")
    if var==VAR_ET0:
        return PATH_ET0 / METHOD_INFO[method][f"file_{var}"].format(
                country=country, window=window
        )
    elif var==VAR_SPEI:
        return PATH_SPEI / METHOD_INFO[method][f"file_{var}"].format(
                country=country, window=window
        )
    elif var == VAR_PR:
        return PATH_PR / METHOD_INFO[method][f"file_{var}"].format(
            country=country, window=window
        )
    elif var == VAR_WB:
        return PATH_WB / METHOD_INFO[method][f"file_{var}"].format(
            country=country, window=window
        )
    else:
        raise ValueError(f"Unknown variable: {var}")


def build_output_file(var, country, method, window, year, month):
    year_label = str(year)
    if var in [VAR_ET0, VAR_SPEI, VAR_WB]:
        return PATH_OUT / f"{var}_{method}_{window}m_{year_label}{month:02d}_{country}.png"
    elif var == VAR_PR:
        return PATH_OUT / f"{var}_{window}m_{year_label}{month:02d}_{country}.png"


def build_output_file_combo(country, method, window, year, month):
    year_label = str(year)
    return PATH_OUT / f"ET0-SPEI-PR-WB_{method}_{window}m_{year_label}{month:02d}_{country}.png"


# =============================================================================
# Loaders
# =============================================================================
def load_var(var, country, method, window):
    file_path = build_input_file(var, country, method, window)
    ds = xr.open_dataset(file_path)
    da = ds[var]
    if var==VAR_ET0:
        da = convert_day_to_month(da)
    return da


def convert_day_to_month(da, dim_time="time"):
    days = xr.DataArray(
        da[dim_time].dt.days_in_month,
        coords={dim_time: da[dim_time]},
        dims=[dim_time]
    )
    return da * days


# =============================================================================
# Time selectors
# =============================================================================
def select_month_year(da, month, year):
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    selected = da.where(da.time.dt.month == month, drop=True)
    if is_climatology_year(year):
        if selected.sizes["time"] == 0:
            raise ValueError(f"No data found for climatology of {MONTHS[month - 1]}.")
        result = selected.mean(dim="time", skipna=True)
        time_stamp = selected.time.values[0]
        return result, time_stamp
    selected = selected.where(selected.time.dt.year == year, drop=True)
    if selected.sizes["time"] == 0:
        raise ValueError(f"No data found for {MONTHS[month - 1]} {year}.")
    return selected.isel(time=0), selected.time[0].values
    

def is_climatology_year(year):
    return isinstance(year, str) and year.lower() == "clim"


def window_mean_or_sum_ending_month(da, month, window, year, l_stats):
    if window < 1:
        raise ValueError("window must be >= 1")
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    if window == 1:
        selected = da.where(da.time.dt.month == month, drop=True)
    else:
        if l_stats.lower() == "mean":
            rolling_stats = da.rolling(time=window, center=False, min_periods=window).mean()
        elif l_stats.lower() == "sum":
            rolling_stats = da.rolling(time=window, center=False, min_periods=window).sum()
        else:
            raise ValueError("stats must be 'mean' or 'sum'")
        selected = rolling_stats.where(rolling_stats.time.dt.month == month, drop=True)
    if is_climatology_year(year):
        if selected.sizes["time"] == 0:
            raise ValueError(
                f"No data found for climatology with a full {window}-month window ending in "
                f"{MONTHS[month - 1]}."
            )
        result = selected.mean(dim="time", skipna=True)
        time_stamp = selected.time.values[0]
        return result, time_stamp
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
def build_period_label(month, year):
    if is_climatology_year(year):
        return f"{MONTHS[month - 1]} climatology"
    return f"{MONTHS[month - 1]} {year}"


def build_title_et0(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = build_period_label(month, year)
    window_label = "single month" if window == 1 else f"{window}-month mean"
    return f"ET0 - {method_label}\n{window_label}, {period}"


def build_title_spei(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = build_period_label(month, year)
    return f"SPEI-{window} - {method_label}\n{period}"


def build_title_pr(month, window, year):
    period = build_period_label(month, year)
    window_label = "single month" if window == 1 else f"{window}-month mean"
    return f"Precipitation\n {window_label}, {period}"


def build_title_wb(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = build_period_label(month, year)
    window_label = "single month" if window == 1 else f"{window}-month mean"
    return f"Water balance - {method_label}\n{window_label}, {period}"


def build_title_combo(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = f"{MONTHS[month - 1]} {year}"
    window_label = "single month" if window == 1 else f"{window} months"
    return f"ET0, SPEI-{window}, P, WB - {method_label}\n{window_label}, {period}"


# =============================================================================
# Color settings
# =============================================================================
def get_color_settings_et0():
    vmin, vmax = 0, 200
    levels = MaxNLocator(nbins=10).tick_values(vmin, vmax)

    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_over("purple")

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


def get_color_settings_spei():
    levels = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5])

    cmap = plt.colormaps["RdBu"].copy()
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


def get_color_settings_pr():
    vmin, vmax = 0, 400
    levels = MaxNLocator(nbins=8).tick_values(vmin, vmax)
    cmap = plt.colormaps["Blues"].copy()
    cmap.set_over("k")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


def get_color_settings_wb():
    levels = np.array([-300, -250, -200, -150, -100, -50, -25, 25, 50, 100, 150, 200, 250, 300])
    cmap = plt.colormaps["BrBG"].copy()
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
        label=f"ET0 ({window}-month mean, mm)",
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


def plot_pr(da, month, window, year, outpath):
    title = build_title_pr(month, window, year)

    lon, lat, vals = _get_lon_lat_vals(da)
    cmap, norm = get_color_settings_pr()

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
        label=f"P ({window}-month mean, mm)",
        shrink=0.8,
        extend="max",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_wb(da, method, month, window, year, outpath):
    title = build_title_wb(method, month, window, year)

    lon, lat, vals = _get_lon_lat_vals(da)
    cmap, norm = get_color_settings_wb()

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
        label=f"P - ET0 ({window}-month mean, mm)",
        shrink=0.8,
        extend="both",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# =============================================================================
# Plot combined figure
# =============================================================================
def plot_et0_spei_pr_wb(et0, spei, pr, wb, method, month, window, year, outpath):
    title = build_title_combo(method, month, window, year)

    lon_et0, lat_et0, vals_et0 = _get_lon_lat_vals(et0)
    lon_spei, lat_spei, vals_spei = _get_lon_lat_vals(spei)
    lon_pr, lat_pr, vals_pr = _get_lon_lat_vals(pr)
    lon_wb, lat_wb, vals_wb = _get_lon_lat_vals(wb)

    cmap_et0, norm_et0 = get_color_settings_et0()
    cmap_spei, norm_spei = get_color_settings_spei()
    cmap_pr, norm_pr = get_color_settings_pr()
    cmap_wb, norm_wb = get_color_settings_wb()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im0 = axes[0, 0].pcolormesh(
        lon_spei, lat_spei, vals_spei,
        transform=ccrs.PlateCarree(),
        cmap=cmap_spei, norm=norm_spei
    )
    _format_map(axes[0, 0], f"SPEI-{window}")
    
    im1 = axes[0, 1].pcolormesh(
        lon_et0, lat_et0, vals_et0,
        transform=ccrs.PlateCarree(),
        cmap=cmap_et0, norm=norm_et0
    )
    _format_map(axes[0, 1], f"ET0 ({window}-month mean)")

    im2 = axes[1, 0].pcolormesh(
        lon_pr, lat_pr, vals_pr,
        transform=ccrs.PlateCarree(),
        cmap=cmap_pr, norm=norm_pr
    )
    _format_map(axes[1, 0], f"P ({window}-month mean)")

    im3 = axes[1, 1].pcolormesh(
        lon_wb, lat_wb, vals_wb,
        transform=ccrs.PlateCarree(),
        cmap=cmap_wb, norm=norm_wb
    )
    _format_map(axes[1, 1], f"WB = P - ET0")

    fig.suptitle(title)

    fig.colorbar(im0, ax=axes[0, 0], label=f"SPEI-{window} (-)", shrink=0.8, extend="both")
    fig.colorbar(im1, ax=axes[0, 1], label="ET0 (mm/month)", shrink=0.8, extend="max")
    fig.colorbar(im2, ax=axes[1, 0], label="P (mm/month)", shrink=0.8, extend="max")
    fig.colorbar(im3, ax=axes[1, 1], label="WB (mm/month)", shrink=0.8, extend="both")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# =============================================================================
# Summary
# =============================================================================
def print_summary(file_out, label, time_stamp, result, month=None, year=None):
    lon_name = "lon" if "lon" in result.coords else "longitude"
    lat_name = "lat" if "lat" in result.coords else "latitude"
    print(f"Wrote {file_out}")
    print(f"  field  : {label}")
    if month is not None and year is not None and is_climatology_year(year):
        print(f"  period : {MONTHS[month - 1]} climatology")
    else:
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
    file_out_pr = build_output_file(VAR_PR, COUNTRY, METHOD, WINDOW, YEAR, MONTH)
    file_out_wb = build_output_file(VAR_WB, COUNTRY, METHOD, WINDOW, YEAR, MONTH)
    file_out_combo = build_output_file_combo(COUNTRY, METHOD, WINDOW, YEAR, MONTH)

    da_et0 = load_var(VAR_ET0, COUNTRY, METHOD, WINDOW)
    da_spei = load_var(VAR_SPEI, COUNTRY, METHOD, WINDOW)
    da_pr = load_var(VAR_PR, COUNTRY, METHOD, WINDOW)
    da_wb = load_var(VAR_WB, COUNTRY, METHOD, WINDOW)

    result_et0, time_et0 = window_mean_or_sum_ending_month(da_et0, MONTH, WINDOW, YEAR, "mean")
    result_spei, time_spei = select_month_year(da_spei, MONTH, YEAR)
    result_pr, time_pr = window_mean_or_sum_ending_month(da_pr, MONTH, WINDOW, YEAR, "mean")
    result_wb, time_wb = window_mean_or_sum_ending_month(da_wb, MONTH, WINDOW, YEAR, "mean")

    plot_et0(result_et0, METHOD, MONTH, WINDOW, YEAR, file_out_et0)
    plot_spei(result_spei, METHOD, MONTH, WINDOW, YEAR, file_out_spei)
    plot_pr(result_pr, MONTH, WINDOW, YEAR, file_out_pr)
    plot_wb(result_wb, METHOD, MONTH, WINDOW, YEAR, file_out_wb)
    plot_et0_spei_pr_wb(result_et0, result_spei, result_pr, result_wb, METHOD, MONTH, WINDOW, YEAR, file_out_combo)

    print_summary(file_out_et0, f"ET0-{WINDOW}", time_et0, result_et0, MONTH, YEAR)
    print_summary(file_out_spei, f"SPEI-{WINDOW}", time_spei, result_spei, MONTH, YEAR)
    print_summary(file_out_pr, f"P-{WINDOW}", time_pr, result_pr, MONTH, YEAR)
    print_summary(file_out_wb, f"WB-{WINDOW}", time_wb, result_wb, MONTH, YEAR)
    print(f"Wrote {file_out_combo}")


if __name__ == "__main__":
    main()
