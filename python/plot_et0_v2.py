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


MONTHS = list(calendar.month_abbr)[1:]  # Jan..Dec


def load_var(path, var, country, method=None):
    if var == "ET0":
        if method == "H":
            file = f"{path}ET0_Hargreaves_monthly_1993-2024_{country}.nc"
        elif method == "MH":
            file = f"{path}ET0_Mod-Hargreaves_monthly_1993-2024_{country}.nc"
    ds = xr.open_dataset(file)
    da = ds[var]

    # Normalise coordinate names to (time, lat, lon).
    rename = {}
    for cand_t in ("time", "t"):
        if cand_t in da.dims:
            rename[cand_t] = "time"; break
    for cand_y in ("lat", "latitude", "y", "nav_lat"):
        if cand_y in da.dims:
            rename[cand_y] = "lat"; break
    for cand_x in ("lon", "longitude", "x", "nav_lon"):
        if cand_x in da.dims:
            rename[cand_x] = "lon"; break
    if rename:
        da = da.rename(rename)

    # Make sure dims are ordered (time, lat, lon) so .values is (n_lat, n_lon).
    da = da.transpose("time", "lat", "lon")

    # Normalise longitudes to [-180, 180]. ERA5-Land ships lon as 0..360;
    # PlateCarree expects -180..180, otherwise pcolormesh is drawn off-screen
    # and set_extent([0, 360]) silently produces an empty frame.
    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=(((da["lon"] + 180) % 360) - 180))
        da = da.sortby("lon")

    return da


def window_mean_ending_month(da, month, window, year):
    if window == 1:
        sel = da.where(da.time.dt.month == month, drop=True)
    else:
        roll = da.rolling(time=window, center=False,
                          min_periods=window).mean()
        sel = roll.where(roll.time.dt.month == month, drop=True)

    sel = sel.where(sel.time.dt.year == year, drop=True)
    if sel.sizes["time"] == 0:
        raise ValueError(
            f"No data with a full {window}-month window ending in "
            f"{MONTHS[month-1]} {year}."
        )
    return sel.isel(time=0), sel.time[0].values


def plot_spatial(da, method, month, window, year, outpath):
    period = f"{MONTHS[month-1]} {year}"
    win_label = "single month" if window == 1 else f"{window}-month mean"
    method_label = {
        "H":  "Hargreaves",
        "MH": "Modified Hargreaves",
    }
    title = f"ET0 - {method_label[method]}\n{win_label}, {period}"
    print(title)

    lon = np.asarray(da["lon"].values)
    lat = np.asarray(da["lat"].values)
    vals = np.squeeze(np.asarray(da.values))
    
    fig, ax = plt.subplots(figsize=(4.4,8),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    fig.suptitle(title, y=0.9)
    im = ax.pcolormesh(lon, lat, vals,
                       transform=ccrs.PlateCarree(),
                       cmap="YlGnBu", shading="auto")
    ax.coastlines(resolution="50m", linewidth=0.6)
    cb = fig.colorbar(im, ax=ax, label="ET0 (mm/day)", shrink=0.85)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    fig.subplots_adjust(top=0.93, left=0.12, right=0.82, bottom=0.15)
    fig.savefig(outpath)
    plt.close(fig)


def main():
    var = 'ET0'
    method = "MH"          # "H" or "MH"
    month = 12
    window = 1
    year = 2023
    country = "Madagascar"

    path_in = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"
    path_out = "/home/PERSONALE/alice.portal2/scratch/figures/ET0/"
    file_out = f"{path_out}ET0-{method}_{window}m_{year}{month:02d}_{country}.png"

    da = load_var(path_in, var, country, method)
    result, time_stamp = window_mean_ending_month(da, month, window, year)
    plot_spatial(result, method, month, window, year, file_out)

    print(f"Wrote {file_out}")
    print(f"  method : {method}")
    print(f"  window : {window} month(s) ending in {MONTHS[month-1]}")
    print(f"  ending : {np.datetime_as_string(time_stamp, unit='D')}")
    print(f"  shape  : lat={result.sizes['lat']}, lon={result.sizes['lon']}")
    print(f"  range  : "
          f"{float(np.nanmin(result.values)):.2f} .. "
          f"{float(np.nanmax(result.values)):.2f} mm/day")


if __name__ == "__main__":
    main()

