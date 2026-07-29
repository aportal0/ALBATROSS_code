#!/usr/bin/env python3
"""
Plot ET0 from monthly NetCDF files (Hargreaves or Modified Hargreaves).

The files are expected to contain a 3-D variable (time, lat, lon) with monthly
time stamps. The script:
  1. selects the method  -> reads the corresponding .nc file
  2. optionally averages over N consecutive months ENDING in the chosen month
     (trailing window; year wrap handled automatically by rolling on the real
      time axis)
  3. plots a spatial map (climatology over all years, or a single year)
"""

import calendar
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


MONTHS = list(calendar.month_abbr)[1:]  # Jan..Dec


def load_var(path, var, country, method=None):
    if var=="ET0":
        if method=="H":
       	    file = f"{path}ET0_Hargreaves_monthly_1993-2024_{country}.nc"
        elif method=="MH":
            file = f"{path}ET0_Mod-Hargreaves_monthly_1993-2024_{country}.nc"
    ds = xr.open_dataset(file)
    da = ds[var]
    
    rename = {}
    for cand_t in ("time", "t"):
        if cand_t in da.dims:
            rename[cand_t] = "time"
            break
    for cand_y in ("lat", "latitude", "y", "nav_lat"):
        if cand_y in da.dims:
            rename[cand_y] = "lat"
            break
    for cand_x in ("lon", "longitude", "x", "nav_lon"):
        if cand_x in da.dims:
            rename[cand_x] = "lon"
            break
    if rename:
        da = da.rename(rename)
    return da


def window_mean_ending_month(da, month, window, year):
    """
    Trailing mean over `window` consecutive months ENDING at `month`.

    - rolling on the real time axis: year wrap is automatic.
    - keep only the window ending in (year, month)
    """
    if window == 1:
        # No smoothing: just pick the chosen month across years.
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
    period = (f"{MONTHS[month-1]} {year}")
    win_label = "single month" if window == 1 else f"{window}-month mean"
    method_label = {
        "H": "Hargreaves",
        "MH": "Modified Hargreaves (Droogers & Allen 2002)",
    }
    title = f"ET0 - {method_label[method]}\n{win_label}, {period}"

    lon, lat = da.lon, da.lat
    vals = np.squeeze(da.values)            # shape (n_lat, n_lon) after time selection
    fig = plt.figure(figsize=(8, 4.4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon, lat, vals,
                       transform=ccrs.PlateCarree(),
                       cmap="YlGnBu", shading="auto")
    ax.set_extent([float(lon.min()), float(lon.max()),
                   float(lat.min()), float(lat.max())],
                  crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.6)
    cb = fig.colorbar(im, ax=ax, label="ET0 (mm/day)", shrink=0.85)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    ax.set_title(title)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    
    # Parameters
    var = 'ET0'
    method = "MH" # choose between H (Hargreaves) and MH (modified Hargreaves)
    month = 12
    window = 1
    year = 2023
    country = "Madagascar"
    
    # I/O files
    path_in = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"
    path_out = "/home/PERSONALE/alice.portal2/scratch/figures/ET0/"
    file_out = f"{path_out}ET0-{method}_{window}m_{year}{month:02d}_{country}.png"
    
    # Load variable
    da = load_var(path_in, var, country, method)
    result, time_stamp = window_mean_ending_month(
        da, month, window, year
    )

    plot_spatial(result, method, month, window,
                 year, file_out)
    print(f"Wrote {file_out}")
    print(f"  method : {method}")
    print(f"  window : {window} month(s) ending in {MONTHS[month-1]}")
    print(f"  ending : {np.datetime_as_string(time_stamp, unit='D')}")
    print(f"  shape  : lat={result.sizes['latitude']}, lon={result.sizes['longitude']}")
    print(f"  range  : "
          f"{float(np.nanmin(result.values)):.2f} .. "
          f"{float(np.nanmax(result.values)):.2f} mm/day")


if __name__ == "__main__":
    main()
