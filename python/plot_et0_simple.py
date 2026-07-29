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
year = 2023
month = 11
country = "Madagascar"

# Directories
path_in = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"
path_out = "/home/PERSONALE/alice.portal2/scratch/figures/ET0/"
file_in = f"{path_in}ET0_Mod-Hargreaves_monthly_1993-2024_{country}.nc"
file_out = f"{path_out}ET0-MH_1m_{year}{month:02d}_{country}.png"

# Select
ds = xr.open_dataset(file_in)
da = ds["ET0"]
da = da.where(da.time.dt.month == month, drop=True)
da = da.where(da.time.dt.year == year, drop=True)
time_stamp = da.time[0].values
da = da.isel(time=0)

# labels and titles
period = f"{MONTHS[month-1]} {year}"
method_label = "Modified Hargreaves"
title = f"ET0 - {method_label}\n{period}"

# arrays
lon = np.asarray(da["longitude"].values)
lat = np.asarray(da["latitude"].values)
vals = np.squeeze(np.asarray(da.values))

# plot
fig, ax = plt.subplots(nrows=1, ncols=1,
                       figsize=(4.6,7),
                       subplot_kw={"projection": ccrs.PlateCarree()})
im = ax.pcolormesh(lon, lat, vals,
                   transform=ccrs.PlateCarree(),
                   cmap="YlGnBu", shading="auto")
ax.coastlines(resolution="50m", linewidth=0.6)
plt.colorbar(im, ax=ax, label="ET0 (mm/day)", shrink=0.8)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5", alpha=0.5)
gl.top_labels = gl.right_labels = False
fig.suptitle(title, y=0.98)
fig.tight_layout()
fig.savefig(file_out, dpi=300)


