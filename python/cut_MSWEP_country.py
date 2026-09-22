import os
import glob
import xarray as xr
import xesmf as xe
import functions_spei as fSPEI
import pandas as pd
import calendar

# -----------------------------
# Paths
# -----------------------------
dir_scratch = fSPEI.get_scratch_path()

dir_mswep = dir_scratch + "MSWEP/MSWEP_V316_test/Past/Daily/"
dir_out   = dir_mswep + "countries/"

country = "Ghana"
years = range(1993, 2024 + 1)

os.makedirs(dir_out, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def cut_to_country(da, name_country):
    """
    Cut dataarray to preset country boundaries.
    """

    box = fSPEI.boxes_african_countries(name_country)
    da = fSPEI.subset_box(da, box)

    return da


def open_mswep_daily_file(year, day_no):
    """
    Open one MSWEP daily file named YYYYDDD.nc
    """
    fname = f"{year}{day_no:03d}.nc"
    fpath = os.path.join(dir_mswep, fname)

    if not os.path.exists(fpath):
        return None

    ds = xr.open_dataset(fpath)

    if "precipitation" not in ds:
        raise KeyError(f"'precipitation' not found in {fpath}")

    da = ds["precipitation"]
    da = fSPEI.standardize_latlon(da)

    # remove singleton time dimension if present
    if "time" in da.dims and da.sizes["time"] == 1:
        da = da.isel(time=0, drop=True)

    # assign time from year + day_of_year
    date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day_no - 1)
    da = da.expand_dims(time=[date])

    return da


# -----------------------------
# Main
# -----------------------------

for year in years:
    max_day = 366 if calendar.isleap(year) else 365
    daily_list = []

    for day_no in range(1, max_day+1):
        da_day = open_mswep_daily_file(year, day_no)

        if da_day is None:
            print(f"Missing file: {year}{day_no:03d}.nc")
            continue

        da_cut = cut_to_country(da_day, country)
        daily_list.append(da_cut)

    if not daily_list:
        print(f"No valid daily MSWEP files found for {year}")
        continue

    precip_daily = xr.concat(daily_list, dim="time").sortby("time")

    # Group by month and save one file per month
    months = precip_daily.groupby("time.month")
    for month, da_month in months:
        ds_out = xr.Dataset({
            "precipitation": da_month
        })

        out_file = os.path.join(
            dir_out,
            f"precip_daily_{year}{month:02d}_{country}.nc"
        )
        ds_out.to_netcdf(out_file)

        print(f"Saved {out_file}")

    print(f"Done {year}")
