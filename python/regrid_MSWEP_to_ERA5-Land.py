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
dir_era5  = dir_scratch + "ERA5-Land/t2m/daily/"
dir_out   = dir_mswep + "regridded_ERA5-Land/"

country = "Madagascar"
weights_file = os.path.join(dir_out, f"weights_mswep_to_era5land_{country}.nc")

years = range(1994, 2024 + 1)

os.makedirs(dir_out, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def standardize_latlon(da):
    rename_dict = {}
    if "latitude" in da.dims or "latitude" in da.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in da.dims or "longitude" in da.coords:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        da = da.rename(rename_dict)
    return da


def find_one_era5_file(year):
    year_dir = os.path.join(dir_era5, str(year))
    files = sorted(
        f for f in os.listdir(year_dir)
        if f.endswith(".nc")
    )
    if not files:
        raise FileNotFoundError(f"No ERA5-Land files found in {year_dir}")
    return os.path.join(year_dir, files[0])


def build_target_grid(name_country):
    """
    Build one Madagascar ERA5-Land target grid from a single ERA5 daily file.
    """
    sample_file = find_one_era5_file(1993)
    ds = xr.open_dataset(sample_file)

    if "t2m" not in ds:
        raise KeyError(f"'t2m' not found in ERA5 file: {sample_file}")

    da = ds["t2m"]

    if "valid_time" in da.dims:
        da = da.isel(valid_time=0, drop=True)
    elif "time" in da.dims:
        da = da.isel(time=0, drop=True)

    da = standardize_latlon(da)
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
    da = standardize_latlon(da)

    # remove singleton time dimension if present
    if "time" in da.dims and da.sizes["time"] == 1:
        da = da.isel(time=0, drop=True)

    # assign time from year + day_of_year
    date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day_no - 1)
    da = da.expand_dims(time=[date])

    return da


def build_regridder(year, weights_path, name_country):
    """
    Reuse existing Madagascar conservative weights.
    """
    target = build_target_grid(name_country)

    # find one MSWEP file as source template
    sample_src = None
    for day_no in range(1, 367):
        sample_src = open_mswep_daily_file(year, day_no)
        if sample_src is not None:
            break

    if sample_src is None:
        raise FileNotFoundError("No sample MSWEP daily file found for building regridder.")

    regridder = xe.Regridder(
        sample_src,
        target,
        method="conservative",
        periodic=False,
        weights=weights_path
    )

    return regridder


# -----------------------------
# Main
# -----------------------------
regridder = build_regridder(1993, weights_file, country)

for year in years:
    max_day = 366 if calendar.isleap(year) else 365
    daily_list = []

    for day_no in range(1, max_day+1):
        da_day = open_mswep_daily_file(year, day_no)

        if da_day is None:
            print(f"Missing file: {year}{day_no:03d}.nc")
            continue

        da_rg = regridder(da_day)
        da_rg = da_rg.rename("precipitation")
        daily_list.append(da_rg)

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
            f"precip_daily_{year}{month:02d}_{country}_res_ERA5-Land.nc"
        )
        ds_out.to_netcdf(out_file)

        print(f"Saved {out_file}")

    print(f"Done {year}")
