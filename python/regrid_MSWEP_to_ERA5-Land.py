import os
import glob
import xarray as xr
import xesmf as xe
import functions_spei as fSPEI

# -----------------------------
# Paths
# -----------------------------
dir_scratch = fSPEI.get_scratch_path()

dir_mswep = dir_scratch + "MSWEP/MSWEP_V316_test/Past/Daily/"
dir_era5  = dir_scratch + "ERA5-Land/t2m/daily/"
dir_out   = dir_mswep + "regridded_ERA5-Land/"

weights_file = os.path.join(dir_out, "weights_mswep_to_era5land.nc")

years = range(1993, 2024 + 1)

os.makedirs(dir_out, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def find_era5_daily_file(year):
    """
    Finds one ERA5-Land daily file for the given year to define the target grid.
    Adjust the glob if your ERA5 filenames follow a different pattern.
    """
    year_dir = os.path.join(dir_era5, str(year))
    candidates = sorted(glob.glob(os.path.join(year_dir, "*.nc")))
    if not candidates:
        raise FileNotFoundError(f"No ERA5-Land daily files found in {year_dir}")
    return candidates[0]


def get_target_grid(year):
    """
    Opens one ERA5-Land daily file and extracts the target grid from t2m.
    """
    era5_file = find_era5_daily_file(year)
    ds = xr.open_dataset(era5_file)

    if "t2m" not in ds:
        raise KeyError(f"'t2m' not found in ERA5 file: {era5_file}")

    da = ds["t2m"]

    # Remove time dimension if present
    if "time" in da.dims:
        target = da.isel(time=0, drop=True)
    elif "valid_time" in da.dims:
        target = da.isel(valid_time=0, drop=True)
    else:
        target = da

    # Standardize coordinate names for xESMF
    rename_dict = {}
    if "latitude" in target.dims or "latitude" in target.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in target.dims or "longitude" in target.coords:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        target = target.rename(rename_dict)

    return target


def open_mswep_day(year, day_no):
    """
    Opens one MSWEP daily file named YYYYDDD.nc
    """
    fname = f"{year}{day_no:03d}.nc"
    fpath = os.path.join(dir_mswep, fname)

    if not os.path.exists(fpath):
        return None

    ds = xr.open_dataset(fpath)

    if "precipitation" not in ds:
        raise KeyError(f"'precipitation' not found in MSWEP file: {fpath}")

    da = ds["precipitation"]

    # Standardize coordinate names
    rename_dict = {}
    if "latitude" in da.dims or "latitude" in da.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in da.dims or "longitude" in da.coords:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        da = da.rename(rename_dict)

    # Remove singleton time dim if present
    if "time" in da.dims and da.sizes["time"] == 1:
        da = da.isel(time=0, drop=True)

    # Re-add proper time coordinate from filename
    date = xr.DataArray(
        [xr.cftime_range(start=f"{year}-01-01", periods=365, calendar="standard")[day_no - 1]],
        dims=["time"],
        name="time"
    )

    da = da.expand_dims(time=date)

    return da


def build_regridder(sample_year, weights_path):
    """
    Builds the regridder once, or reuses saved weights.
    """
    target = get_target_grid(sample_year)

    sample_src = None
    for d in day_numbers:
        sample_src = open_mswep_day(sample_year, d)
        if sample_src is not None:
            break

    if sample_src is None:
        raise FileNotFoundError(f"No MSWEP daily files found for sample year {sample_year}")

    if os.path.exists(weights_path):
        regridder = xe.Regridder(
            sample_src,
            target,
            method="conservative",
            periodic=False,
            weights=weights_path
        )
    else:
        regridder = xe.Regridder(
            sample_src,
            target,
            method="conservative",
            periodic=False,
            reuse_weights=False
        )
        regridder.to_netcdf(weights_path)

    return regridder


# -----------------------------
# Main
# -----------------------------
regridder = build_regridder(1993, weights_file)

for year in years:
    max_day = 366 if calendar.isleap(year) else 365
    daily_list = []

    for day_no in range(1, max_day+1):
        da_day = open_mswep_day(year, day_no)

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
            f"MSWEP_precip_daily_{year}{month:02d}_Madagascar_res_ERA5-Land.nc"
        )
        ds_out.to_netcdf(out_file)

        print(f"Saved {out_file}")

    print(f"Done {year}")
