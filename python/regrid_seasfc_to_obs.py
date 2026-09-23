import os
import glob
import xarray as xr
import xesmf as xe
import functions_spei as fSPEI
import pandas as pd
import calendar

# -----------------------------
# Parameters and paths
# -----------------------------
model = "ecmwf51"
init  = 10
list_var = ["t2m-max", "t2m-min", "tp"]

dir_scratch = fSPEI.get_scratch_path()
dir_mswep = dir_scratch + "MSWEP/MSWEP_V316_test/Past/Daily/"
dir_era5  = dir_scratch + "ERA5-Land/t2m/daily/"
dir_model = dir_scratch + f"C3S_seasonal/{model}/" # "24h/init_{init}/"

country = "Madagascar"
weights_file = os.path.join(dir_model, f"weights_{model}_to_era5land_{country.lower()}.nc")

years = range(1993, 1993 + 1)



# -----------------------------
# Helpers
# -----------------------------
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

    da = fSPEI.standardize_latlon(da)
    box = fSPEI.boxes_african_countries(name_country)
    da = fSPEI.subset_box(da, box)

    return da


def open_model_file(model, varname, year, month_init):
    """
    Open one model daily precipitation file.
    """
    fdir = dir_model + f"24h/init_{month_init:02d}/{varname}/" 
    fname = f"{varname}_24h_{model}_init{year}{month_init:02d}_subsaharan-africa.nc"
    fpath = os.path.join(fdir, fname)

    if not os.path.exists(fpath):
        return None

    ds = xr.open_dataset(fpath)

    var = varname_to_var(varname)
    if var not in ds:
        raise KeyError(f"{var} not found in {fpath}")
    da = ds[var]
    da = fSPEI.standardize_latlon(da)

    return da


def varname_to_var(varname):
    """
    Return variable (used in file) from name variable (used for file name).
    """
    if varname == "tp":
        return varname
    if varname == "t2m-max":
        return "mx2t24"
    if varname == "t2m-min":
        return "mn2t24"


def build_regridder(model, year, name_country, weights_path=None):
    """
    Reuse existing Madagascar conservative weights.
    """
    target = build_target_grid(name_country)

    # find one model file as source template
    sample_src = None
    for month_init in range(1, 12):
        sample_src = open_model_file(model, "tp", year, month_init)
        # remove time dimension
        if sample_src is not None: 
            if "forecast_period" in sample_src.dims:
                sample_src = sample_src.isel(forecast_period=24, drop=True)
            break

    if sample_src is None:
        raise FileNotFoundError("No sample model daily file found for building regridder.")

    if weights_path and os.path.exists(weights_path):
        # reuse saved weights
        return xe.Regridder(
            sample_src,
            target,
            method="conservative",
            periodic=False,
            weights=weights_path
        )
    # first run: compute weights, save them
    regridder = xe.Regridder(
            sample_src, 
            target, 
            method="conservative",
            periodic=False, 
            reuse_weights=False
            )
    if weights_path:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        regridder.to_netcdf(weights_path)

    return regridder


# -----------------------------
# Main
# -----------------------------
regridder = build_regridder(model, 1993, country, weights_file)

for varname in list_var:
    dir_in    = dir_model + f"24h/init_{init:02d}/{varname}/"  
    dir_out    = dir_model + f"24h/init_{init:02d}/{varname}/regridded_ERA5-Land/"  
    os.makedirs(dir_out, exist_ok=True)
    var = varname_to_var(varname)
    
    for year in years:
        da_year = None
        da_year = open_model_file(model, varname, year, init)
    
        if da_year is None:
            print(f"Missing file: {varname}_24h_{model}_init{year}{init:02d}_subsaharan-africa.nc")
            continue 
            
        da_rg = regridder(da_year)
        da_rg = da_rg.rename(var)
    
        # Save one file per year
        ds_out = xr.Dataset({
                var: da_rg
            })
        out_file = os.path.join(
                dir_out,
                f"{varname}_24h_{model}_init{year}{init:02d}_subsaharan-africa_res_ERA5-Land.nc"
            )
        ds_out.to_netcdf(out_file)
    
        print(f"Saved {out_file}")
