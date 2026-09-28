import os
import numpy as np
import xarray as xr
import xesmf as xe
from scipy.ndimage import gaussian_filter
import functions_spei as fSPEI

# -----------------------------
# Parameters and paths
# -----------------------------
model = "ecmwf51"
init  = 10
list_var = ["tp"]                          # ["t2m-max", "t2m-min", "tp"]

dir_scratch = fSPEI.get_scratch_path()
dir_era5  = dir_scratch + "ERA5-Land/t2m/daily/"
dir_model = dir_scratch + f"C3S_seasonal/{model}/"

country = "Madagascar"

# one weights file per method
method_patch = "patch"
method_cons  = "conservative"
weights_patch = os.path.join(dir_model, f"weights_{model}_to_era5land_{method_patch}_{country.lower()}.nc")
weights_cons  = os.path.join(dir_model, f"weights_{model}_to_era5land_{method_cons}_{country.lower()}.nc")

years = range(1993, 2025 + 1)
UNITS_MAP   = {"tp": "m", "t2m-max": "K", "t2m-min": "K"}


# -----------------------------
# Helpers
# -----------------------------
def deaccumulate(da, time_dim):
    """
    Convert accumulated-since-forecast-start precipitation to per-day increments.
    """
    incr = da.diff(time_dim, label="upper")
    first = da.isel({time_dim: 0})
    out = xr.concat([first.expand_dims(time_dim), incr], dim=time_dim)
    return out.clip(min=0.0)


def find_one_era5_file(year):
    year_dir = os.path.join(dir_era5, str(year))
    files = sorted(f for f in os.listdir(year_dir) if f.endswith(".nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5-Land files found in {year_dir}")
    return os.path.join(year_dir, files[0])


def build_target_grid(name_country):
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
    if varname == "tp":
        return varname
    if varname == "t2m-max":
        return "mx2t24"
    if varname == "t2m-min":
        return "mn2t24"


def get_sample_src(model, year):
    """One model file as the source template, time dim removed."""
    for month_init in range(1, 12):
        sample_src = open_model_file(model, "tp", year, month_init)
        if sample_src is not None:
            if "forecast_period" in sample_src.dims:
                sample_src = sample_src.isel(forecast_period=24, drop=True)
            return sample_src
    raise FileNotFoundError("No sample model daily file found for building regridder.")


def make_regridder(src, dst, method, weights_path=None):
    """Build or reuse a regridder, caching weights when a path is given."""
    if weights_path and os.path.exists(weights_path):
        return xe.Regridder(src, dst, method=method, periodic=False, weights=weights_path)

    regridder = xe.Regridder(src, dst, method=method, periodic=False, reuse_weights=False)
    if weights_path:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        regridder.to_netcdf(weights_path)
    return regridder


def mask_area_weights(da):    
    """    Area weights so sum(dim=['lat','lon']) is a proper grid-cell integral.
    Falls back to cos(lat) if the grid has no cell-area variable.    
    """    
    if "area" in da.coords or "area" in da:        
        return da["area"] if "area" in da else da.coords["area"]    
    lat = da["lat"]    
    return np.cos(np.deg2rad(lat)) * xr.ones_like(da.isel(lat=0, drop=True))


def regrid_precip(da_year, regridder_patch, regridder_cons, land_mask):    
    """    
    Patch regrid for tp, then rescale by one domain-total scalar per timestep 
    so the fine-grid total matches the (partially-)conservative estimate.
    The correction is a function of time only, so it carries no spatial
    structure and cannot reintroduce the source grid's blockiness.    
    """    
    # 1. smooth, continuous fine field    
    da_rg = regridder_patch(da_year, skipna=True, na_thres=0.5)    
    da_rg = da_rg.where(land_mask)
    
    # 2. conservative estimate of the true target-domain total    
    da_cons = regridder_cons(da_year, skipna=True, na_thres=0.5)    
    da_cons = da_cons.where(land_mask)
    
    w = mask_area_weights(da_rg)
    
    # integrate over lat/lon -> a field in time only    
    total_patch = (da_rg * w).sum(dim=["lat", "lon"], skipna=True)    
    total_cons  = (da_cons * w).sum(dim=["lat", "lon"], skipna=True)
    
    # 3. single scalar correction per timestep; leave dry timesteps alone
    scale = xr.where(np.abs(total_patch) > 1e-6, total_cons / total_patch, 1.0)  
    scale = scale.clip(0.0, 10.0)        # guard against pathological blow-ups    
    scale = scale.where(np.abs(total_cons) > 1e-6, 1.0)
    
    return (da_rg * scale).where(da_rg.notnull())

# -----------------------------
# Main
# -----------------------------
target = build_target_grid(country)
land_mask = target.notnull()                # True where ERA5-Land has data
sample_src = get_sample_src(model, 1993)

regridder_patch = make_regridder(sample_src, target, method_patch, weights_patch)
regridder_cons = make_regridder(sample_src, target, method_cons, weights_cons)

for varname in list_var:
    dir_in  = dir_model + f"24h/init_{init:02d}/{varname}/"
    dir_out = dir_model + f"24h/init_{init:02d}/{varname}/regridded_ERA5-Land/"
    os.makedirs(dir_out, exist_ok=True)
    var = varname_to_var(varname)

    is_precip = (varname == "tp")
    method_used = method_patch

    for year in years:
        da_year = open_model_file(model, varname, year, init)
        if da_year is None:
            print(f"Missing file: {varname}_24h_{model}_init{year}{init:02d}_subsaharan-africa.nc")
            continue

        # Deaccumulate precipitation values
        if is_precip:
            da_year = deaccumulate(da_year, "forecast_period")

        # Regrid daily values — smooth-conservative for tp, patch otherwise
        if is_precip:
            da_rg = regrid_precip(da_year, regridder_patch, regridder_cons, land_mask)
        else:
            da_rg = regridder_patch(da_year, skipna=True, na_thres=0.5)
            da_rg = da_rg.where(land_mask)
        
        da_rg = da_rg.rename(var)
        da_rg.attrs["units"] = UNITS_MAP[varname]
        da_rg.attrs["regrid_method"] = method_used

        ds_out = xr.Dataset({var: da_rg})
        out_file = os.path.join(
            dir_out,
            f"{varname}_24h_{model}_init{year}{init:02d}_{country}_res_ERA5-Land_{method_used}.nc"
        )
        ds_out.to_netcdf(out_file)
        print(f"Saved {out_file}")

