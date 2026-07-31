import os
import xarray as xr
import xesmf as xe
import functions_spei as fSPEI

dir_ERA5Land = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/t2m/monthly/"
dir_MSWEP    = "/home/PERSONALE/alice.portal2/scratch/MSWEP/MSWEP_V316_test/Past/Monthly/"
dir_out      = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"

weights_file = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/weights_madagascar.nc"
box = fSPEI.boxes_african_countries('madagascar')
years = range(1993, 2024 + 1)

# --- Build regridder once on year 1, then reuse via saved weights ---
def build_regridder(year, weights_path=None):
    ds_tmax = xr.open_dataset(f"{dir_ERA5Land}t2m_maximum_monthly_{year}.nc",
                             chunks={'valid_time': 12})
    target = fSPEI.subset_box(ds_tmax['t2m'].rename({'valid_time': 'time'}), box)
    ds_precip = xr.open_dataset(f"{dir_MSWEP}{year}.nc", chunks={'time': 12})
    src = ds_precip['precipitation']
    if weights_path and os.path.exists(weights_path):
        # reuse saved weights
        return xe.Regridder(src, target, method="conservative",
                           periodic=False, weights=weights_path)
    # first run: compute weights, save them
    regridder = xe.Regridder(src, target, method="conservative",
                             periodic=False, reuse_weights=False)
    if weights_path:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        regridder.to_netcdf(weights_path)
    return regridder

regridder = build_regridder(years[0], weights_file)

for year in years:
    # --- load ---
    ds_tmin   = xr.open_dataset(f"{dir_ERA5Land}t2m_minimum_monthly_{year}.nc", chunks={'valid_time': 12})
    ds_tmax   = xr.open_dataset(f"{dir_ERA5Land}t2m_maximum_monthly_{year}.nc", chunks={'valid_time': 12})
    ds_precip = xr.open_dataset(f"{dir_MSWEP}{year}.nc", chunks={'time': 12})

    tmin   = ds_tmin['t2m']
    tmax   = ds_tmax['t2m']
    precip = ds_precip['precipitation']

    # --- Kelvin -> Celsius (probe one slice, don't compute the mean) ---
    u = (tmin.attrs.get('units') or '').lower()
    if ('k' in u and 'c' not in u) or float(tmin.isel(valid_time=0).values.flat[0]) > 100:
        tmin = tmin - 273.15
        tmax = tmax - 273.15

    # --- subset + rename ---
    tmin_mg = fSPEI.subset_box(tmin, box).rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'})
    tmax_mg = fSPEI.subset_box(tmax, box).rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'})

    # --- regrid precip (reusing saved weights) ---
    precip_mg = (
            regridder(precip)
            .resample(time='1MS')
            .sum()
            .rename('precipitation')
            .rename({'latitude': 'lat', 'longitude': 'lon'})
    )

    # --- align time index ---
    tmin_mg = tmin_mg.resample(time='1MS').mean()
    tmax_mg = tmax_mg.resample(time='1MS').mean()

    # --- assemble dataset ---
    ds_monthly = xr.Dataset({
        'tmin':   tmin_mg,
        'tmax':   tmax_mg,
        'precip': precip_mg,
    })

    # --- ET0: compute Ra once, share across both formulas ---
    Ra = fSPEI._ra_dataarray(ds_monthly)
    ET0_MH = fSPEI.hargreaves_modified(ds_monthly, Ra=Ra)
    ET0_H  = fSPEI.hargreaves(ds_monthly, Ra=Ra)

    # --- save ---
    os.makedirs(dir_out, exist_ok=True)
    ET0_MH.to_netcdf(os.path.join(dir_out, f'ET0_Mod-Hargreaves_monthly_{year}_Madagascar.nc'))
    ET0_H.to_netcdf(os.path.join(dir_out,  f'ET0_Hargreaves_monthly_{year}_Madagascar.nc'))
    precip_mg.to_netcdf(os.path.join(dir_out,  f'precip_monthly_{year}_Madagascar.nc')) 

    print(f"Done {year}")

