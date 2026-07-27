import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xesmf as xe
import os

# Own functions
import functions_SPEI as fSPEI

# Directories
dir_ERA5Land = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/t2m/monthly/"
dir_MSWEP = "/home/PERSONALE/alice.portal2/scratch/MSWEP/MSWEP_V316_test/Past/Monthly/"
dir_out = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"

year = 1993

# Input files
tmin_file = dir_ERA5Land + f"t2m_minimum_monthly_{year}.nc"
tmax_file = dir_ERA5Land + f"t2m_maximum_monthly_{year}.nc"
tmean_file = dir_ERA5Land + f"t2m_mean_monthly_{year}.nc"
precip_file = dir_MSWEP + f"{year}.nc"

# Load datasets
ds_tmin = xr.open_dataset(tmin_file)
ds_tmax = xr.open_dataset(tmax_file)
ds_tmean = xr.open_dataset(tmean_file)
ds_precip = xr.open_dataset(precip_file)

# Extract variables (change names according to your files)
tmin = ds_tmin["t2m"]
tmax = ds_tmax["t2m"]
tmean = ds_tmean["t2m"]
precip = ds_precip["precipitation"]

# Convert temperature from Kelvin to Celsius if needed
if tmin.mean() > 100:
    tmin = tmin - 273.15
    tmax = tmax - 273.15
    tmean = tmean - 273.15

# Boxes for single countries
ghana_box = fSPEI.boxes_african_countries('ghana')
madagascar_box = fSPEI.boxes_african_countries('madagascar') 

# Select boxes 
tmean_mg = fSPEI.subset_box(tmean, madagascar_box).rename({'valid_time': 'time'})
tmin_mg  = fSPEI.subset_box(tmin, madagascar_box).rename({'valid_time': 'time'})
tmax_mg  = fSPEI.subset_box(tmax, madagascar_box).rename({'valid_time': 'time'})

# Regrid precipitation to temperature grid
target_grid = tmean_mg
regridder = xe.Regridder(
    precip,
    target_grid,
    method="conservative",
    periodic=False,
    reuse_weights=False
)
precip_mg = regridder(precip).resample(time='1MS').sum()

# Regrid temperature to time
tmean_mg = tmean_mg.resample(time='1MS').mean()
tmin_mg = tmin_mg.resample(time='1MS').mean()
tmax_mg = tmax_mg.resample(time='1MS').mean()

# Aggregate datasets
ds_monthly = xr.Dataset({
    'tmin': tmin_mg,
    'tmax': tmax_mg,
    'precip': precip_mg
})

# Compute ET0 (Modified Hargreaves)
ET0_MH = fSPEI.hargreaves_modified_vectorized(ds_monthly)
ET0_H = fSPEI.hargreaves_vectorized(ds_monthly)

# Save ET0
os.makedirs(dir_out, exist_ok=True)
ET0_MH.to_netcdf(os.path.join(dir_out, f'ET0_Mod-Hargreaves_monthly_{year}_Madagascar.nc'))
ET0_H.to_netcdf(os.path.join(dir_out, f'ET0_Hargreaves_monthly_{year}_Madagascar.nc'))



