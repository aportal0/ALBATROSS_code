import xarray as xr
import os

in_dir = '/home/PERSONALE/alice.portal2/scratch/C3S_seasonal/ecmwf51/6h/init_10/t2m/'
out_dir = '/home/PERSONALE/alice.portal2/scratch/C3S_seasonal/ecmwf51/24h/init_10/t2m/' 
os.makedirs(out_dir, exist_ok=True)

for year in range(1994,2026):
    ds = xr.open_dataset(f"{in_dir}t2m_6h_ecmwf51_init{year}10_subsaharan-africa.nc")
    
    out = ds.coarsen(forecast_period=4, boundary="trim").mean()
    
    # Keep the last valid time of each 4-step window
    out["valid_time"] = ds.valid_time.isel(forecast_period=slice(3, None, 4))
    out["forecast_period"] = ds.forecast_period.isel(forecast_period=slice(3, None, 4))
    
    out.to_netcdf(f"{out_dir}t2m_24h_ecmwf51_init{year}10_subsaharan-africa.nc")
