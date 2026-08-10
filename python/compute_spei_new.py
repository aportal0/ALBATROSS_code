import os
import numpy as np
import xarray as xr
import functions_spei as fSPEI


def main():
    dir_pet    = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"
    dir_precip = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"
    dir_out    = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/SPEI/monthly/"
    dir_mask   = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/"

    year_range = [1993, 2024]
    scales = [1,3,6,12]

    cal_start = f"{year_range[0]}-01-01"
    cal_end   = f"{year_range[1]}-12-31"

    method = "Hargreaves" # "Mod-Hargreaves" or "Hargreaves"
    country = "Madagascar"

    # --- load ---
    ds_pet = xr.open_dataset(
        f"{dir_pet}ET0_{method}_monthly_1993-2024_{country}.nc",
        chunks={'time': 12}
    )
    ds_precip = xr.open_dataset(
        f"{dir_precip}precip_monthly_1993-2024_{country}.nc",
        chunks={'time': 12}
    )
    ds_mask = xr.open_dataset(
        f"{dir_mask}land_mask_ERA5-Land.nc"
    )

    pet = ds_pet['ET0'].sel(time=slice(cal_start, cal_end))
    precip = ds_precip['precipitation'].sel(time=slice(cal_start, cal_end))
    lsm = ds_mask['lsm']
    box = fSPEI.boxes_african_countries('madagascar')
    lsm = fSPEI.subset_box(lsm, box).rename({'latitude': 'lat', 'longitude': 'lon'})

    print("Input loaded")

    # --- convert PET from mm/day to mm/month ---
    days = xr.DataArray(
        pet['time'].dt.days_in_month,
        coords={'time': pet['time']},
        dims=['time']
    )
    pet = pet * days
    print("ET0 converted to monthly sum")

    # --- set metadata ---
    pet.name = 'pet'
    precip.name = 'precip'
    pet.attrs['units'] = 'mm'
    precip.attrs['units'] = 'mm'

    # --- align ---
    precip, pet = xr.align(precip, pet, join='inner')
    print("Input data aligned")

    # --- compute monthly water balance ---
    balance = (precip - pet).rename('balance')
    balance.attrs['units'] = 'mm'
    print("Balance computed")

    # --- mask balance over ocean using land-sea mask ---
    lsm_on_balance = lsm.interp(
        lat=balance.lat,
        lon=balance.lon,
        method="nearest"
    )
    land_mask = lsm_on_balance > 0.5
    balance = balance.where(land_mask)
    print("Balance masked with land-sea mask")

    # --- compute SPEI for selected scales ---
    os.makedirs(dir_out, exist_ok=True)

    for scale in scales:
        spei = fSPEI.compute_spei(
            balance,
            scale=scale,
            cal_start=cal_start,
            cal_end=cal_end
        )
        spei.to_netcdf(
            os.path.join(
                dir_out,
                f'SPEI_{method}_{scale}m_{year_range[0]}-{year_range[1]}_{country}.nc'
            )
        )
        print(f"Saved SPEI-{scale}")

    # --- save water balance too ---
    balance.to_netcdf(
        os.path.join(
            dir_out,
            f'water-balance_{method}_monthly_{year_range[0]}-{year_range[1]}_{country}.nc'
        )
    )

    print("Done")


if __name__ == "__main__":
    main()
