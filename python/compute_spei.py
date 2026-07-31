import os
import numpy as np
import xarray as xr
import functions_spei as fSPEI


def main():
    dir_pet    = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"
    dir_precip = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"
    dir_out    = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/SPEI/monthly/"

    year_range = [1993, 2024]
    scales = [1]

    cal_start = f"{year_range[0]}-01-01"
    cal_end   = f"{year_range[1]}-12-31"

    # --- load ---
    ds_pet    = xr.open_dataset(
        f"{dir_pet}ET0_Mod-Hargreaves_monthly_1993-2024_Madagascar.nc",
        chunks={'time': 12}
    )
    ds_precip = xr.open_dataset(
        f"{dir_precip}precip_monthly_1993-2024_Madagascar.nc",
        chunks={'time': 12}
    )

    pet    = ds_pet['ET0'].sel(time=slice(cal_start, cal_end))
    precip = ds_precip['precipitation'].sel(time=slice(cal_start, cal_end))
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

    # --- mask balance in ocean cells ---
    mask_ocean = np.isfinite(balance.isel(time=0))
    balance = balance.where(mask_ocean)
    print("Balance masked over ocean")

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
                f'SPEI_{scale}m_{year_range[0]}-{year_range[1]}_Madagascar.nc'
            )
        )
        print(f"Saved SPEI-{scale}")

    # --- save water balance too ---
    balance.to_netcdf(
        os.path.join(
            dir_out,
            f'water-balance_monthly_{year_range[0]}-{year_range[1]}_Madagascar.nc'
        )
    )

    print("Done")


if __name__ == "__main__":
    main()

