import os
import numpy as np
import xarray as xr
import functions_spei_debug as fSPEI


def main():
    dir_pet    = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"
    dir_precip = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/ET0/monthly/"
    dir_out    = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/SPEI/monthly/"
    dir_mask   = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/"

    year_range = [1993, 2024]
    scale = 1
    month = 1

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

    # --- balance in one exemplary grid point ---
    balance_gp = balance.sel(lat=-18.1, lon=49.1, method="nearest")
    balance_dates = balance_gp.time          # matching monthly dates
    balance_accum = fSPEI.rolling_water_balance(balance_gp, scale=scale)
    print("Balance grid-point selected")

    # --- compute SPEI for grid popint ---
    diag = fSPEI.monthwise_spei_diagnostic(
        values=balance_accum,
        dates=balance_dates,
        month=month,
        cal_start=cal_start,
        cal_end=cal_end,
    )
    print(diag)
    fit_info = fSPEI.inspect_loglogistic_fit(diag["sample"]["cal_values"])
    fit = {
        "beta": abs(fit_info["beta"]),
        "loc": fit_info["loc"],
        "scale": fit_info["scale"],
    }
    print(fit) 
    fSPEI.plot_fit_diagnostic(diag["sample"]["cal_values"], fit, output_path=f"mon{month}_fit_diagnostic.png")
#     print()
#     print(diag["fit"])
#     print("cal mean:", diag["cal_mean"])
#     print("cal std:", diag["cal_std"])
    print("Done")


if __name__ == "__main__":
    main()
