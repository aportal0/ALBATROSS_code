"""Merged ET0 + SPEI driver.

Monthly precipitation (MSWEP, regridded to ERA5-Land) + monthly Tmax/Tmin
(ERA5-Land) -> Hargreaves / modified-Hargreaves ET0 -> climatic water
balance -> SPEI at multiple accumulation scales.

All of the work lives in `functions_spei_merged`; this file is only the
configuration and the run order. Edit the constants below, then run:

    python et0_spei_merged.py
"""

from __future__ import annotations

import os
import sys

import functions_spei as fSPEI


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


# COUNTRY must be a key of fSPEI.boxes_african_countries().
# YEARS is an inclusive (start, end) pair.
COUNTRY = "Madagascar"
YEARS = (1993, 2024)
SCALES = [1, 3, 6, 12]

# "Hargreaves", "Mod-Hargreaves", or "both" (runs the SPEI chain twice).
METHOD = "both"

SAVE_YEARLY = False      # also write one ET0 / precip file per year
FORCE_REGRID = False     # ignore cached regridding weights


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main() -> int:
    paths = fSPEI.resolve_paths(COUNTRY)

    methods = (["Hargreaves", "Mod-Hargreaves"] if METHOD == "both"
               else [METHOD])
    y0, y1 = YEARS

    for key in ("t2m", "mswep", "et0", "precip", "spei"):
        os.makedirs(paths[key], exist_ok=True)

    et0, precip = fSPEI.build_et0_and_precip(
        paths, COUNTRY, YEARS, methods, force_regrid=FORCE_REGRID)

    for method in methods:
        if method not in et0:
            continue

        et0[method].to_netcdf(os.path.join(
            paths["et0"], f"ET0_{method}_monthly_{y0}-{y1}_{COUNTRY}.nc"))
        print(f"[ET0] wrote {method} multi-year file", flush=True)

        lsm = fSPEI.land_fraction(paths, COUNTRY, et0[method])
        balance = fSPEI.water_balance(precip["precipitation"],
                                      et0[method], lsm)
        fSPEI.write_spei(balance, paths["spei"], method, YEARS, SCALES, COUNTRY)

    precip["precipitation"].to_netcdf(os.path.join(
        paths["precip"], f"precip_monthly_{y0}-{y1}_{COUNTRY}.nc"))

    if SAVE_YEARLY:
        for year in range(y0, y1 + 1):
            sel = {"time": str(year)}
            for method in methods:
                if method not in et0:
                    continue
                et0[method].sel(**sel).to_netcdf(os.path.join(
                    paths["et0"],
                    f"ET0_{method}_monthly_{year}_{COUNTRY}.nc"))
            precip["precipitation"].sel(**sel).to_netcdf(os.path.join(
                paths["precip"], f"precip_monthly_{year}_{COUNTRY}.nc"))

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

