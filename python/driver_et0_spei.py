"""Merged ET0 + SPEI driver.

Two sources of SPEI, both built from the same ET0 machinery:

  * OBSERVATIONS -- ERA5-Land Tmax/Tmin + MSWEP precipitation, regridded to
    ERA5-Land, at several accumulation scales.
  * FORECAST -- the calibrated C3S seasonal forecasts (ecmwf51), as merged by
    monthly_mean.py, for a chosen init month and lead window.

Edit the constants below, then run:

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
# YEARS is an inclusive (start, end) pair, used for the observation branch.
COUNTRY = "Madagascar"
YEARS = (1993, 2024)
SCALES = [1, 3, 6, 12]

# "Hargreaves", "Mod-Hargreaves", or "both" (runs the SPEI chain twice).
METHOD = "both"

FORCE_REGRID = False     # ignore cached regridding weights

# --- which branches to run -------------------------------------------------
RUN_OBS = False           # observations: ERA5-Land + MSWEP -> SPEI
RUN_FORECAST = True     # calibrated seasonal forecasts -> forecast SPEI

# --- forecast branch -------------------------------------------------------
# The init month whose seasonal run is used, 1..12.
FORECAST_INIT_MONTH = 10

# The hindcast span in the monthly forecast FILENAMES. This is the file
# naming span, not the analysis span -- monthly_mean.py writes whatever
# year_range it was given, so this must match it exactly.
FORECAST_YEARS = (1993, 2025)

# Lead windows to compute, as tuples of lead months. (1,) is the lead-1 SPEI;
# (1, 2, 3) is the lead-1-to-3 SPEI. The forecast balance is built once and
# shared by every window.
FORECAST_WINDOWS = [(1,), (1, 2, 3)]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main() -> int:
    paths = fSPEI.resolve_paths(COUNTRY)
    y0, y1 = YEARS

    methods = (["Hargreaves", "Mod-Hargreaves"] if METHOD == "both"
               else [METHOD])

    keys = ("t2m", "mswep", "et0", "precip", "spei")
    if RUN_OBS:
        for key in keys:
            os.makedirs(paths[key], exist_ok=True)

    # ---------------------------------------------------------- observations
    if RUN_OBS:
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
            fSPEI.write_spei(balance, paths["spei"], method, YEARS, SCALES,
                             COUNTRY)

        precip["precipitation"].to_netcdf(os.path.join(
            paths["precip"], f"precip_monthly_{y0}-{y1}_{COUNTRY}.nc"))

    # -------------------------------------------------------------- forecast
    if RUN_FORECAST:
        lsm_fc = None
        if RUN_OBS and methods and methods[0] in et0:
            # Reuse the land mask built for the observations when available.
            lsm_fc = fSPEI.land_fraction(paths, COUNTRY, et0[methods[0]])

        for method in methods:
            fSPEI.run_forecast_spei(
                paths, COUNTRY, FORECAST_INIT_MONTH, method,
                windows=FORECAST_WINDOWS, years=FORECAST_YEARS,
                lsm=lsm_fc)

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

