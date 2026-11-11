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
import xarray as xr

import functions_spei_driver as fSPEI


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


# COUNTRY must be a key of fSPEI.boxes_african_countries().
# YEARS is an inclusive (start, end) pair, used for the observation branch.
COUNTRY = "Madagascar"
YEARS = (1993, 2026)
SCALES = [12]

# "Hargreaves", "Mod-Hargreaves", or "both" (runs the SPEI chain twice).
METHOD = "Mod-Hargreaves"

FORCE_REGRID = False     # True to ignore cached regridding weights
FORCE_RECOMPUTE = False  # True to ignore saved precip and et0 files

# --- which branches to run -------------------------------------------------
RUN_OBS = True           # observations: ERA5-Land + MSWEP -> SPEI
RUN_FORECAST = False     # calibrated seasonal forecasts -> forecast SPEI

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

def main() -> int:
    paths = fSPEI.resolve_paths(COUNTRY)
    y0, y1 = YEARS

    methods = (["Hargreaves", "Mod-Hargreaves"] if METHOD == "both"
               else [METHOD])

    # ---------------------------------------------------------- observations
    if RUN_OBS:
        et0_names = {
            m: os.path.join(paths["et0"],
                            f"ET0_{m}_monthly_{y0}-{y1}_{COUNTRY}.nc")
            for m in methods
        }
        precip_name = os.path.join(
            paths["precip"], f"precip_monthly_{y0}-{y1}_{COUNTRY}.nc")

        obs_complete = (not FORCE_RECOMPUTE
                        and all(os.path.exists(p) for p in et0_names.values())
                        and os.path.exists(precip_name))

        if obs_complete:
            et0 = {m: xr.open_dataarray(p) for m, p in et0_names.items()}
            for m, da in et0.items():
                da.load()          # close the handle; the file may be rewritten
                print(f"[ET0] {m} loaded from {da.encoding.get('source', 'file')}",
                      flush=True)
            precip = xr.Dataset({"precipitation": xr.open_dataarray(precip_name)})
            precip.load()
            print("[precip] loaded from cache", flush=True)
        else:
            et0, precip = fSPEI.build_et0_and_precip(
                paths, COUNTRY, YEARS, methods, force_regrid=FORCE_REGRID)

            for method in methods:
                et0[method].to_netcdf(et0_names[method])
                print(f"[ET0] wrote {method} multi-year file", flush=True)

            precip["precipitation"].to_netcdf(precip_name)
            print("[precip] wrote multi-year file", flush=True)

        # The SPEI chain runs off whatever is now in memory, cached or fresh.
        for method in methods:
            if method not in et0:
                continue
            lsm = fSPEI.land_fraction(paths, COUNTRY, et0[method])
            balance = fSPEI.water_balance(precip["precipitation"],
                                          et0[method], lsm)
            fSPEI.write_spei(balance, paths["spei"], method, YEARS, SCALES,
                             COUNTRY)

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

