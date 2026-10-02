#!/usr/bin/env python3
"""
QDM bias adjustment of ECMWF SEAS5 daily precipitation (24h tp) vs MSWEP,
FOR A SINGLE GRID POINT -- reduced version for debugging.

Same method as the full-grid script, but everything is done on 1-D time series
extracted at the nearest lon/lat, so there is no stack/unstack, no apply_ufunc,
no core dims and no alignment: a straight line of code you can step through and
print at will.

Method (unchanged):
  * training 1993-2022, verification 2023-2025;
  * one transfer function per (lead = target month), trained on that month only;
  * wet days only, threshold 1 mm/day identical for model and obs;
  * dry days stay dry, observed dry-day frequency never imposed;
  * multiplicative QDM: delta from the training climatologies, re-applied to
    the target value, so the anomaly survives.

Run:
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00 \
         python qdm_point_debug.py
"""

from pathlib import Path
import os
import numpy as np
import xarray as xr

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
FC_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/regridded_ERA5-Land")
OB_DIR = Path("/ec/res4/scratch/ecme4047/MSWEP/MSWEP_V316_test/Past/Daily/regridded_ERA5-Land")
OUT_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/calibrated_MSWEP")

REGION      = "Madagascar"
TRAIN_YEARS = range(1993, 2022 + 1)
VERIF_YEARS = range(2023, 2025 + 1)
INIT_MONTH  = 10

# ---- THE POINT TO DEBUG: edit these two ----
LAT_POINT = -18.9          # e.g. Antananarivo area
LON_POINT = 47.5

FC_VAR = "tp"
OB_VAR = "precipitation"

FC_UNITS_TO_MM = 1000.0    # tp in metres (24 h accumulation) -> mm/day
OB_UNITS_TO_MM = 1.0       # MSWEP already mm/day

THRESHOLD_MM = 1.0         # wet-day threshold, identical for model and obs
N_MEMBERS_KEEP = 25        # members common to all years (verify!)

MEMBER_DIM      = "number"
FC_TIME_DIM     = "forecast_period"
FC_SELECT_COORD = "valid_time"
FC_CONCAT_DIM   = "forecast_reference_time"
OB_TIME_DIM     = "time"

N_QUANTILES  = 100
QUANTILE_MIN = 0.01

OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_MONTHS = [((INIT_MONTH - 1 + k) % 12) + 1 for k in range(4)]
TARGET = [(m, (INIT_MONTH - 1 + k) // 12) for k, m in enumerate(LEAD_MONTHS)]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _init_year(fname):
    stem = fname.name if isinstance(fname, Path) else str(fname)
    return int(stem.split("init")[1][:4])


def fc_path(y):
    return FC_DIR / f"tp_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}_res_ERA5-Land_patch.nc"


def ob_path(y, yoff, month):
    return OB_DIR / f"precip_daily_{y + yoff}{month:02d}_{REGION}_res_ERA5-Land.nc"


def extract_point(ds, var, lat, lon, spatial_lat="lat", spatial_lon="lon"):
    """Nearest-grid-point 1-D extraction. Returns a numpy array over the file's
    own time axis, with every other dimension reduced/selected explicitly."""
    da = ds[var].sel({spatial_lat: lat, spatial_lon: lon}, method="nearest")
    # drop any leftover singleton dims (e.g. forecast_reference_time)
    da = da.squeeze(drop=True)
    return da


def qdm_point(mod_train, obs_train, mod_target,
              nq=N_QUANTILES, qmin=QUANTILE_MIN):
    """
    Multiplicative QDM on three 1-D arrays of WET days (mm/day).

    dry days are excluded by the caller (they are simply absent from these
    arrays), so no NaN handling is needed here.
    """
    mod_train = np.asarray(mod_train, dtype=np.float64)
    obs_train = np.asarray(obs_train, dtype=np.float64)
    mod_target = np.asarray(mod_target, dtype=np.float64)

    mod_train = mod_train[np.isfinite(mod_train)]
    obs_train = obs_train[np.isfinite(obs_train)]

    print(f"    qdm_point: mod_train wet={mod_train.size} "
          f"obs_train wet={obs_train.size} target wet={mod_target.size}",
          flush=True)

    if mod_train.size < 2 or obs_train.size < 2 or mod_target.size == 0:
        print("    -> insufficient wet days: returning target unchanged", flush=True)
        return mod_target

    q = np.linspace(qmin, 1.0 - qmin, nq)

    xm = np.maximum.accumulate(np.quantile(mod_train, q))
    xo = np.maximum.accumulate(np.quantile(obs_train, q))

    p = np.interp(mod_target, xm, q, left=qmin, right=1.0 - qmin)
    obs_at_p = np.interp(p, q, xo)
    mod_at_p = np.interp(p, q, xm)

    ratio = obs_at_p / mod_at_p
    out = mod_target * ratio

    print(f"    -> mod range {mod_train.min():.2f}..{mod_train.max():.2f}  "
          f"obs range {obs_train.min():.2f}..{obs_train.max():.2f}  "
          f"ratio {ratio.min():.3f}..{ratio.max():.3f}  "
          f"target range {mod_target.min():.2f}..{mod_target.max():.2f}  "
          f"out range {out.min():.2f}..{out.max():.2f}", flush=True)
    return out


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    print(f"single-point QDM debug  lat={LAT_POINT}  lon={LON_POINT}", flush=True)

    for k, (month, yoff) in enumerate(TARGET):
        print(f"\n=== lead {k} -- target month {month:02d} "
              f"(year offset {yoff}) ===", flush=True)

        # ---- 1. training pool: all years x all members, wet days only ----
        m_train = []
        for y in TRAIN_YEARS:
            f = fc_path(y)
            if not f.exists():
                continue
            ds = xr.open_dataset(f)
            da = extract_point(ds, FC_VAR, LAT_POINT, LON_POINT)
            # keep the days of the target month, using the valid_time coordinate
            da = da.sel({FC_TIME_DIM: da[FC_SELECT_COORD].dt.month == month},
                        drop=True)
            if da.sizes.get(FC_TIME_DIM, 0) == 0:
                ds.close()
                continue
            da = da.isel({MEMBER_DIM: slice(0, N_MEMBERS_KEEP)})
            vals = (da.values * FC_UNITS_TO_MM).ravel()      # (time x member)
            vals = np.clip(vals, 0.0, None)                  # kill patch negatives
            m_train.append(vals)
            ds.close()
        m_train = np.concatenate(m_train) if m_train else np.array([])

        # ---- 2. obs training pool ---------------------------------------
        o_train = []
        for y in TRAIN_YEARS:
            f = ob_path(y, yoff, month)
            if not f.exists():
                continue
            ds = xr.open_dataset(f)
            da = extract_point(ds, OB_VAR, LAT_POINT, LON_POINT)
            vals = (da.values * OB_UNITS_TO_MM).ravel()
            vals = np.clip(vals, 0.0, None)
            o_train.append(vals)
            ds.close()
        o_train = np.concatenate(o_train) if o_train else np.array([])

        # ---- 3. target: verification years, per year/member --------------
        tgt = []          # (year, member, day) kept flat with a record of shape
        for y in VERIF_YEARS:
            f = fc_path(y)
            if not f.exists():
                continue
            ds = xr.open_dataset(f)
            da = extract_point(ds, FC_VAR, LAT_POINT, LON_POINT)
            da = da.sel({FC_TIME_DIM: da[FC_SELECT_COORD].dt.month == month},
                        drop=True)
            if da.sizes.get(FC_TIME_DIM, 0) == 0:
                ds.close()
                continue
            da = da.isel({MEMBER_DIM: slice(0, N_MEMBERS_KEEP)})
            vals = np.clip(da.values * FC_UNITS_TO_MM, 0.0, None)
            tgt.append(vals)
            ds.close()

        if not m_train.size or not o_train.size or not tgt:
            print("  missing data -> skip", flush=True)
            continue

        # ---- 4. wet-day selection: keep ONLY wet days -------------------
        m_wet = m_train[m_train > THRESHOLD_MM]
        o_wet = o_train[o_train > THRESHOLD_MM]

        print(f"  train wet: model={m_wet.size}/{m_train.size} "
              f"({100*m_wet.size/max(m_train.size,1):.1f}%)  "
              f"obs={o_wet.size}/{o_train.size} "
              f"({100*o_wet.size/max(o_train.size,1):.1f}%)", flush=True)

        # ---- 5. apply QDM to wet days only, keep dry days at 0 ----------
        corrected_years = []
        for arr in tgt:
            flat = arr.ravel()
            wet_mask = flat > THRESHOLD_MM
            out = np.zeros_like(flat)
            if wet_mask.any():
                out[wet_mask] = qdm_point(m_wet, o_wet, flat[wet_mask])
            corrected_years.append(out.reshape(arr.shape))

        corrected = np.concatenate(corrected_years)

        raw = np.concatenate([a.ravel() for a in tgt])

        # ---- 6. summary -------------------------------------------------
        wet_frac_out = 100.0 * (corrected > 0).mean()

        print(f"  target raw : mean={raw.mean():.2f} max={raw.max():.2f} "
              f"wet_days={100*(raw > THRESHOLD_MM).mean():.1f}%", flush=True)
        print(f"  corrected  : mean={corrected.mean():.2f} "
              f"max={corrected.max():.2f} wet_days={wet_frac_out:.1f}%",
              flush=True)
        print(f"  obs (train): mean={o_train.mean():.2f} "
              f"wet_days={100*(o_train > THRESHOLD_MM).mean():.1f}%", flush=True)

        # ---- 7. write a tiny per-point output ---------------------------
        out_ds = xr.Dataset(
            {
                "tp_raw":       ("sample", raw.astype("float32")),
                "tp_qdm":       ("sample", corrected.astype("float32")),
            },
            coords={"sample": np.arange(raw.size)},
            attrs={
                "lat": str(LAT_POINT), "lon": str(LON_POINT),
                "month": str(month), "lead": str(k),
                "threshold_mm": str(THRESHOLD_MM),
                "train": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
                "members_kept": str(N_MEMBERS_KEEP),
            },
        )
        p = OUT_DIR / f"qdm_point_lead{k}_mon{month:02d}_lat{LAT_POINT}_lon{LON_POINT}.nc"
        out_ds.to_netcdf(p)
        print(f"  wrote {p}", flush=True)

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
