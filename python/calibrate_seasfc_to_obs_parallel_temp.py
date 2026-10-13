#!/usr/bin/env python3
"""
QDM bias adjustment of ECMWF SEAS5 daily MIN/MAX TEMPERATURE vs a reference,
ON THE FULL GRID, POINT BY POINT, with dask parallelism.

Adapted from the precipitation version. THE DRY-DAY LOGIC IS GONE:
temperature has no mass at zero and no trace threshold, so there is no
censoring and no re-zeroing -- the correction runs on all days, on the whole
distribution.

Method:
  * training 1993-2022, verification 2023-2025;
  * one transfer function per (lead = target month), trained on that month only;
  * ADDITIVE QDM (not multiplicative): the delta is a temperature difference
    between the observed and model training climatologies, re-applied to the
    target value. An anomalous warm/cold month therefore survives the correction
    instead of being flattened onto the observed climatology. A multiplicative
    ratio makes no physical sense for temperature, which is centred on zero and
    can be negative;
  * the first SKIP_LEAD_DAYS days after the initialization are excluded from the
    calibration (forecast spin-up). The reference has no lead, so its pool spans
    the full calendar month -- for the first lead the two windows differ and this
    is recorded in the output attributes.

MEMORY MODEL -- why the spatial loop is blocked
    The per-cell work is scheduled with dask.delayed, but the three sample sets
    are materialised as numpy arrays on the client. Loading the whole grid at
    once does not fit in the worker budget, so the grid is processed in
    LAT_BLOCK row strips: each strip is loaded, its cells corrected, the strip
    result kept, and the strip arrays freed. The strips are concatenated along
    the latitude axis at the end.
    If a worker still dies, lower LAT_BLOCK first.

Run:
    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00 \
         python calibrate_grid_temp.py
"""

from pathlib import Path
import os
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask import delayed, compute
import tempfile

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
LOCAL_SCRATCH = os.environ.get("TMPDIR")
if not LOCAL_SCRATCH or not Path(LOCAL_SCRATCH).is_dir():
    LOCAL_SCRATCH = tempfile.gettempdir()
print(f"dask local_directory: {LOCAL_SCRATCH}", flush=True)

# ---- WHICH TEMPERATURE: "tmin" or "tmax" ------------------------------
TEMP_TYPE = "tmin"

REGION      = "Madagascar"
TRAIN_YEARS = range(1993, 2022 + 1)
VERIF_YEARS = range(2023, 2025 + 1)
INIT_MONTH  = 10
N_MONTHS    = 4


# Variable names on disk, per temperature type. EDIT these to match your files:
# the forecast variable in a SEAS5 file is usually "mx2t6"/"mn2t6" (6-hourly
# maxima) or "t2m"/"mx2t"/"mn2t" depending on how the files were produced, and
# the reference variable depends on the observational product you regridded.
FC_VAR_BY_TYPE = {"tmin": "mn2t24", "tmax": "mx2t24"}
OB_VAR_BY_TYPE = {"tmin": "t2m",  "tmax": "t2m"}

FC_VAR = FC_VAR_BY_TYPE[TEMP_TYPE]
OB_VAR = OB_VAR_BY_TYPE[TEMP_TYPE]

# Directories
FC_DIR = Path(f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}/{FC_VAR}/regridded_ERA5-Land")
OB_DIR = Path(f"/ec/res4/scratch/ecme4047/ERA5-Land/{OB_VAR}/daily")
OUT_DIR = Path(f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}/{FC_VAR}/calibrated_ERA5-Land")

# Units: BOTH series must be in the SAME unit. Set the conversion to the unit
# you want to work and plot in. K -> degC subtracts 273.15; if the files are
# already in degC leave both at "celsius".
FC_UNITS_IN  = "K"         # "K" or "celsius"
OB_UNITS_IN  = "K"
OUT_UNITS    = "K"

K_TO_C = 273.15


def to_celsius(arr, units_in):
    """Convert a temperature array to degrees Celsius. K subtracts 273.15."""
    if str(units_in).lower() in ("k", "kelvin"):
        return arr - K_TO_C
    return arr


N_MEMBERS_KEEP = 25        # members common to all years (verify!)
SKIP_LEAD_DAYS = 10        # drop the first N days after init from CALIBRATION

MEMBER_DIM      = "number"
FC_TIME_DIM     = "forecast_period"
FC_SELECT_COORD = "valid_time"
FC_CONCAT_DIM   = "forecast_reference_time"
OB_TIME_DIM     = "valid_time"

SPATIAL_LAT = "lat"
SPATIAL_LON = "lon"

N_QUANTILES  = 100
QUANTILE_MIN = 0.01

# ---- parallelism / memory ----
N_WORKERS  = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
WORKER_MEM = "3GB"
TIME_CHUNK = 40
LAT_BLOCK  = 20            # latitude rows loaded per strip (lower if workers die)

OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_MONTHS = [((INIT_MONTH - 1 + k) % 12) + 1 for k in range(N_MONTHS)]
TARGET = [(m, (INIT_MONTH - 1 + k) // 12) for k, m in enumerate(LEAD_MONTHS)]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _init_year(fname):
    stem = fname.name if isinstance(fname, Path) else str(fname)
    return int(stem.split("init")[1][:4])


def fc_path(y):
    return FC_DIR / (f"{FC_VAR}_24h_ecmwf51_init{y}{INIT_MONTH:02d}_"
                     f"{REGION}_res_ERA5-Land_patch.nc")


def ob_path(y, yoff, month):
    if TEMP_TYPE == "tmin":
        stat = "minimum"
    elif TEMP_TYPE == "tmax":
        stat = "maximum"
    return OB_DIR / str(y+yoff) / (f"{OB_VAR}_daily_{stat}_{y + yoff}{month:02d}.nc")


def lead_keep_mask(da, month, init_year, month_day=1):
    """Steps to keep: valid_time in `month` AND lead >= SKIP_LEAD_DAYS days
    after the initialization date (forecast spin-up excluded)."""
    in_month = da[FC_SELECT_COORD].dt.month == month
    if not SKIP_LEAD_DAYS:
        return in_month
    cutoff = (np.datetime64(f"{int(init_year):04d}-{INIT_MONTH:02d}-{month_day:02d}")
              + np.timedelta64(SKIP_LEAD_DAYS, "D"))
    return in_month & (da[FC_SELECT_COORD] >= cutoff)


def qdm_point_additive(mod_train, obs_train, mod_target,
                       nq=N_QUANTILES, qmin=QUANTILE_MIN):
    """
    ADDITIVE Quantile Delta Mapping on three 1-D arrays (degC, all days).

    The delta is a difference in temperature:

        p         = F_mod_train(x)                where the target sits in the
                                                  model training climatology
        delta(p)  = F_obs_train(p) - F_mod_train(p)
        y         = x + delta(p)                  delta RE-APPLIED to the target

    so the anomaly relative to the model climatology survives. A ratio would be
    meaningless for a variable centred on zero and able to go negative.
    """
    mod_target = np.asarray(mod_target, dtype=np.float64)
    mt = np.asarray(mod_train, dtype=np.float64)
    ot = np.asarray(obs_train, dtype=np.float64)

    mt = mt[np.isfinite(mt)]
    ot = ot[np.isfinite(ot)]

    if mt.size < 2 or ot.size < 2 or mod_target.size == 0:
        return mod_target.astype(np.float32)

    q = np.linspace(qmin, 1.0 - qmin, nq)
    xm = np.quantile(mt, q)
    xo = np.quantile(ot, q)
    # temperature quantiles are not guaranteed monotonic on coarse samples
    xm = np.maximum.accumulate(xm)
    xo = np.maximum.accumulate(xo)

    p = np.interp(mod_target, xm, q, left=qmin, right=1.0 - qmin)
    delta = np.interp(p, q, xo) - np.interp(p, q, xm)
    return (mod_target + delta).astype(np.float32)


@delayed
def correct_cell(m_pool, o_pool, t_in, iy, ix):
    """
    Correct ONE grid cell -- the whole additive QDM for that point.

    m_pool : (ny_m, ntime_m, nmem) model pool, ALL years/members, degC
    o_pool : (ny_o, ntime_o)       reference pool, ALL years, degC
    t_in   : (ny, ntime, nmem)     target for this cell, degC
    iy, ix : grid indices (provenance)

    No censoring, no re-zeroing: the correction applies to every day.
    """
    m_flat = np.asarray(m_pool, dtype=np.float64).ravel()
    o_flat = np.asarray(o_pool, dtype=np.float64).ravel()
    m_flat = m_flat[np.isfinite(m_flat)]
    o_flat = o_flat[np.isfinite(o_flat)]

    t_in = np.asarray(t_in, dtype=np.float64)

    if m_flat.size < 2 or o_flat.size < 2:
        # not enough sample in this cell: return the target unchanged
        return t_in.astype(np.float32)

    q = np.linspace(QUANTILE_MIN, 1.0 - QUANTILE_MIN, N_QUANTILES)
    xm = np.maximum.accumulate(np.quantile(m_flat, q))
    xo = np.maximum.accumulate(np.quantile(o_flat, q))

    out = np.empty_like(t_in, dtype=np.float64)
    for yi in range(t_in.shape[0]):
        flat = t_in[yi].ravel()
        p = np.interp(flat, xm, q, left=QUANTILE_MIN, right=1.0 - QUANTILE_MIN)
        delta = np.interp(p, q, xo) - np.interp(p, q, xm)
        out[yi] = (flat + delta).reshape(t_in[yi].shape)

    return out.astype(np.float32)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        memory_limit=WORKER_MEM,
        local_directory=LOCAL_SCRATCH,
        dashboard_address=None,
    )
    client = Client(cluster)
    print(f"dask: {client}", flush=True)
    print(f"variable: {TEMP_TYPE}  forecast var '{FC_VAR}'  reference var '{OB_VAR}'",
          flush=True)

    try:
        train_fc_files = [fc_path(y) for y in TRAIN_YEARS if fc_path(y).exists()]
        verif_fc_files = [fc_path(y) for y in VERIF_YEARS if fc_path(y).exists()]
        print(f"forecast files: {len(train_fc_files)} train, "
              f"{len(verif_fc_files)} verif", flush=True)
        if not train_fc_files:
            print(f"no forecast files found in {FC_DIR} for pattern {fc_path(TRAIN_YEARS[0]).name}",
                  flush=True)
            return

        for k, (month, yoff) in enumerate(TARGET):
            print(f"\n=== lead {k} -- target month {month:02d} "
                  f"(year offset {yoff}) ===", flush=True)
            first_lead = (k == 0) and SKIP_LEAD_DAYS > 0

            # ---- 1. model pool: all years, all members, month, spin-up cut --
            m_parts = []
            for y in TRAIN_YEARS:
                f = fc_path(y)
                if not f.exists():
                    continue
                ds = xr.open_dataset(f, chunks={FC_TIME_DIM: TIME_CHUNK})
                da = ds[FC_VAR]
                mask = lead_keep_mask(da, month, _init_year(f))
                sub = da.sel({FC_TIME_DIM: mask}, drop=True)
                if sub.sizes.get(FC_TIME_DIM, 0) == 0:
                    ds.close()
                    continue
                sub = sub.isel({MEMBER_DIM: slice(0, N_MEMBERS_KEEP)})
                sub = sub.squeeze(FC_CONCAT_DIM, drop=True)
                sub = xr.apply_ufunc(to_celsius, sub, FC_UNITS_IN,
                                     dask="parallelized",
                                     output_dtypes=[np.float64])
                m_parts.append(sub.transpose(FC_TIME_DIM, MEMBER_DIM,
                                             SPATIAL_LAT, SPATIAL_LON))
                ds.close()
            if not m_parts:
                print("  no model training data -> skip", flush=True)
                continue
            m_da = xr.concat(m_parts, dim="year", join="override")
            print(f"  model pool {dict(m_da.sizes)}", flush=True)

            # ---- 2. reference pool: same month, all years --------------------
            o_parts = []
            for y in TRAIN_YEARS:
                f = ob_path(y, yoff, month)
                if not f.exists():
                    continue
                ds = xr.open_dataset(f, chunks={OB_TIME_DIM: TIME_CHUNK})
                ob = ds[OB_VAR]
                extra = [d for d in ob.dims if d not in (SPATIAL_LAT, SPATIAL_LON)
                         and d != OB_TIME_DIM]
                if extra:
                    ob = ob.isel({d: 0 for d in extra}, drop=True)
                ob = xr.apply_ufunc(to_celsius, ob, OB_UNITS_IN,
                                    dask="parallelized",
                                    output_dtypes=[np.float64])
                o_parts.append(ob.transpose(OB_TIME_DIM, SPATIAL_LAT, SPATIAL_LON))
                ds.close()
            if not o_parts:
                print("  no reference training data -> skip", flush=True)
                continue
            o_da = xr.concat(o_parts, dim="year", join="override")
            print(f"  reference pool {dict(o_da.sizes)}", flush=True)

            # ---- 3. target: verification years -------------------------------
            t_parts = []
            for y in VERIF_YEARS:
                f = fc_path(y)
                if not f.exists():
                    continue
                ds = xr.open_dataset(f, chunks={FC_TIME_DIM: TIME_CHUNK})
                da = ds[FC_VAR]
                mask = lead_keep_mask(da, month, _init_year(f))
                sub = da.sel({FC_TIME_DIM: mask}, drop=True)
                if sub.sizes.get(FC_TIME_DIM, 0) == 0:
                    ds.close()
                    continue
                sub = sub.isel({MEMBER_DIM: slice(0, N_MEMBERS_KEEP)})
                sub = sub.squeeze(FC_CONCAT_DIM, drop=True)
                sub = xr.apply_ufunc(to_celsius, sub, FC_UNITS_IN,
                                     dask="parallelized",
                                     output_dtypes=[np.float64])
                t_parts.append(sub.transpose(FC_TIME_DIM, MEMBER_DIM,
                                             SPATIAL_LAT, SPATIAL_LON))
                ds.close()
            if not t_parts:
                print("  no verification data -> skip", flush=True)
                continue
            t_da = xr.concat(t_parts, dim="year", join="override")
            print(f"  target {dict(t_da.sizes)}", flush=True)

            # ---- 4. point-by-point additive QDM, in latitude strips ----------
            n_lat = m_da.sizes[SPATIAL_LAT]
            n_lon = m_da.sizes[SPATIAL_LON]
            n_year = t_da.sizes["year"]
            n_time = t_da.sizes[FC_TIME_DIM]
            n_mem  = t_da.sizes[MEMBER_DIM]
            print(f"  grid {n_lat} x {n_lon}; target {n_year} x {n_time} x {n_mem}",
                  flush=True)

            corrected_blocks = []
            for lat0 in range(0, n_lat, LAT_BLOCK):
                lat1 = min(lat0 + LAT_BLOCK, n_lat)
                sl = {SPATIAL_LAT: slice(lat0, lat1)}
                n_rows = lat1 - lat0

                m_blk = np.asarray(
                    m_da.isel(sl).transpose("year", FC_TIME_DIM, MEMBER_DIM,
                                            SPATIAL_LAT, SPATIAL_LON).values)
                o_blk = np.asarray(
                    o_da.isel(sl).transpose("year", OB_TIME_DIM,
                                            SPATIAL_LAT, SPATIAL_LON).values)
                t_blk = np.asarray(
                    t_da.isel(sl).transpose("year", FC_TIME_DIM, MEMBER_DIM,
                                            SPATIAL_LAT, SPATIAL_LON).values)

                tasks = []
                for iy in range(n_rows):
                    for ix in range(n_lon):
                        tasks.append(correct_cell(m_blk[..., iy, ix],
                                                  o_blk[..., iy, ix],
                                                  t_blk[..., iy, ix],
                                                  lat0 + iy, ix))
                res = compute(*tasks)

                blk = np.stack(res).reshape(n_rows, n_lon,
                                            n_year, n_time, n_mem)
                assert blk.shape[0] == n_rows and blk.shape[1] == n_lon, blk.shape
                corrected_blocks.append(blk)

                print(f"    strip lat {lat0}:{lat1} done "
                      f"({len(tasks)} cells), block {blk.shape}", flush=True)

                del m_blk, o_blk, t_blk, tasks, res, blk

            corrected_latlon = np.concatenate(corrected_blocks, axis=0)
            assert corrected_latlon.shape[0] == n_lat, corrected_latlon.shape
            assert corrected_latlon.shape[1] == n_lon, corrected_latlon.shape
            corrected_np = corrected_latlon.transpose(2, 3, 4, 0, 1)
            assert corrected_np.shape == (n_year, n_time, n_mem, n_lat, n_lon), \
                corrected_np.shape
            print(f"  corrected array {corrected_np.shape}", flush=True)

            # ---- 5. back to xarray and write ---------------------------------
            corrected = xr.DataArray(
                corrected_np.astype("float32"),
                dims=("year", FC_TIME_DIM, MEMBER_DIM, SPATIAL_LAT, SPATIAL_LON),
                coords={
                    "year": t_da["year"].values,
                    FC_TIME_DIM: t_da[FC_TIME_DIM].values,
                    MEMBER_DIM: t_da[MEMBER_DIM].values,
                    SPATIAL_LAT: t_da[SPATIAL_LAT].values,
                    SPATIAL_LON: t_da[SPATIAL_LON].values,
                },
                name=f"{FC_VAR}_qdm",
                attrs={
                    "units": OUT_UNITS,
                    "long_name": f"QDM-adjusted daily {TEMP_TYPE} "
                                 f"(additive delta, all days corrected)",
                    "method": "Quantile Delta Mapping (Cannon et al. 2015), "
                              "ADDITIVE form",
                    "variable": TEMP_TYPE,
                    "lead_index": str(k),
                    "target_month": f"{month:02d}",
                    "skip_lead_days": str(SKIP_LEAD_DAYS),
                    "members_kept": str(N_MEMBERS_KEEP),
                    "train": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
                    "window_note": (
                        f"model calibrated and corrected on days > {SKIP_LEAD_DAYS} "
                        f"days after init; reference training pool spans the FULL "
                        f"month {month:02d}"
                        if first_lead else
                        f"model and reference both span the full month {month:02d}"
                    ),
                },
            )
            out_path = OUT_DIR / (
                f"{FC_VAR}_24h_mon{month:02d}_init{INIT_MONTH:02d}_"
                f"{VERIF_YEARS[0]}-{VERIF_YEARS[1]}_{REGION}_"
                "res_ERA5-Land_qdm_ERA5-Land.nc"
            )
            out_ds = corrected.to_dataset()
            out_ds.to_netcdf(out_path, encoding={
                v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars
            })
            print(f"  wrote {out_path}", flush=True)

        client.close()
        cluster.close()
        print("done", flush=True)

    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()

