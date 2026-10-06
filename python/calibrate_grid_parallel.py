#!/usr/bin/env python3
"""
QDM bias adjustment of ECMWF SEAS5 daily precipitation (24h tp) vs MSWEP,
ON THE FULL GRID, POINT BY POINT, with dask parallelism.

This is the operational counterpart of qdm_point_debug.py: the SAME correction
logic, applied to every grid cell independently. No stacking and no apply_ufunc
across coordinates -- each cell is corrected on its own 1-D time series, which
is exactly what the single-point debug script does.

Method:
  * training 1993-2022, verification 2023-2025;
  * one transfer function per (lead = target month), trained on that month only;
  * multiplicative QDM: delta from the training climatologies, re-applied to the
    target value, so the anomaly survives;
  * dry days as CENSORED values below a trace threshold of 0.05 mm/day: zeros in
    both model and observations become nonzero uniform values below the trace
    BEFORE the correction; values below the trace are set back to zero AFTER;
  * the first SKIP_LEAD_DAYS days after the initialization are excluded from the
    calibration (forecast spin-up). The observation has no lead, so its pool
    spans the full calendar month -- for the first lead the two windows differ
    and this is recorded in the output attributes.

Output: one NetCDF per (lead, month) with the corrected daily forecast fields
(tp_qdm), on the same grid and dims as the input, verification years only.
No plots.

Run:
    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00 \
         python qdm_grid_parallel.py
"""

from pathlib import Path
import os
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
import tempfile

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
LOCAL_SCRATCH = os.environ.get("TMPDIR")
if not LOCAL_SCRATCH or not Path(LOCAL_SCRATCH).is_dir():
    LOCAL_SCRATCH = tempfile.gettempdir()
print(f"dask local_directory: {LOCAL_SCRATCH}", flush=True)

FC_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/regridded_ERA5-Land")
OB_DIR = Path("/ec/res4/scratch/ecme4047/MSWEP/MSWEP_V316_test/Past/Daily/regridded_ERA5-Land")
OUT_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/calibrated_MSWEP")

REGION      = "Madagascar"
TRAIN_YEARS = range(1993, 2022 + 1)
VERIF_YEARS = range(2023, 2025 + 1)
INIT_MONTH  = 10
N_MONTHS = 3

FC_VAR = "tp"
OB_VAR = "precipitation"

FC_UNITS_TO_MM = 1000.0    # tp in metres (24 h accumulation) -> mm/day
OB_UNITS_TO_MM = 1.0       # MSWEP already mm/day

TRACE_MM     = 0.1         # censoring / trace threshold (mm/day)
THRESHOLD_MM = 1.0         # DIAGNOSTIC ONLY -- never filters the CDF pool
N_MEMBERS_KEEP = 25        # members common to all years (verify!)
SKIP_LEAD_DAYS = 10        # drop the first N days after init from CALIBRATION

MEMBER_DIM      = "number"
FC_TIME_DIM     = "forecast_period"
FC_SELECT_COORD = "valid_time"
FC_CONCAT_DIM   = "forecast_reference_time"
OB_TIME_DIM     = "time"

SPATIAL_LAT = "lat"
SPATIAL_LON = "lon"

N_QUANTILES  = 100
QUANTILE_MIN = 0.01
RNG_SEED     = 12345

# ---- parallelism ----
N_WORKERS  = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
WORKER_MEM = "3GB"
TIME_CHUNK = 40

OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_MONTHS = [((INIT_MONTH - 1 + k) % 12) + 1 for k in range(N_MONTHS)]
TARGET = [(m, (INIT_MONTH - 1 + k) // 12) for k, m in enumerate(LEAD_MONTHS)]


# ----------------------------------------------------------------------
# helpers (identical logic to the single-point script)
# ----------------------------------------------------------------------
def _init_year(fname):
    stem = fname.name if isinstance(fname, Path) else str(fname)
    return int(stem.split("init")[1][:4])


def fc_path(y):
    return FC_DIR / f"tp_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}_res_ERA5-Land_patch.nc"


def ob_path(y, yoff, month):
    return OB_DIR / f"precip_daily_{y + yoff}{month:02d}_{REGION}_res_ERA5-Land.nc"


def lead_keep_mask(da, month, init_year, month_day=1):
    """Steps to keep: valid_time in `month` AND lead >= SKIP_LEAD_DAYS days
    after the initialization date (forecast spin-up excluded)."""
    in_month = da[FC_SELECT_COORD].dt.month == month
    if not SKIP_LEAD_DAYS:
        return in_month
    cutoff = (np.datetime64(f"{int(init_year):04d}-{INIT_MONTH:02d}-{month_day:02d}")
              + np.timedelta64(SKIP_LEAD_DAYS, "D"))
    return in_month & (da[FC_SELECT_COORD] >= cutoff)


def censor_dry(arr, trace=TRACE_MM, rng=None):
    """Exact zeros -> nonzero uniform values in (0, trace)."""
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    zeros = out == 0.0
    n = int(zeros.sum())
    if n:
        out[zeros] = rng.uniform(0.0, trace, size=n)
    return out, n


def require_zero(arr, trace=TRACE_MM):
    """Below the trace threshold -> exact zero (dry days stay dry)."""
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    out[out < trace] = 0.0
    return out


def qdm_point(mod_train, obs_train, mod_target,
              nq=N_QUANTILES, qmin=QUANTILE_MIN):
    """Multiplicative QDM on three 1-D all-days arrays (mm/day, already
    censored). Returns the corrected target, same length."""
    mod_target = np.asarray(mod_target, dtype=np.float64)
    mod_train = np.asarray(mod_train, dtype=np.float64)
    obs_train = np.asarray(obs_train, dtype=np.float64)

    mod_train = mod_train[np.isfinite(mod_train)]
    obs_train = obs_train[np.isfinite(obs_train)]

    if mod_train.size < 2 or obs_train.size < 2 or mod_target.size == 0:
        return mod_target.astype(np.float32)

    q = np.linspace(qmin, 1.0 - qmin, nq)
    xm = np.maximum.accumulate(np.quantile(mod_train, q))
    xo = np.maximum.accumulate(np.quantile(obs_train, q))
    xm = np.where(xm <= 0.0, np.finfo(np.float64).tiny, xm)

    p = np.interp(mod_target, xm, q, left=qmin, right=1.0 - qmin)
    ratio = np.interp(p, q, xo) / np.interp(p, q, xm)
    return (mod_target * ratio).astype(np.float32)


def correct_stack(m_pool, o_pool, t_cens, iy, ix):
    """
    Correct ONE grid cell.

    m_pool : (n_pool, n_members) model wet/main pool, already censored
    o_pool : (n_pool,)           obs pool, already censored
    t_cens : (ny, ntime, n_members) censored target for this cell
    Returns the corrected target array, same shape as t_cens.

    The transfer function is estimated from the POOLED model sample (all years
    and all members, flattened) and the pooled obs sample, then applied per
    member of the target -- the same construction as the single-point script.
    """
    m_flat = m_pool.ravel()
    o_flat = o_pool.ravel()

    out = np.zeros_like(t_cens, dtype=np.float32)
    ny = t_cens.shape[0]
    for yi in range(ny):
        tgt = t_cens[yi]                       # (ntime, n_members)
        sh = tgt.shape
        out[yi] = qdm_point(m_flat, o_flat, tgt.ravel()).reshape(sh)
    return require_zero(out, TRACE_MM)


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

    try:
        train_fc_files = [fc_path(y) for y in TRAIN_YEARS if fc_path(y).exists()]
        verif_fc_files = [fc_path(y) for y in VERIF_YEARS if fc_path(y).exists()]
        print(f"forecast files: {len(train_fc_files)} train, "
              f"{len(verif_fc_files)} verif", flush=True)

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
                # (time, number, lat, lon) -> keep member and spatial dims
                sub = (sub.isel({MEMBER_DIM: slice(0, N_MEMBERS_KEEP)})
                          .squeeze(FC_CONCAT_DIM, drop=True) * FC_UNITS_TO_MM)
                m_parts.append(sub.transpose(FC_TIME_DIM, MEMBER_DIM,
                                             SPATIAL_LAT, SPATIAL_LON))
                ds.close()
            if not m_parts:
                print("  no model training data -> skip", flush=True)
                continue
            m_da = xr.concat(m_parts, dim="year", join="override")
            m_da = m_da.clip(min=0.0)
            print(f"  model pool {dict(m_da.sizes)}", flush=True)

            # ---- 2. obs pool: same month, all years -------------------------
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
                o_parts.append((ob * OB_UNITS_TO_MM).transpose(
                    OB_TIME_DIM, SPATIAL_LAT, SPATIAL_LON))
                ds.close()
            if not o_parts:
                print("  no obs training data -> skip", flush=True)
                continue
            o_da = xr.concat(o_parts, dim="year", join="override")
            o_da = o_da.clip(min=0.0)
            print(f"  obs pool   {dict(o_da.sizes)}", flush=True)

            # ---- 3. target: verification years ------------------------------
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
                sub = (sub.isel({MEMBER_DIM: slice(0, N_MEMBERS_KEEP)})
                          .squeeze(FC_CONCAT_DIM, drop=True) * FC_UNITS_TO_MM)
                t_parts.append(sub.transpose(FC_TIME_DIM, MEMBER_DIM,
                                             SPATIAL_LAT, SPATIAL_LON))
                ds.close()
            if not t_parts:
                print("  no verification data -> skip", flush=True)
                continue
            t_da = xr.concat(t_parts, dim="year", join="override")
            t_da = t_da.clip(min=0.0)
            print(f"  target     {dict(t_da.sizes)}", flush=True)

            # ---- 4. censoring (per cell, needs a seeded rng per block) ------
            def _censor_block(block, seed_off):
                rng = np.random.default_rng(RNG_SEED + seed_off)
                out, _ = censor_dry(block, TRACE_MM, rng)
                return out

            m_c = xr.apply_ufunc(
                _censor_block, m_da,
                kwargs={"seed_off": 0},
                input_core_dims=[[FC_TIME_DIM, MEMBER_DIM]],
                output_core_dims=[[FC_TIME_DIM, MEMBER_DIM]],
                vectorize=True, dask="parallelized", output_dtypes=[np.float64],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )
            o_c = xr.apply_ufunc(
                _censor_block, o_da,
                kwargs={"seed_off": 1},
                input_core_dims=[[OB_TIME_DIM]],
                output_core_dims=[[OB_TIME_DIM]],
                vectorize=True, dask="parallelized", output_dtypes=[np.float64],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )
            t_c = xr.apply_ufunc(
                _censor_block, t_da,
                kwargs={"seed_off": 2},
                input_core_dims=[[FC_TIME_DIM, MEMBER_DIM]],
                output_core_dims=[[FC_TIME_DIM, MEMBER_DIM]],
                vectorize=True, dask="parallelized", output_dtypes=[np.float64],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )

            # ---- 5. point-by-point QDM -------------------------------------
            m_stack = m_c.stack(cell=(SPATIAL_LAT, SPATIAL_LON)).transpose(
                "cell", "year", FC_TIME_DIM, MEMBER_DIM)
            o_stack = o_c.stack(cell=(SPATIAL_LAT, SPATIAL_LON)).transpose(
                "cell", "year", OB_TIME_DIM)
            t_stack = t_c.stack(cell=(SPATIAL_LAT, SPATIAL_LON)).transpose(
                "cell", "year", FC_TIME_DIM, MEMBER_DIM)

            # dask chunking: one chunk per group of cells along `cell`
            m_stack = m_stack.chunk({"cell": 1})
            o_stack = o_stack.chunk({"cell": 1})
            t_stack = t_stack.chunk({"cell": 1})

            corrected = xr.apply_ufunc(
                correct_stack,
                m_stack, o_stack, t_stack,
                input_core_dims=[["year", FC_TIME_DIM, MEMBER_DIM],
                                 ["year", OB_TIME_DIM],
                                 ["year", FC_TIME_DIM, MEMBER_DIM]],
                output_core_dims=[["year", FC_TIME_DIM, MEMBER_DIM]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.float32],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )

            # ---- 6. back to grid, write ------------------------------------
            corrected = corrected.unstack("cell")
            corrected = corrected.transpose("year", FC_TIME_DIM, MEMBER_DIM,
                                            SPATIAL_LAT, SPATIAL_LON)
            corrected = corrected.rename(f"{FC_VAR}_qdm").assign_attrs(
                units="mm/day",
                long_name="QDM-adjusted daily precipitation "
                          "(wet days corrected, dry days kept dry)",
                trace_mm=str(TRACE_MM),
                threshold_mm=str(THRESHOLD_MM),
                method="Quantile Delta Mapping (Cannon et al. 2015)",
                dry_day_treatment="censored below trace; zeros -> U(0,trace) "
                                  "before correction, re-zeroed after",
                lead_index=str(k), target_month=f"{month:02d}",
                skip_lead_days=str(SKIP_LEAD_DAYS),
                members_kept=str(N_MEMBERS_KEEP),
                train=f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
                window_note=(
                    f"model calibrated and corrected on days > {SKIP_LEAD_DAYS} "
                    f"days after init; MSWEP training pool spans the FULL month {month:02d}"
                    if first_lead else
                    f"model and MSWEP both span the full month {month:02d}"
                ),
            )
            out_path = OUT_DIR / (
                f"{FC_VAR}_24h_mon{month:02d}_init{INIT_MONTH:02d}_"
                f"{VERIF_YEARS[0]}-{VERIF_YEARS[1]}_{REGION}_"
                "res_ERA5-Land_qdm_MSWEP.nc"
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

