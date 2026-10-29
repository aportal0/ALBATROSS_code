#!/usr/bin/env python3
"""
QDM bias adjustment of ECMWF SEAS5 daily fields vs a reference,
ON THE FULL GRID, POINT BY POINT, with dask parallelism.

MERGED SCRIPT -- precipitation (24h tp vs MSWEP) and temperature (tmin/tmax vs
ERA5-Land) are now the SAME code, driven by the VARIABLE switch below. The two
source scripts were the same program apart from the transfer function and the
dry-day handling; everything else that differed is configuration.

WHAT IS PER-VARIABLE (and must be, physically)
  * "pr"  : MULTIPLICATIVE QDM. Dry days have no mass that can be multiplied;
            the CDF is censored to a trace threshold.
  * "tas" : ADDITIVE QDM. A ratio is meaningless for a variable centred on
            zero that can go negative. No censoring, no re-zeroing -- every
            day is corrected on the whole distribution.

Method (both variables):
  * training 1993-2022, verification 2023-2025;
  * one transfer function per (lead = target month), trained on that month only;
  * the delta (multiplicative or additive) is taken from the training
    climatologies and re-applied to the target value, so the anomaly survives;
  * the first SKIP_LEAD_DAYS days after the initialization are excluded from the
    calibration (forecast spin-up). The reference has no lead, so its pool spans
    the full calendar month -- for the first lead the two windows differ and
    this is recorded in the output attributes.

MEMORY MODEL -- why the spatial loop is blocked
    The per-cell work is scheduled with dask.delayed, but the sample sets are
    materialised as numpy arrays on the client. Loading the whole grid at once
    does not fit in the worker budget (it killed the workers with OOM), so the
    grid is processed in LAT_BLOCK row strips: each strip is loaded, its cells
    corrected, the strip result kept, and the strip arrays freed. The strips are
    concatenated along the latitude axis at the end.
    If a worker still dies, lower LAT_BLOCK first.

Output: one NetCDF per (lead, month) with the corrected daily forecast fields,
on the same grid and dims as the input, verification years only.
No plots.

Run:
    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00 \
         python calibrate_grid.py
"""

from pathlib import Path
import os
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask import delayed, compute
import tempfile

# optional: only needed for VARIABLE in ("tmin", "tmax") with
# MASK_TO_COUNTRY = True
try:
    import functions_spei as fSPEI
except ImportError:  # pragma: no cover
    fSPEI = None


# ======================================================================
# configuration
# ======================================================================
# ---- WHICH VARIABLE TO CALIBRATE --------------------------------------
#   "pr"   : daily precipitation (multiplicative QDM, dry-day censoring)
#   "tmax" : daily maximum temperature (additive QDM, no censoring)
#   "tmin" : daily minimum temperature (additive QDM, no censoring)
VARIABLE = "pr"

LOCAL_SCRATCH = os.environ.get("TMPDIR")
if not LOCAL_SCRATCH or not Path(LOCAL_SCRATCH).is_dir():
    LOCAL_SCRATCH = tempfile.gettempdir()
print(f"dask local_directory: {LOCAL_SCRATCH}", flush=True)

REGION      = "Madagascar"
TRAIN_YEARS = range(1993, 2022 + 1)
VERIF_YEARS = range(1993, 2025 + 1)
INIT_MONTH  = 10

N_MONTHS    = 4

N_MEMBERS_KEEP = 25        # members common to all years (verify!)
SKIP_LEAD_DAYS = 10        # drop the first N days after init from CALIBRATION

MEMBER_DIM      = "number"
FC_TIME_DIM     = "forecast_period"
FC_SELECT_COORD = "valid_time"
FC_CONCAT_DIM   = "forecast_reference_time"

N_QUANTILES  = 50
QUANTILE_MIN = 0.01
RNG_SEED     = 12345

SPATIAL_LAT_FC = "lat"       # forecast spatial names
SPATIAL_LON_FC = "lon"

# ---- parallelism / memory ----
N_WORKERS  = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
WORKER_MEM = "3GB"
TIME_CHUNK = 40
LAT_BLOCK  = 20            # latitude rows loaded per strip (lower if workers die)

C_TO_K = 273.15


# ======================================================================
# per-variable configuration
# ======================================================================
if VARIABLE == "pr":
    # ---------------------------------------------------------------- precip
    # forecast: SEAS5 24h tp in METRES.  reference: MSWEP, already mm/day.
    FC_VAR = "tp"
    OB_VAR = "precipitation"
    FC_UNITS_TO_MM = 1000.0
    OB_UNITS_TO_MM = 1.0

    TRACE_MM     = 0.1     # censoring / trace threshold (mm/day)
    THRESHOLD_MM = 1.0     # DIAGNOSTIC ONLY -- never filters the CDF pool

    METHOD = "multiplicative"
    CENSOR_DRY = True

    OB_TIME_DIM = "time"
    OB_LAT, OB_LON = "lat", "lon"

    MASK_TO_COUNTRY = False   # MSWEP reference was already cut to the region
    REF_TAG = "MSWEP"

    FC_DIR = Path(
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}"
        f"/{FC_VAR}/regridded_ERA5-Land"
    )
    OB_DIR = Path(
        "/ec/res4/scratch/ecme4047/MSWEP/MSWEP_V316_test/Past/Daily/"
        "regridded_ERA5-Land"
    )
    OUT_DIR = Path(
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}"
        f"/{FC_VAR}/calibrated_MSWEP"
    )

    def fc_path(y):
        return FC_DIR / (
            f"tp_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}"
            f"_res_ERA5-Land_patch.nc"
        )

    def ob_path(y, yoff, month):
        return OB_DIR / (
            f"precip_daily_{y + yoff}{month:02d}_{REGION}_res_ERA5-Land.nc"
        )

elif VARIABLE in ("tmax", "tmin"):
    # ---------------------------------------------------------------- temp
    # VARIABLE itself selects the type -- no second switch to keep in sync.
    TEMP_TYPE = VARIABLE
    FC_VAR_BY_TYPE = {"tmin": "mn2t24", "tmax": "mx2t24"}
    FC_VAR = FC_VAR_BY_TYPE[TEMP_TYPE]
    OB_VAR = "t2m"

    # Units: BOTH series must be in the SAME unit. K -> degC subtracts 273.15;
    # if the files are already in degC leave both at "celsius".
    FC_UNITS_IN = "K"      # "K" or "celsius"
    OB_UNITS_IN = "K"
    OUT_UNITS   = "K"

    METHOD = "additive"
    CENSOR_DRY = False
    TRACE_MM = None

    OB_TIME_DIM = "valid_time"
    OB_LAT, OB_LON = "latitude", "longitude"   # reference files use the long form

    MASK_TO_COUNTRY = True
    REF_TAG = "ERA5-Land"

    FC_DIR = Path(
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}"
        f"/{FC_VAR}/regridded_ERA5-Land"
    )
    OB_DIR = Path(f"/ec/res4/scratch/ecme4047/ERA5-Land/{OB_VAR}/daily")
    OUT_DIR = Path(
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}"
        f"/{FC_VAR}/calibrated_ERA5-Land"
    )

    def fc_path(y):
        return FC_DIR / (
            f"{FC_VAR}_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}"
            f"_res_ERA5-Land_patch.nc"
        )

    def ob_path(y, yoff, month):
        stat = "minimum" if TEMP_TYPE == "tmin" else "maximum"
        return OB_DIR / str(y + yoff) / (
            f"{OB_VAR}_daily_{stat}_{y + yoff}{month:02d}.nc"
        )

else:
    raise ValueError(f"VARIABLE must be 'pr' or 'tas', got {VARIABLE!r}")


OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_MONTHS = [((INIT_MONTH - 1 + k) % 12) + 1 for k in range(N_MONTHS)]
TARGET = [(m, (INIT_MONTH - 1 + k) // 12) for k, m in enumerate(LEAD_MONTHS)]


# ======================================================================
# helpers
# ======================================================================
def cut_to_country(da, name_country):
    """Cut a DataArray to preset country boundaries."""
    if fSPEI is None:
        raise ImportError(
            "MASK_TO_COUNTRY is on but 'functions_spei' could not be imported "
            "-- put it on PYTHONPATH or set MASK_TO_COUNTRY = False."
        )
    box = fSPEI.boxes_african_countries(name_country)
    return fSPEI.subset_box(da, box)


def _init_year(fname):
    stem = fname.name if isinstance(fname, Path) else str(fname)
    return int(stem.split("init")[1][:4])


def lead_keep_mask(da, month, init_year, month_day=1):
    """Steps to keep: valid_time in `month` AND lead >= SKIP_LEAD_DAYS days
    after the initialization date (forecast spin-up excluded)."""
    in_month = da[FC_SELECT_COORD].dt.month == month
    if not SKIP_LEAD_DAYS:
        return in_month
    cutoff = (np.datetime64(f"{int(init_year):04d}-{INIT_MONTH:02d}-{month_day:02d}")
              + np.timedelta64(SKIP_LEAD_DAYS, "D"))
    return in_month & (da[FC_SELECT_COORD] >= cutoff)


def to_target_units(arr, kind):
    """Precipitation: forecast m -> mm, reference passed through.
    Temperature: everything -> Kelvin."""
    if VARIABLE == "pr":
        factor = FC_UNITS_TO_MM if kind == "fc" else OB_UNITS_TO_MM
        return arr * factor
    units_in = FC_UNITS_IN if kind == "fc" else OB_UNITS_IN
    if str(units_in).lower() in ("c", "celsius"):
        return arr + C_TO_K
    return arr


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


def qdm_transfer(m_pool, o_pool, target):
    """
    The one place the two methods differ.

    multiplicative (pr) :  y = x * ( xo(p) / xm(p) )
    additive       (tas):  y = x + ( xo(p) - xm(p) )

    with p = F_mod_train(x) the position of the target in the model training
    climatology, so the anomaly relative to the model survives the correction.

    Returns (corrected, p, o_at_p) -- p and o_at_p are needed downstream for
    the dry-day test on the observed quantile, and are discarded by the
    temperature path.
    """
    m_arr = np.asarray(m_pool, dtype=np.float64)
    o_arr = np.asarray(o_pool, dtype=np.float64)
    t_arr = np.asarray(target, dtype=np.float64)

    q = np.linspace(QUANTILE_MIN, 1.0 - QUANTILE_MIN, N_QUANTILES)
    xm = np.maximum.accumulate(np.quantile(m_arr, q))
    xo = np.maximum.accumulate(np.quantile(o_arr, q))

    if METHOD == "multiplicative":
        # guard against a zero model quantile before dividing
        xm = np.where(xm <= 0.0, np.finfo(np.float64).tiny, xm)

    p = np.interp(t_arr, xm, q, left=QUANTILE_MIN, right=1.0 - QUANTILE_MIN)
    o_at_p = np.interp(p, q, xo)
    m_at_p = np.interp(p, q, xm)

    if METHOD == "multiplicative":
        corrected = t_arr * (o_at_p / m_at_p)
    else:
        corrected = t_arr + (o_at_p - m_at_p)

    return corrected, p, o_at_p


@delayed
def correct_cell(m_pool, o_pool, t_in, iy, ix):
    """
    Correct ONE grid cell -- the whole QDM for that point.

    m_pool : (ny_m, ntime_m, nmem) model pool, ALL years/members
    o_pool : (ny_o, ntime_o)       reference pool, ALL years
    t_in   : (ny, ntime, nmem)     target for this cell
    iy, ix : grid indices (passed through for provenance)

    The transfer function is estimated from the POOLED model and reference
    samples (all years and members flattened), then applied per year and per
    member of the target.

    For "pr" the arrays arrive already censored (zeros -> U(0, trace)) and the
    dry days are re-zeroed here: where the observed value at that quantile is
    below the trace, and where the corrected value falls below it.
    For "tas" nothing is censored and nothing is re-zeroed.
    """
    m_flat = np.asarray(m_pool, dtype=np.float64).ravel()
    o_flat = np.asarray(o_pool, dtype=np.float64).ravel()
    m_flat = m_flat[np.isfinite(m_flat)]
    o_flat = o_flat[np.isfinite(o_flat)]

    t_in = np.asarray(t_in, dtype=np.float64)

    # not enough sample in this cell
    if m_flat.size < 2 or o_flat.size < 2:
        if CENSOR_DRY:
            # return the target re-zeroed only, so the dry-day convention is
            # applied uniformly
            return require_zero(t_in, TRACE_MM).astype(np.float32)
        return t_in.astype(np.float32)

    out = np.empty_like(t_in, dtype=np.float64)
    for yi in range(t_in.shape[0]):
        tgt = t_in[yi]                       # (ntime, nmem)
        corrected, p, o_at_p = qdm_transfer(m_flat, o_flat, tgt.ravel())

        if CENSOR_DRY:
            # zero where the OBSERVED value at that quantile is below the
            # trace: the position in the distribution is dry in the reference
            corrected[o_at_p <= TRACE_MM] = 0.0
            # zero where the CORRECTED value is below the trace
            corrected[corrected < TRACE_MM] = 0.0

        out[yi] = corrected.reshape(tgt.shape)

    return out.astype(np.float32)


# ======================================================================
# output attributes
# ======================================================================
def build_attrs(k, month, first_lead):
    """Output attributes, with the method and the dry-day convention recorded."""
    if VARIABLE == "pr":
        units = "mm/day"
        long_name = ("QDM-adjusted daily precipitation "
                     "(multiplicative delta, dry days censored)")
        method = ("Quantile Delta Mapping (Cannon et al. 2015), "
                  "MULTIPLICATIVE form, dry days censored below "
                  f"{TRACE_MM} mm/day")
        variable = "pr"
    else:
        units = OUT_UNITS
        long_name = (f"QDM-adjusted daily {TEMP_TYPE} "
                     f"(additive delta, all days corrected)")
        method = ("Quantile Delta Mapping (Cannon et al. 2015), "
                  "ADDITIVE form")
        variable = TEMP_TYPE

    attrs = {
        "units": units,
        "long_name": long_name,
        "method": method,
        "variable": variable,
        "lead_index": str(k),
        "target_month": f"{month:02d}",
        "skip_lead_days": str(SKIP_LEAD_DAYS),
        "members_kept": str(N_MEMBERS_KEEP),
        "n_quantiles": str(N_QUANTILES),
        "train": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
        "reference_product": REF_TAG,
        "window_note": (
            f"model calibrated and corrected on days > {SKIP_LEAD_DAYS} "
            f"days after init; reference training pool spans the FULL "
            f"month {month:02d}"
            if first_lead else
            f"model and reference both span the full month {month:02d}"
        ),
    }
    if VARIABLE == "pr":
        attrs["trace_mm"] = str(TRACE_MM)
        attrs["rng_seed"] = str(RNG_SEED)
    return attrs


# ======================================================================
# main
# ======================================================================
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
    print(f"variable: {VARIABLE}  method: {METHOD}  forecast var '{FC_VAR}'  "
          f"reference var '{OB_VAR}'", flush=True)

    try:
        train_fc_files = [fc_path(y) for y in TRAIN_YEARS if fc_path(y).exists()]
        verif_fc_files = [fc_path(y) for y in VERIF_YEARS if fc_path(y).exists()]
        print(f"forecast files: {len(train_fc_files)} train, "
              f"{len(verif_fc_files)} verif", flush=True)
        if not train_fc_files:
            print(f"no forecast files found in {FC_DIR}", flush=True)
            return

        for k, (month, yoff) in enumerate(TARGET):
            print(f"\n=== lead {k} -- target month {month:02d} "
                  f"(year offset {yoff}) ===", flush=True)
            first_lead = (k == 0) and SKIP_LEAD_DAYS > 0

            # ---- 1. model pool: all years, all members, month, spin-up cut --
            m_parts = []
            m_years = []
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
                sub = xr.apply_ufunc(to_target_units, sub, "fc",
                                     dask="parallelized",
                                     output_dtypes=[np.float64])
                m_parts.append(sub.transpose(FC_TIME_DIM, MEMBER_DIM,
                                             SPATIAL_LAT_FC, SPATIAL_LON_FC))
                m_years.append(int(y))
                ds.close()
            if not m_parts:
                print("  no model training data -> skip", flush=True)
                continue
            m_da = xr.concat(m_parts, dim="init_year", join="override")
            m_da = m_da.assign_coords(init_year=m_years)
            # hourly precip accumulations can go slightly negative
            m_da = m_da.clip(min=0.0) if VARIABLE == "pr" else m_da
            print(f"  model pool {dict(m_da.sizes)}", flush=True)

            # ---- 2. reference pool: same month, all years --------------------
            o_parts = []
            o_years = []
            for y in TRAIN_YEARS:
                f = ob_path(y, yoff, month)
                if not f.exists():
                    continue
                ds = xr.open_dataset(f, chunks={OB_TIME_DIM: TIME_CHUNK})
                ob = ds[OB_VAR]
                if MASK_TO_COUNTRY:
                    ob = cut_to_country(ob, REGION)
                extra = [d for d in ob.dims
                         if d not in (OB_LAT, OB_LON) and d != OB_TIME_DIM]
                if extra:
                    ob = ob.isel({d: 0 for d in extra}, drop=True)
                ob = xr.apply_ufunc(to_target_units, ob, "ob",
                                    dask="parallelized",
                                    output_dtypes=[np.float64])
                o_parts.append(ob.transpose(OB_TIME_DIM, OB_LAT, OB_LON))
                o_years.append(int(y))
                ds.close()
            if not o_parts:
                print("  no reference training data -> skip", flush=True)
                continue
            o_da = xr.concat(o_parts, dim="init_year", join="override")
            o_da = o_da.assign_coords(init_year=o_years)
            o_da = o_da.clip(min=0.0) if VARIABLE == "pr" else o_da
            print(f"  reference pool {dict(o_da.sizes)}", flush=True)

            # ---- 3. target: verification years -------------------------------
            t_parts = []
            t_years = []
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
                sub = xr.apply_ufunc(to_target_units, sub, "fc",
                                     dask="parallelized",
                                     output_dtypes=[np.float64])
                t_parts.append(sub.transpose(FC_TIME_DIM, MEMBER_DIM,
                                             SPATIAL_LAT_FC, SPATIAL_LON_FC))
                t_years.append(int(y))
                ds.close()
            if not t_parts:
                print("  no verification data -> skip", flush=True)
                continue
            t_da = xr.concat(t_parts, dim="init_year", join="override")
            t_da = t_da.assign_coords(init_year=t_years)
            t_da = t_da.clip(min=0.0) if VARIABLE == "pr" else t_da
            print(f"  target     {dict(t_da.sizes)}", flush=True)

            # ---- 4. censoring (precipitation only) ---------------------------
            # Zeros -> U(0, trace) on all three sample sets, BEFORE any mapping.
            def _censor_block(block, seed_off):
                rng = np.random.default_rng(RNG_SEED + seed_off)
                out, _ = censor_dry(block, TRACE_MM, rng)
                return out

            if CENSOR_DRY:
                m_c = xr.apply_ufunc(
                    _censor_block, m_da, kwargs={"seed_off": 0},
                    input_core_dims=[[FC_TIME_DIM, MEMBER_DIM]],
                    output_core_dims=[[FC_TIME_DIM, MEMBER_DIM]],
                    vectorize=True, dask="parallelized",
                    output_dtypes=[np.float64],
                    dask_gufunc_kwargs={"allow_rechunk": True},
                )
                o_c = xr.apply_ufunc(
                    _censor_block, o_da, kwargs={"seed_off": 1},
                    input_core_dims=[[OB_TIME_DIM]],
                    output_core_dims=[[OB_TIME_DIM]],
                    vectorize=True, dask="parallelized",
                    output_dtypes=[np.float64],
                    dask_gufunc_kwargs={"allow_rechunk": True},
                )
                t_c = xr.apply_ufunc(
                    _censor_block, t_da, kwargs={"seed_off": 2},
                    input_core_dims=[[FC_TIME_DIM, MEMBER_DIM]],
                    output_core_dims=[[FC_TIME_DIM, MEMBER_DIM]],
                    vectorize=True, dask="parallelized",
                    output_dtypes=[np.float64],
                    dask_gufunc_kwargs={"allow_rechunk": True},
                )
            else:
                m_c, o_c, t_c = m_da, o_da, t_da

            # ---- 5. point-by-point QDM, in latitude strips -------------------
            n_lat = m_c.sizes[SPATIAL_LAT_FC]
            n_lon = m_c.sizes[SPATIAL_LON_FC]
            n_year = t_c.sizes["init_year"]
            n_time = t_c.sizes[FC_TIME_DIM]
            n_mem  = t_c.sizes[MEMBER_DIM]
            print(f"  grid {n_lat} x {n_lon}; target {n_year} x {n_time} x {n_mem}",
                  flush=True)

            # the strip loop indexes model AND reference with the same integer
            # lat0:lat1 range -- that is only safe if the two grids line up
            fc_lat_vals = np.asarray(t_da[SPATIAL_LAT_FC].values)
            ob_lat_vals = np.asarray(o_da[OB_LAT].values)
            if fc_lat_vals.shape != ob_lat_vals.shape:
                raise ValueError(
                    f"forecast/reference lat length mismatch "
                    f"({fc_lat_vals.shape} vs {ob_lat_vals.shape}): the strip "
                    f"loop would calibrate every cell against the wrong "
                    f"reference row. Re-regrid one of them."
                )
            if not np.allclose(fc_lat_vals, ob_lat_vals, atol=1e-6):
                print("  WARNING: forecast and reference latitude coords differ; "
                      "strip alignment assumes identical ordering.", flush=True)

            corrected_blocks = []
            for lat0 in range(0, n_lat, LAT_BLOCK):
                lat1 = min(lat0 + LAT_BLOCK, n_lat)
                fc_sl = {SPATIAL_LAT_FC: slice(lat0, lat1)}
                ob_sl = {OB_LAT: slice(lat0, lat1)}
                n_rows = lat1 - lat0

                # materialise ONE strip (model, reference, target) as numpy
                m_blk = np.asarray(
                    m_c.isel(fc_sl).transpose("init_year", FC_TIME_DIM, MEMBER_DIM,
                                               SPATIAL_LAT_FC,
                                               SPATIAL_LON_FC).values)
                o_blk = np.asarray(
                    o_c.isel(ob_sl).transpose("init_year", OB_TIME_DIM,
                                              OB_LAT, OB_LON).values)
                t_blk = np.asarray(
                    t_c.isel(fc_sl).transpose("init_year", FC_TIME_DIM, MEMBER_DIM,
                                               SPATIAL_LAT_FC,
                                               SPATIAL_LON_FC).values)

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
                assert blk.shape == (n_rows, n_lon, n_year, n_time, n_mem), \
                    blk.shape
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

            # ---- 6. back to xarray and write ---------------------------------
            corrected = xr.DataArray(
                corrected_np.astype("float32"),
                dims=("init_year", FC_TIME_DIM, MEMBER_DIM,
                      SPATIAL_LAT_FC, SPATIAL_LON_FC),
                coords={
                    "init_year": t_da["init_year"].values,
                    FC_TIME_DIM: t_da[FC_TIME_DIM].values,
                    MEMBER_DIM: t_da[MEMBER_DIM].values,
                    SPATIAL_LAT_FC: t_da[SPATIAL_LAT_FC].values,
                    SPATIAL_LON_FC: t_da[SPATIAL_LON_FC].values,
                },
                name=f"{FC_VAR}_qdm",
                attrs=build_attrs(k, month, first_lead),
            )
            out_path = OUT_DIR / (
                f"{FC_VAR}_24h_mon{month:02d}_init{INIT_MONTH:02d}_"
                f"{VERIF_YEARS[0]}-{VERIF_YEARS[-1]}_{REGION}_"
                f"res_ERA5-Land_qdm_{REF_TAG}.nc"
            )
            out_ds = corrected.to_dataset()
            out_ds.to_netcdf(out_path, encoding={
                v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars
            })
            print(f"  wrote {out_path}", flush=True)

        print("done", flush=True)

    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()



