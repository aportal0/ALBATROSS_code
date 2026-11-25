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

Output: one NetCDF per (variable, lead, month) with the corrected daily
forecast fields, on the same grid and dims as the input, verification years
only. No plots.

MULTI-VARIABLE RUN
    A single invocation now runs EVERY variable in RUN_VARIABLES below, one
    after the other, instead of one variable per edit-and-rerun. Everything
    that differed between the variables was configuration, so it is now a dict
    (_VAR_CFG) and everything that is physics stays one function that takes the
    config as an argument. Nothing that a delayed task needs is read from a
    module-level global: the method, the dry-day switch and the trace threshold
    are passed in explicitly, so a task can never pick up the settings of the
    variable that happened to be configured last.

    Set RUN_VARIABLES to any subset, e.g. ("tmax",) to redo only tmax.


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


# ======================================================================
# shared configuration (identical for every variable)
# ======================================================================
REGION      = "Madagascar"
TRAIN_YEARS = range(1993, 2022 + 1)
VERIF_YEARS = range(1993, 2025 + 1)   # NOTE: overlaps TRAIN_YEARS -- see README
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

N_WORKERS  = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
WORKER_MEM = "3GB"
TIME_CHUNK = 40
LAT_BLOCK  = 20            # latitude rows loaded per strip (lower if workers die)

C_TO_K = 273.15


# ======================================================================
# per-variable configuration
# ======================================================================
# One entry per variable. Plain values and small closures over a single base
# directory -- NOT over a module-level global, so no state leaks between runs.
#
#   method          : the QDM transfer function ("multiplicative" / "additive")
#   censor_dry      : whether zeros are censored to U(0, trace) and re-zeroed
#   trace_mm        : the trace threshold, None when censoring is off
#   units_to_target : "factor" (multiply) or "offset_kelvin" (add 273.15)
#   mask_to_country : whether the reference is cut with fSPEI.subset_box
#
# soil is not a physical axis here: nothing in the correction reads it, so it
# is not carried.


def _pr_cfg():
    """Precipitation: SEAS5 24h tp in METRES vs MSWEP in mm/day."""
    OUT_DIR = Path(
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}"
        f"/tp/calibrated_MSWEP"
    )
    FC_DIR = Path(
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}"
        f"/tp/regridded_ERA5-Land"
    )
    OB_DIR = Path(
        "/ec/res4/scratch/ecme4047/MSWEP/MSWEP_V316_test/Past/Daily/"
        "regridded_ERA5-Land"
    )
    return dict(
        name="pr",
        fc_var="tp",
        ob_var="precipitation",
        method="multiplicative",
        censor_dry=True,
        trace_mm=0.1,               # censoring / trace threshold (mm/day)
        threshold_mm=1.0,           # DIAGNOSTIC ONLY -- never filters the pool
        units="mm/day",
        units_to_target="factor",
        fc_units_to_target=1000.0,  # metres -> mm
        ob_units_to_target=1.0,     # already mm/day
        ob_time_dim="time",
        ob_lat="lat",
        ob_lon="lon",
        mask_to_country=False,      # MSWEP reference already cut to the region
        ref_tag="MSWEP",
        fc_dir=FC_DIR,
        ob_dir=OB_DIR,
        out_dir=OUT_DIR,
        fc_path=lambda y, _d=FC_DIR: _d / (
            f"tp_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}"
            f"_res_ERA5-Land_patch.nc"
        ),
        ob_path=lambda y, yoff, month, _d=OB_DIR: _d / (
            f"precip_daily_{y + yoff}{month:02d}_{REGION}_res_ERA5-Land.nc"
        ),
        long_name="QDM-adjusted daily precipitation "
                  "(multiplicative delta, dry days censored)",
        method_note="Quantile Delta Mapping (Cannon et al. 2015), "
                    "MULTIPLICATIVE form, dry days censored below "
                    f"{0.1} mm/day",
    )


def _temp_cfg(name):
    """Temperature: SEAS5 mn2t24/mx2t24 vs ERA5-Land 2m, both in K."""
    fc_var = {"tmin": "mn2t24", "tmax": "mx2t24"}[name]
    ob_var = "t2m"
    OUT_UNITS = "K"
    FC_DIR = Path(
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}"
        f"/{fc_var}/regridded_ERA5-Land"
    )
    OB_DIR = Path(f"/ec/res4/scratch/ecme4047/ERA5-Land/{ob_var}/daily")
    OUT_DIR = Path(
        f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_{INIT_MONTH}"
        f"/{fc_var}/calibrated_ERA5-Land"
    )
    stat = "minimum" if name == "tmin" else "maximum"
    return dict(
        name=name,
        fc_var=fc_var,
        ob_var=ob_var,
        method="additive",
        censor_dry=False,
        trace_mm=None,
        threshold_mm=None,
        units=OUT_UNITS,
        units_to_target="kelvin",   # K passes through, celsius -> +273.15
        fc_units_in="K",
        ob_units_in="K",
        ob_time_dim="valid_time",
        ob_lat="latitude",
        ob_lon="longitude",
        mask_to_country=True,
        ref_tag="ERA5-Land",
        fc_dir=FC_DIR,
        ob_dir=OB_DIR,
        out_dir=OUT_DIR,
        fc_path=lambda y, _d=FC_DIR, _v=fc_var: _d / (
            f"{_v}_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}"
            f"_res_ERA5-Land_patch.nc"
        ),
        ob_path=lambda y, yoff, month, _d=OB_DIR, _v=ob_var, _s=stat: _d / (
            f"_{y + yoff}".replace("_", "") + f"/{_v}_daily_{_s}_"
            f"{y + yoff}{month:02d}.nc"
        ),
        long_name=f"QDM-adjusted daily {name} (additive delta, all days corrected)",
        method_note="Quantile Delta Mapping (Cannon et al. 2015), ADDITIVE form",
    )


_VAR_CFG = {
    "pr":   _pr_cfg(),
    "tmax": _temp_cfg("tmax"),
    "tmin": _temp_cfg("tmin"),
}

# ---- WHICH VARIABLES TO RUN, IN ORDER ---------------------------------
# Any subset of ("pr", "tmax", "tmin"). The dask cluster is created once and
# reused; results do not depend on the order.
RUN_VARIABLES = ("pr", "tmax", "tmin")

for _v in RUN_VARIABLES:
    if _v not in _VAR_CFG:
        raise ValueError(f"unknown variable {_v!r}; known: {sorted(_VAR_CFG)}")

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



def to_target_units(arr, kind, cfg):
    """Bring a sample set into the working unit, per variable.

    "factor" : multiply (precipitation m -> mm).
    "kelvin" : add 273.15 when the file is in celsius, pass K through.
    """
    if cfg["units_to_target"] == "factor":
        factor = (cfg["fc_units_to_target"] if kind == "fc"
                  else cfg["ob_units_to_target"])
        return arr * factor
    units_in = cfg["fc_units_in"] if kind == "fc" else cfg["ob_units_in"]
    if str(units_in).lower() in ("c", "celsius"):
        return arr + C_TO_K
    return arr

def censor_dry_fn(arr, trace, rng=None):
    """Exact zeros -> nonzero uniform values in (0, trace)."""
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    zeros = out == 0.0
    n = int(zeros.sum())
    if n:
        out[zeros] = rng.uniform(0.0, trace, size=n)
    return out, n


def require_zero(arr, trace):
    """Below the trace threshold -> exact zero (dry days stay dry)."""
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    out[out < trace] = 0.0
    return out


# ======================================================================
# the transfer function -- the ONE place the methods differ
# ======================================================================
def qdm_transfer(m_pool, o_pool, target, method):
    """
    multiplicative (pr) :  y = x * ( xo(p) / xm(p) )
    additive       (tas):  y = x + ( xo(p) - xm(p) )

    with p = F_mod_train(x) the position of the target in the model training
    climatology, so the anomaly relative to the model survives the correction.

    `method` is an ARGUMENT, not a global: the delayed tasks run in worker
    processes, and a global read there would be whichever variable was
    configured last.

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

    if method == "multiplicative":
        # guard against a zero model quantile before dividing
        xm = np.where(xm <= 0.0, np.finfo(np.float64).tiny, xm)

    p = np.interp(t_arr, xm, q, left=QUANTILE_MIN, right=1.0 - QUANTILE_MIN)
    o_at_p = np.interp(p, q, xo)
    m_at_p = np.interp(p, q, xm)

    if method == "multiplicative":
        corrected = t_arr * (o_at_p / m_at_p)
    else:
        corrected = t_arr + (o_at_p - m_at_p)

    return corrected, p, o_at_p


def correct_cell(m_pool, o_pool, t_in, iy, ix, method, censor_dry, trace_mm):
    """
    Correct ONE grid cell -- the whole QDM for that point.

    m_pool : (ny_m, ntime_m, nmem) model pool, ALL years/members
    o_pool : (ny_o, ntime_o)       reference pool, ALL years
    t_in   : (ny, ntime, nmem)     target for this cell
    iy, ix : grid indices (passed through for provenance)
    method, censor_dry, trace_mm : the VARIABLE's settings, passed in rather
        than read from globals so a task cannot pick up another variable's.

    For censoring variables the arrays arrive already censored (zeros ->
    U(0, trace)) and the dry days are re-zeroed here. For the others nothing
    is censored and nothing is re-zeroed.
    """
    m_flat = np.asarray(m_pool, dtype=np.float64).ravel()
    o_flat = np.asarray(o_pool, dtype=np.float64).ravel()
    m_flat = m_flat[np.isfinite(m_flat)]
    o_flat = o_flat[np.isfinite(o_flat)]

    t_in = np.asarray(t_in, dtype=np.float64)

    # not enough sample in this cell
    if m_flat.size < 2 or o_flat.size < 2:
        if censor_dry:
            # return the target re-zeroed only, so the dry-day convention is
            # applied uniformly
            return require_zero(t_in, trace_mm).astype(np.float32)
        return t_in.astype(np.float32)

    out = np.empty_like(t_in, dtype=np.float64)
    for yi in range(t_in.shape[0]):
        tgt = t_in[yi]                       # (ntime, nmem)
        corrected, p, o_at_p = qdm_transfer(m_flat, o_flat, tgt.ravel(), method)

        if censor_dry:
            # zero where the OBSERVED value at that quantile is below the
            # trace: the position in the distribution is dry in the reference
            corrected[o_at_p <= trace_mm] = 0.0
            # zero where the CORRECTED value is below the trace
            corrected[corrected < trace_mm] = 0.0

        out[yi] = corrected.reshape(tgt.shape)

    return out.astype(np.float32)



# ======================================================================
# output attributes
# ======================================================================
def build_attrs(cfg, k, month, first_lead):
    """Output attributes, with the method and the dry-day convention recorded."""
    attrs = {
        "units": cfg["units"],
        "long_name": cfg["long_name"],
        "method": cfg["method_note"],
        "variable": cfg["name"],
        "lead_index": str(k),
        "target_month": f"{month:02d}",
        "skip_lead_days": str(SKIP_LEAD_DAYS),
        "members_kept": str(N_MEMBERS_KEEP),
        "n_quantiles": str(N_QUANTILES),
        "train": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
        "verify": f"{VERIF_YEARS[0]}-{VERIF_YEARS[-1]}",
        "reference_product": cfg["ref_tag"],
        "window_note": (
            f"model calibrated and corrected on days > {SKIP_LEAD_DAYS} "
            f"days after init; reference training pool spans the FULL "
            f"month {month:02d}"
            if first_lead else
            f"model and reference both span the full month {month:02d}"
        ),
    }
    if cfg["censor_dry"]:
        attrs["trace_mm"] = str(cfg["trace_mm"])
        attrs["rng_seed"] = str(RNG_SEED)
    return attrs



# ======================================================================
# one variable, all leads
# ======================================================================
def run_one_variable(var, cfg, client):
    """The whole calibration for ONE variable. Everything variable-specific
    comes from `cfg`; nothing is read from a module-level global."""
    fc_path = cfg["fc_path"]
    ob_path = cfg["ob_path"]
    FC_VAR = cfg["fc_var"]
    OB_VAR = cfg["ob_var"]
    OB_LAT, OB_LON = cfg["ob_lat"], cfg["ob_lon"]
    OB_TIME_DIM = cfg["ob_time_dim"]
    METHOD = cfg["method"]
    CENSOR_DRY = cfg["censor_dry"]
    TRACE_MM = cfg["trace_mm"]
    MASK_TO_COUNTRY = cfg["mask_to_country"]
    REF_TAG = cfg["ref_tag"]
    OUT_DIR = cfg["out_dir"]
    FC_DIR = cfg["fc_dir"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}", flush=True)
    print(f"# variable: {var}  method: {METHOD}  forecast var '{FC_VAR}'  "
          f"reference var '{OB_VAR}'", flush=True)
    print(f"{'#' * 70}", flush=True)

    train_fc_files = [fc_path(y) for y in TRAIN_YEARS if fc_path(y).exists()]
    verif_fc_files = [fc_path(y) for y in VERIF_YEARS if fc_path(y).exists()]
    print(f"forecast files: {len(train_fc_files)} train, "
          f"{len(verif_fc_files)} verif", flush=True)
    if not train_fc_files:
        print(f"no forecast files found in {FC_DIR} -- skipping {var}", flush=True)
        return

    for k, (month, yoff) in enumerate(TARGET):
        print(f"\n=== {var}: lead {k} -- target month {month:02d} "
              f"(year offset {yoff}) ===", flush=True)
        first_lead = (k == 0) and SKIP_LEAD_DAYS > 0

        # ---- 1. model pool: all years, all members, month, spin-up cut ----
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
            sub = xr.apply_ufunc(to_target_units, sub, "fc", cfg,
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
        m_da = m_da.clip(min=0.0) if CENSOR_DRY else m_da
        print(f"  model pool {dict(m_da.sizes)}", flush=True)

        # ---- 2. reference pool: same month, all years ---------------------
        # The reference file for init year y is read from year y+yoff: for a
        # lead that crosses into the next calendar year the December lead of an
        # October init is verified against December of y+1. That is the correct
        # pairing. The pool is labelled with the INIT years so it lines up with
        # m_da and t_da -- a positional label would drift wherever a file is
        # missing.
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
            ob = xr.apply_ufunc(to_target_units, ob, "ob", cfg,
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
        o_da = o_da.clip(min=0.0) if CENSOR_DRY else o_da
        print(f"  reference pool {dict(o_da.sizes)} "
              f"(init years {o_years[0]}-{o_years[-1]}, offset {yoff:+d} -> "
              f"files {o_years[0] + yoff}-{o_years[-1] + yoff})", flush=True)

        # ---- 3. target: verification years --------------------------------
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
            sub = xr.apply_ufunc(to_target_units, sub, "fc", cfg,
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
        # Label with the INIT years, not the position: a skipped year would
        # otherwise shift every later year's label silently.
        t_da = t_da.assign_coords(init_year=t_years)
        t_da = t_da.clip(min=0.0) if CENSOR_DRY else t_da
        print(f"  target     {dict(t_da.sizes)} "
              f"(init years {t_years[0]}-{t_years[-1]})", flush=True)

        # ---- 4. censoring (precipitation only) ----------------------------
        # Zeros -> U(0, trace) on all three sample sets, BEFORE any mapping.
        if CENSOR_DRY:
            def _censor_block(block, seed_off, _trace=TRACE_MM):
                rng = np.random.default_rng(RNG_SEED + seed_off)
                out, _ = censor_dry_fn(block, _trace, rng)
                return out

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

        # ---- 5. point-by-point QDM, in latitude strips --------------------
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
                    tasks.append(delayed(correct_cell)(
                        m_blk[..., iy, ix],
                        o_blk[..., iy, ix],
                        t_blk[..., iy, ix],
                        lat0 + iy, ix,
                        METHOD, CENSOR_DRY, TRACE_MM))
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

        # ---- 6. back to xarray and write ----------------------------------
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
            attrs=build_attrs(cfg, k, month, first_lead),
        )
        out_path = OUT_DIR / (
            f"{FC_VAR}_24h_mon{month:02d}_init{INIT_MONTH:02d}_"
            f"{t_years[0]}-{t_years[-1]}_{REGION}_"
            f"res_ERA5-Land_qdm_{REF_TAG}.nc"
        )
        out_ds = corrected.to_dataset()
        out_ds.to_netcdf(out_path, encoding={
            v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars
        })
        print(f"  wrote {out_path}", flush=True)

    print(f"{var}: done", flush=True)


# ======================================================================
# main
# ======================================================================
def main():
    LOCAL_SCRATCH = os.environ.get("TMPDIR")
    if not LOCAL_SCRATCH or not Path(LOCAL_SCRATCH).is_dir():
        LOCAL_SCRATCH = tempfile.gettempdir()
    print(f"dask local_directory: {LOCAL_SCRATCH}", flush=True)

    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        memory_limit=WORKER_MEM,
        local_directory=LOCAL_SCRATCH,
        dashboard_address=None,
    )
    client = Client(cluster)
    print(f"dask: {client}", flush=True)
    print(f"running variables: {list(RUN_VARIABLES)}", flush=True)

    try:
        for var in RUN_VARIABLES:
            # One variable at a time: free the previous variable's futures
            # before the next one loads its pools.
            client.cancel(client.futures, force=True)
            run_one_variable(var, _VAR_CFG[var], client)
        print("\nall variables done", flush=True)

    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()


