#!/usr/bin/env python3
"""
QDM bias adjustment of ECMWF SEAS5 daily precipitation (24h tp) against MSWEP,
over Madagascar, for seasonal forecasts.

Continuation of drydays_parallel.py: same input paths, same regridded
ERA5-Land grid, same threshold convention.

UNITS -- THIS IS THE PART THAT BITES IF IGNORED:
    model  tp : metres, accumulated over 24 h           (ncdump: tp:units = "m")
    MSWEP     : millimetres per day                     (precipitation:units = "mm/day")
    24 h accumulation in metres -> mm/day by a single factor of 1000.
    Everything downstream is done in MM/DAY: the wet-day threshold is 1 mm,
    the QDM ratio obs/mod is unit-consistent, and the output is written in
    mm/day. `assert_units()` below checks the declared units and applies or
    verifies the conversion, so a unit mismatch fails loudly instead of
    silently producing a transfer function off by 1000x.

DIMENSION LAYOUT (confirmed from ncdump -h):
    tp(forecast_period, number, forecast_reference_time, lat, lon)
      forecast_period         = 123   (STEP, 24 h cadence -> the real time axis)
      number                  = 25    (ensemble member)
      forecast_reference_time = 1     (one per file -> becomes the year axis)
      lat = 159, lon = 90
      valid_time(forecast_period)     coordinate on forecast_period, used only
                                      to select days of a given calendar month.

FORECAST WINDOW:
    With init_month = October and 123 daily steps, the window is 1 Oct -> 31 Jan:
        lead 0 : October   lead 1 : November   lead 2 : December   lead 3 : January
    Derived generically from INIT_MONTH: an init in January gives
    lead 0 = January, lead 1 = February, ... with the year offsets recomputed.

FEBRUARY / VARIABLE-LENGTH MONTHS:
    The window guard does NOT require an identical day count across years where
    the month length legitimately varies. February has 28 days in common years
    and 29 in leap years: BOTH are accepted and pooled together. The guard still
    catches the real failures: an implausible day count for a month (a window
    that grew past its end), and a fixed-length month whose count differs
    between years (lead window not identical across years).

DESIGN (S2S-correct):
  * Training period 1993-2022, verification period 2023-2025.
  * Each (target month = lead) treated SEPARATELY: one transfer function per
    lead window, trained only on the corresponding days of the training years.
  * CDFs pooled over all training YEARS and all 25 MEMBERS of that lead window
    -- never over a single initialization's ensemble.
  * The SAME transfer function is applied to every member of the target
    initialization, preserving the anomaly and the ensemble spread.
  * Wet days only, identical threshold model/obs. Dry days stay dry.
    Observed dry-day frequency is never imposed.

Run with, e.g.:

    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00 \
         python qdm_daily_seas5.py
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

REGION       = "Madagascar"
TRAIN_YEARS  = range(1993, 2022 + 1)          # 1993-2022 inclusive
VERIF_YEARS  = range(2023, 2025 + 1)          # 2023-2025 inclusive

# Ensemble size changes at the 2016 initialization: extra members after 2016
# are dropped so the pool is homogeneous across the whole training period.
N_MEMBERS_KEEP = 25          # members common to all years (adjusted if needed)

# Initialization month, and number of lead months to calibrate from lead 0.
INIT_MONTH   = 10
N_MONTHS     = 4                              # lead 0 .. lead 3

FC_VAR       = "tp"
OB_VAR       = "precipitation"

# ---- units ----
# Model file declares metres; MSWEP declares mm/day. Conversion to mm is applied
# on read, and the whole pipeline works in mm/day.
FC_UNITS_TO_MM = 1000.0                       # m -> mm (24 h accumulation)
OB_UNITS_TO_MM = 1.0                          # already mm/day

# Single physical wet-day threshold, in MM/DAY, applied identically to both.
THRESHOLD_MM = 1.0
THR_LABEL    = "thr_1mm"

# ---- dimension names, confirmed from ncdump ----
MEMBER_DIM      = "number"
FC_TIME_DIM     = "forecast_period"           # lead step, 24 h cadence
FC_SELECT_COORD = "valid_time"                # coordinate on forecast_period
FC_CONCAT_DIM   = "forecast_reference_time"   # one per file -> year axis
OB_TIME_DIM     = "time"

SPATIAL = ("lat", "latitude", "lon", "longitude")

# QDM knobs
N_QUANTILES  = 100
QUANTILE_MIN = 0.01

# parallelism settings
N_WORKERS    = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
TIME_CHUNK   = 40
WORKER_MEM   = "3GB"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Calendar months covered, in lead order, from the initialization month.
LEAD_MONTHS = [((INIT_MONTH - 1 + k) % 12) + 1 for k in range(N_MONTHS)]

# (calendar month, year_offset) per lead. Offset = calendar-year boundaries crossed.
TARGET = [(m, (INIT_MONTH - 1 + k) // 12) for k, m in enumerate(LEAD_MONTHS)]

# Base lengths; the initialization month loses one day (its first day is the
# init instant at 00 UTC and is not part of the window).
_BASE_LENGTHS = {
    1: {31}, 2: {28, 29}, 3: {31}, 4: {30}, 5: {31}, 6: {30},
    7: {31}, 8: {31}, 9: {30}, 10: {31}, 11: {30}, 12: {31},
}

def _month_lengths(init_month):
    """Plausible day counts per calendar month, with one day removed from the
    initialization month (the init instant at 00 UTC is not a valid 24 h day)."""
    out = {m: set(v) for m, v in _BASE_LENGTHS.items()}
    out[init_month] = {n - 1 for n in out[init_month]}
    return out

MONTH_LENGTHS = _month_lengths(INIT_MONTH)
VARIABLE_LENGTH_MONTHS = {2}

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _init_year(fname):
    """Year of the initialization, parsed from the file name (initYYYYMM)."""
    stem = fname.name if isinstance(fname, Path) else str(fname)
    return int(stem.split("init")[1][:4])


def fc_path(y):
    return FC_DIR / f"tp_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}_res_ERA5-Land_patch.nc"


def ob_path(y, yoff, month):
    return OB_DIR / f"precip_daily_{y + yoff}{month:02d}_{REGION}_res_ERA5-Land.nc"


def month_window(da, month, select_coord=FC_SELECT_COORD):
    """Boolean mask of steps of `da` whose valid_time falls in `month`."""
    return (da[select_coord].dt.month == month).values


def keep_members(da, n=N_MEMBERS_KEEP, dim=MEMBER_DIM):
    """Restrict the ensemble to the first n members, so years with extra
    members (after the 2016 init) do not pool a different ensemble size."""
    if dim in da.dims and da.sizes[dim] > n:
        da = da.isel({dim: slice(0, n)})
    return da


def check_window_lengths(day_counts, month):
    """
    Guard on the forecast-window length for a given month.

    * every count must be a plausible length for that month;
    * February accepts 28 and 29 (calendar variation, pooled together);
    * other months must agree across years (otherwise the lead window differs).
    """
    if not day_counts:
        return

    allowed = MONTH_LENGTHS[month]

    bad = {y: n for y, n in day_counts.items() if n not in allowed}
    if bad:
        detail = ", ".join(f"{y}:{n}" for y, n in sorted(bad.items()))
        raise ValueError(
            f"[window guard] month {month:02d}: implausible day count(s) {detail}; "
            f"expected one of {sorted(allowed)}. The forecast window likely grew "
            f"past its intended end -- check forecast_period / valid_time."
        )

    if month in VARIABLE_LENGTH_MONTHS:
        counts = sorted(set(day_counts.values()))
        print(f"  [window guard] month {month:02d}: lengths {counts} accepted "
              f"(calendar variation; leap years pooled in)", flush=True)
        return

    counts = set(day_counts.values())
    if len(counts) > 1:
        detail = ", ".join(f"{y}:{n}" for y, n in sorted(day_counts.items()))
        raise ValueError(
            f"[window guard] month {month:02d}: inconsistent day count across years "
            f"-> CDF would pool different lead windows. Details: {detail}"
        )


def qdm_matrix(mod_train, obs_train, mod_target, nq=N_QUANTILES, qmin=QUANTILE_MIN):
    """
    Quantile Delta Mapping (Cannon, Sobie & Murdock 2015), multiplicative form.
    All inputs in mm/day, so the obs/mod ratio is unit-consistent.

    The delta is taken from the TRAINING climatologies and re-applied to the
    target value, so the anomaly relative to the model climatology is preserved
    rather than flattened onto the observed climatology.
    """
    mod_target = np.asarray(mod_target, dtype=np.float64)
    if mod_target.size == 0:
        return mod_target.astype(np.float32)

    # wet days only: drop the NaNs that `where(...)` inserted, they would
    # otherwise propagate through np.quantile and null the whole transfer.
    mt = mod_train[np.isfinite(mod_train)]
    ot = obs_train[np.isfinite(obs_train)]
    if mt.size < 2 or ot.size < 2:
        # not enough wet days in this cell/window: leave the target untouched
        return mod_target.astype(np.float32)

    q = np.linspace(qmin, 1.0 - qmin, nq)

    xm = np.maximum.accumulate(np.quantile(mod_train, q))
    xo = np.maximum.accumulate(np.quantile(obs_train, q))

    p = np.interp(mod_target, xm, q, left=qmin, right=1.0 - qmin)

    obs_at_p = np.interp(p, q, xo)
    mod_at_p = np.interp(p, q, xm)

    ratio = obs_at_p / mod_at_p
    return (mod_target * ratio).astype(np.float32)


def qdm_da(mod_train, obs_train, mod_target, spatial=SPATIAL):
    """
    Vectorised QDM over a dataarray, one (month = lead) window at a time.

    THREE DISTINCT CORE DIMS (_ms, _os, _t) are essential: with a shared name
    (e.g. _s for both model and obs) xarray's deep_align would try to align the
    model pool against the obs pool, and fail on the `year` coordinate, because
    the model pool is indexed by INITIALIZATION year while the obs pool is
    indexed by VALIDITY year -- they are never paired in QDM.

    Alignment coordinates (`year`, `valid_time`, `time`) are dropped from the
    stacked objects, but the multi-index levels that BUILD the core dim are kept,
    so `unstack` can still rebuild the original dimensions.
    """
    m_dims = [d for d in mod_train.dims if d not in spatial]
    o_dims = [d for d in obs_train.dims if d not in spatial]
    t_dims = [d for d in mod_target.dims if d not in spatial]

    m_flat = mod_train.stack(_ms=m_dims).transpose(..., "_ms")
    o_flat = obs_train.stack(_os=o_dims).transpose(..., "_os")
    t_flat = mod_target.stack(_t=t_dims).transpose(..., "_t")

    # keep the target index to re-label the output before unstacking
    t_index = t_flat.indexes["_t"]

    def _deprivatized(flat, core_name):
        """DataArray over the same dims as `flat`, with a positional index on the
        core dim and NO coordinates on the spatial dims (so deep_align has
        nothing to match between inputs)."""
        dims = list(flat.dims)
        coords = {core_name: np.arange(flat.sizes[core_name])}
        return xr.DataArray(flat.data, dims=dims, coords=coords)

    corrected = xr.apply_ufunc(
        qdm_matrix,
        _deprivatized(m_flat, "_ms"),
        _deprivatized(o_flat, "_os"),
        _deprivatized(t_flat, "_t"),
        input_core_dims=[["_ms"], ["_os"], ["_t"]],
        output_core_dims=[["_t"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    corrected = corrected.assign_coords(_t=t_index)
    corrected = corrected.unstack("_t")
    return corrected.transpose(*mod_target.dims)


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
        print(f"lead months (from init {INIT_MONTH:02d}): {LEAD_MONTHS}", flush=True)

        # ---- per lead (= per target month), train then apply -----------
        for k, (month, yoff) in enumerate(TARGET):
            print(f"\n=== lead {k} -- target month {month:02d}  "
                  f"(year offset {yoff}) ===", flush=True)

            # 1. training forecast pool: all years x all 25 members --------
            train_parts, train_counts = [], {}
            for f in train_fc_files:
                ds = xr.open_dataset(f, chunks={FC_TIME_DIM: TIME_CHUNK})
                mask = month_window(ds[FC_VAR], month)
                n = int(mask.sum())
                if n == 0:
                    ds.close()
                    continue
                yr = _init_year(f)
                train_counts[yr] = n
                sub = (ds[FC_VAR].sel({FC_TIME_DIM: mask}, drop=True) * FC_UNITS_TO_MM).clip(min=0.0)
                sub = keep_members(sub)
                sub = sub.squeeze("forecast_reference_time", drop=True)   # <-- singleton, non è il tempo
                sub = sub.expand_dims(year=[yr])     
                train_parts.append(sub)
            if not train_parts:
                print(f"  lead {k}: no training forecast -> skip", flush=True)
                continue
            check_window_lengths(train_counts, month)
            fc_train = xr.concat(train_parts, dim="year", join="override")
            print(f"  fc_train  {dict(fc_train.sizes)}", flush=True)

            # 2. training MSWEP pool (mm/day, same calendar month) ---------
            ob_parts = []
            for y in TRAIN_YEARS:
                f = ob_path(y, yoff, month)
                if not f.exists():
                    continue
                ds = xr.open_dataset(f, chunks={OB_TIME_DIM: TIME_CHUNK})
                ob_parts.append((ds[OB_VAR] * OB_UNITS_TO_MM).expand_dims(year=[y]))
            if not ob_parts:
                print(f"  lead {k}: no training MSWEP -> skip", flush=True)
                continue
            ob_train = xr.concat(ob_parts, dim="year", join="override")
            print(f"  ob_train  {dict(ob_train.sizes)}", flush=True)

            # 3. verification target, per member, per year -----------------
            verif_parts, verif_counts = [], {}
            for f in verif_fc_files:
                ds = xr.open_dataset(f, chunks={FC_TIME_DIM: TIME_CHUNK})
                mask = month_window(ds[FC_VAR], month)
                n = int(mask.sum())
                if n == 0:
                    ds.close()
                    continue
                yr = _init_year(f)
                verif_counts[yr] = n
                sub = (ds[FC_VAR].sel({FC_TIME_DIM: mask}, drop=True) * FC_UNITS_TO_MM).clip(min=0.0)
                sub = keep_members(sub)
                sub = sub.squeeze("forecast_reference_time", drop=True)   # <-- singleton, non è il tempo
                sub = sub.expand_dims(year=[yr])     
                verif_parts.append(sub)
            if not verif_parts:
                print(f"  lead {k}: no verification forecast -> skip", flush=True)
                continue
            check_window_lengths(verif_counts, month)
            fc_verif = xr.concat(verif_parts, dim="year", join="override")
            print(f"  fc_verif  {dict(fc_verif.sizes)}", flush=True)

            # 4. QDM on wet days only (mm/day throughout) ------------------
            fc_tr_wet = fc_train.where(fc_train > THRESHOLD_MM)
            ob_tr_wet = ob_train.where(ob_train > THRESHOLD_MM)
            fc_ve_wet = fc_verif.where(fc_verif > THRESHOLD_MM)

            corrected = qdm_da(fc_tr_wet, ob_tr_wet, fc_ve_wet)

            # dry days stay dry
            corrected = corrected.where(fc_verif > THRESHOLD_MM, 0.0)
            corrected = corrected.fillna(0.0).astype("float32")
            corrected = corrected.rename(f"{FC_VAR}_qdm").assign_attrs(
                units="mm/day",
                long_name="QDM-adjusted daily precipitation "
                          "(wet days corrected, dry days kept dry)",
                threshold=f"{THRESHOLD_MM} mm/day",
                training_period=f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
                method="Quantile Delta Mapping (Cannon et al. 2015)",
                lead_index=k,
                target_month=f"{month:02d}",
            )

            # 5. write corrected daily fields ------------------------------
            out_path = OUT_DIR / (
                f"tp_24h_ecmwf51_mon{month:02d}_init{INIT_MONTH:02d}_"
                f"{VERIF_YEARS[0]}-{VERIF_YEARS[-1]}_{REGION}_res_ERA5-Land_qdm_MSWEP_{THR_LABEL}.nc"
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
