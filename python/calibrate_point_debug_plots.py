#!/usr/bin/env python3
"""
QDM bias adjustment of ECMWF SEAS5 daily precipitation (24h tp) vs MSWEP,
FOR A SINGLE GRID POINT -- reduced version for debugging, with plots.

Method:
  * training 1993-2022, verification 2023-2025;
  * one transfer function per (lead = target month), trained on that month only;
  * multiplicative QDM: delta from the training climatologies, re-applied to the
    target value, so the anomaly survives;
  * dry days handled as CENSORED values below a trace threshold of 0.05 mm/day:
      - zeros in BOTH model and observations are replaced by nonzero uniform
        random values below the trace threshold, BEFORE the correction, so the
        discrete mass at zero becomes part of a continuous distribution;
      - after the correction, values below the trace threshold are set back to
        zero, so days that were dry stay dry.

THRESHOLDS:
  TRACE_MM     = 0.05   censoring threshold: where the uniform noise is drawn and
                        where the final re-zeroing happens. The only threshold
                        that ACTS on the data.
  THRESHOLD_MM = 1.0    DIAGNOSTIC ONLY -- never filters the CDF pool; used to
                        report the wet-day fraction and recorded in the output.

PLOTS (matplotlib, saved as PNG next to the NetCDF):
  qdm_ratio_monMM_leadK.png   ratio obs/mod per quantile, with the 1:1 line
  qdm_dist_monMM_leadK.png    distributions: model train, obs train,
                              raw target, corrected target

Run:
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00 \
         python qdm_point_debug.py
"""

from pathlib import Path
import os
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")          # no display on a compute node
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
FC_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/regridded_ERA5-Land")
OB_DIR = Path("/ec/res4/scratch/ecme4047/MSWEP/MSWEP_V316_test/Past/Daily/regridded_ERA5-Land")
OUT_DIR = Path("/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/24h/init_10/tp/calibrated_MSWEP")
FIG_DIR = Path("/ec/res4/scratch/ecme4047/figures/model_calibration/")

REGION      = "Madagascar"
TRAIN_YEARS = range(1993, 2022 + 1)
VERIF_YEARS = range(2023, 2025 + 1)
INIT_MONTH  = 10
N_MONTHS = 3

# ---- THE POINT TO DEBUG: edit these two ----
LAT_POINT = -18.1          # e.g. Tamatave area
LON_POINT = 49.1

FC_VAR = "tp"
OB_VAR = "precipitation"

FC_UNITS_TO_MM = 1000.0    # tp in metres (24 h accumulation) -> mm/day
OB_UNITS_TO_MM = 1.0       # MSWEP already mm/day

TRACE_MM     = 0.05        # censoring / trace threshold (mm/day)
THRESHOLD_MM = 1.0         # DIAGNOSTIC ONLY -- never filters the CDF pool
N_MEMBERS_KEEP = 25        # members common to all years (verify!)

MEMBER_DIM      = "number"
FC_TIME_DIM     = "forecast_period"
FC_SELECT_COORD = "valid_time"
FC_CONCAT_DIM   = "forecast_reference_time"
OB_TIME_DIM     = "time"

N_QUANTILES  = 100
QUANTILE_MIN = 0.01

RNG_SEED = 12345           # reproducible uniform noise for the censoring

SKIP_LEAD_DAYS = 10        # drop the first N days after init from the CALIBRATION
                           # (forecast spin-up). Applied to the FORECAST only --
                           # the observation has no lead. The lead counts from the
                           # initialization date, so the filter removes the first
                           # days of the whole window (October, with an Oct init),
                           # not the first days of every month.

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
    return FC_DIR / f"tp_24h_ecmwf51_init{y}{INIT_MONTH:02d}_{REGION}_res_ERA5-Land_patch.nc"


def ob_path(y, yoff, month):
    return OB_DIR / f"precip_daily_{y + yoff}{month:02d}_{REGION}_res_ERA5-Land.nc"


def lead_keep_mask(da, month, init_year, month_boundary_day=1):
    """
    Boolean mask selecting the forecast steps to KEEP for a given calendar month.

    Two conditions:
      * the step's valid_time falls in `month`;
      * its lead is at least SKIP_LEAD_DAYS days after the initialization date.

    The lead is measured from the INITIALIZATION DATE, not from the start of the
    month, so with SKIP_LEAD_DAYS = 10 only the first days of the window are
    removed (October for an October init); November, December and January are
    untouched. If SKIP_LEAD_DAYS is 0 the lead condition is satisfied by every
    step and the mask reduces to the plain month selection.

    `init_year` is the year of the initialization, from the file name; the
    cutoff date is built from it and INIT_MONTH.
    """
    in_month = da[FC_SELECT_COORD].dt.month == month
    if not SKIP_LEAD_DAYS:
        return in_month
    cutoff = (np.datetime64(f"{int(init_year):04d}-{INIT_MONTH:02d}-"
                            f"{month_boundary_day:02d}")
              + np.timedelta64(SKIP_LEAD_DAYS, "D"))
    return in_month & (da[FC_SELECT_COORD] >= cutoff)


def extract_point(ds, var, lat, lon, spatial_lat="lat", spatial_lon="lon"):
    """Nearest-grid-point 1-D extraction, all singleton dims dropped."""
    da = ds[var].sel({spatial_lat: lat, spatial_lon: lon}, method="nearest")
    return da.squeeze(drop=True)


def censor_dry(arr, trace=TRACE_MM, rng=None):
    """
    Replace EXACT ZEROS with nonzero uniform random values in (0, trace).

    Daily precipitation has a discrete mass at zero on top of a continuous
    distribution. Feeding the zeros as zeros makes the empirical CDF degenerate
    on the first quantiles and the QDM ratio ill-defined. Spreading them as a
    small uniform mass just below the trace threshold keeps them OUT of the wet
    tail while making the distribution continuous and invertible.

    Values strictly between 0 and `trace` are left as they are.
    """
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    zeros = out == 0.0
    n = int(zeros.sum())
    if n:
        out[zeros] = rng.uniform(0.0, trace, size=n)
    return out, n


def require_zero(arr, trace=TRACE_MM):
    """Post-correction: anything below the trace threshold becomes an exact zero,
    so days that were dry stay dry and the discrete mass at zero is restored."""
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    out[out < trace] = 0.0
    return out


def qdm_point(mod_train, obs_train, mod_target,
              nq=N_QUANTILES, qmin=QUANTILE_MIN):
    """
    Multiplicative QDM on three ALL-DAYS 1-D arrays (mm/day, already censored).

    The pool is NOT filtered to wet days: with censoring every day belongs to
    the distribution -- dry days sit just below the trace threshold -- and that
    is what carries the dry-day frequency information.
    """
    mod_train = np.asarray(mod_train, dtype=np.float64)
    obs_train = np.asarray(obs_train, dtype=np.float64)
    mod_target = np.asarray(mod_target, dtype=np.float64)

    mod_train = mod_train[np.isfinite(mod_train)]
    obs_train = obs_train[np.isfinite(obs_train)]

    print(f"    qdm_point: mod_train n={mod_train.size} "
          f"obs_train n={obs_train.size} target n={mod_target.size}", flush=True)

    if mod_train.size < 2 or obs_train.size < 2 or mod_target.size == 0:
        print("    -> insufficient sample: returning target unchanged", flush=True)
        return mod_target

    q = np.linspace(qmin, 1.0 - qmin, nq)

    xm = np.maximum.accumulate(np.quantile(mod_train, q))
    xo = np.maximum.accumulate(np.quantile(obs_train, q))

    # guard: a zero model quantile would divide by zero in the ratio
    xm = np.where(xm <= 0.0, np.finfo(np.float64).tiny, xm)

    p = np.interp(mod_target, xm, q, left=qmin, right=1.0 - qmin)
    obs_at_p = np.interp(p, q, xo)
    mod_at_p = np.interp(p, q, xm)

    ratio = obs_at_p / mod_at_p
    out = mod_target * ratio

    print(f"    -> mod {mod_train.min():.3f}..{mod_train.max():.2f}  "
          f"obs {obs_train.min():.3f}..{obs_train.max():.2f}  "
          f"ratio {ratio.min():.3f}..{ratio.max():.3f}  "
          f"out {out.min():.3f}..{out.max():.2f}", flush=True)
    return out


def make_plots(month, lead, m_train, o_train, raw, corrected):
    """
    Two demonstrative plots for the single-point debug, written as PNG
    into OUT_DIR:

      1. QDM ratio (obs/mod) per quantile, with the 1:1 reference line;
      2. precipitation distributions: model train, obs train, raw target,
         corrected target as frequency polygons on a shared axis.

    Ratios and quantiles are computed from the ACTUAL arrays used in the
    correction, so the figure shows the real transfer function.
    """
    q = np.linspace(QUANTILE_MIN, 1.0 - QUANTILE_MIN, N_QUANTILES)

    xm = np.maximum.accumulate(np.quantile(m_train, q))
    xo = np.maximum.accumulate(np.quantile(o_train, q))
    xm_safe = np.where(xm <= 0.0, np.finfo(np.float64).tiny, xm)
    ratio = xo / xm_safe

    # ---- plot 1: ratio per quantile ------------------------------------
    # The dry region of the OBSERVED distribution: quantiles where the observed
    # value is still at/below the trace threshold. Beyond p_dry the observed
    # distribution is genuinely wet, and the model's ratio there is meaningful.
    dry_mask = xo <= TRACE_MM
    p_dry = float(q[dry_mask][-1]) if dry_mask.any() else float(q[0])

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=140)
    ax.plot(q, ratio, color="#2b6cb0", lw=2, label="obs / mod")
    ax.axhline(1.0, color="#718096", ls="--", lw=1.2, label="1:1")

    # highlight the region where the observed quantile is below the trace
    ax.axvspan(q[0], p_dry, color="#e2e8f0", alpha=0.7, zorder=0)
    ax.axvline(p_dry, color="#a0aec0", lw=1.2, ls=":")
    ax.annotate(
        f"obs <= {TRACE_MM} mm\n(dry region)",
        xy=(0.5 * (q[0] + p_dry), 0.92), xycoords=("data", "axes fraction"),
        ha="center", va="top", fontsize=8, color="#4a5568",
    )

    ax.set_xlabel("Quantile (-)")
    ax.set_ylabel("Ratio obs/mod (-)")
    ax.set_title(f"QDM ratio obs/mod per quantile - lead {lead}, month {month:02d}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    # second x axis: observed value at a set of sampled quantiles
    n_ticks = 11
    tick_pos = np.linspace(q[0], q[-1], n_ticks)
    tick_lab = [f"{np.interp(t, q, xo):.2f}" for t in tick_pos]
    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(tick_pos)
    secax.set_xticklabels(tick_lab, fontsize=7, rotation=45)
    secax.set_xlabel("obs value at that quantile (mm/day)", fontsize=9)
    secax.spines["bottom"].set_position(("outward", 34))

    fig.tight_layout()
    f1 = FIG_DIR / f"qdm_tr{TRACE_MM}mm_{N_QUANTILES}qtls_ratio_mon{month:02d}_init{INIT_MONTH:02d}_{VERIF_YEARS[0]}-{VERIF_YEARS[1]}_lat{LAT_POINT}_lon{LON_POINT}.png"
    fig.savefig(f1, bbox_inches="tight")
    plt.close(fig)

    # ---- plot 2: distributions -----------------------------------------
    all_vals = np.concatenate([m_train, o_train, raw, corrected])
    hi = float(np.percentile(all_vals, 99.5)) or 1.0
    edges = np.linspace(0.0, hi, 41)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def density(a):
        h, _ = np.histogram(a, bins=edges)
        return h / max(a.size, 1) * 100.0

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=140)
    for arr, lbl, col in [
        (m_train,   "model (train)",    "#2b6cb0"),
        (o_train,   "obs (train)",      "#2f855a"),
        (raw,       "raw target",       "#b7791f"),
        (corrected, "corrected target", "#c53030"),
    ]:
        ax.plot(centers, density(arr), color=col, lw=1.8, label=lbl)
    ax.set_xlabel("Precipitation (mm/day)")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(f"Daily precipitation distribution - lead {lead}, month {month:02d}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    f2 = FIG_DIR / f"qdm_tr{TRACE_MM}mm_{N_QUANTILES}qtls_distr_mon{month:02d}_init{INIT_MONTH:02d}_lat{LAT_POINT}_lon{LON_POINT}.png"
    fig.savefig(f2)
    plt.close(fig)

    return [f1, f2]


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    print(f"single-point QDM debug  lat={LAT_POINT}  lon={LON_POINT}  "
          f"trace={TRACE_MM} mm", flush=True)
    rng = np.random.default_rng(RNG_SEED)

    for k, (month, yoff) in enumerate(TARGET):
        print(f"\n=== lead {k} -- target month {month:02d} "
              f"(year offset {yoff}) ===", flush=True)

        # ---- 1. model training pool: all days, all years, all members ----
        m_train = []
        for y in TRAIN_YEARS:
            f = fc_path(y)
            if not f.exists():
                continue
            ds = xr.open_dataset(f)
            da = extract_point(ds, FC_VAR, LAT_POINT, LON_POINT)
            keep = lead_keep_mask(da, month, _init_year(f))
            da = da.sel({FC_TIME_DIM: keep}, drop=True)
            if da.sizes.get(FC_TIME_DIM, 0) == 0:
                ds.close()
                continue
            da = da.isel({MEMBER_DIM: slice(0, N_MEMBERS_KEEP)})
            vals = np.clip(da.values * FC_UNITS_TO_MM, 0.0, None)  # kill patch negatives
            m_train.append(vals.ravel())
            ds.close()
        m_train = np.concatenate(m_train) if m_train else np.array([])

        # ---- 2. obs training pool: all days ------------------------------
        o_train = []
        for y in TRAIN_YEARS:
            f = ob_path(y, yoff, month)
            if not f.exists():
                continue
            ds = xr.open_dataset(f)
            da = extract_point(ds, OB_VAR, LAT_POINT, LON_POINT)
            vals = np.clip(da.values * OB_UNITS_TO_MM, 0.0, None)
            o_train.append(vals.ravel())
            ds.close()
        o_train = np.concatenate(o_train) if o_train else np.array([])

        # ---- 3. target: verification years -------------------------------
        tgt = []
        for y in VERIF_YEARS:
            f = fc_path(y)
            if not f.exists():
                continue
            ds = xr.open_dataset(f)
            da = extract_point(ds, FC_VAR, LAT_POINT, LON_POINT)
            keep = lead_keep_mask(da, month, _init_year(f))
            da = da.sel({FC_TIME_DIM: keep}, drop=True)
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

        # ---- 4. CENSORING: zeros -> uniform(0, trace) ---------------------
        # Applied to the training pools AND to the target, before any mapping.
        m_train, n_z_m = censor_dry(m_train, TRACE_MM, rng)
        o_train, n_z_o = censor_dry(o_train, TRACE_MM, rng)
        n_z_t = 0
        tgt_censored = []
        for arr in tgt:
            flat = arr.ravel()
            flat_c, nz = censor_dry(flat, TRACE_MM, rng)
            n_z_t += nz
            tgt_censored.append(flat_c)

        print(f"  censored zeros (-> U(0,{TRACE_MM})): "
              f"model={n_z_m}/{m_train.size}  obs={n_z_o}/{o_train.size}  "
              f"target={n_z_t}", flush=True)

        # ---- 5. diagnostics on the censored pools ------------------------
        print(f"  wet-day fraction (>{THRESHOLD_MM} mm): "
              f"model={100*(m_train > THRESHOLD_MM).mean():.1f}%  "
              f"obs={100*(o_train > THRESHOLD_MM).mean():.1f}%", flush=True)

        # NOTE on the comparison for the first lead:
        # the MODEL is calibrated and corrected on the days AFTER the first
        # SKIP_LEAD_DAYS days of the window, while the MSWEP training pool is
        # built over the WHOLE calendar month. For the first lead these two
        # windows differ, so the printed obs statistics describe a slightly
        # wider sample than the one the model was matched against. Only the
        # first lead is affected; the later leads have no spin-up cut.
        first_lead = (k == 0) and SKIP_LEAD_DAYS > 0
        if first_lead:
            print(f"  [note] lead 0: model calibrated on days > day "
                  f"{SKIP_LEAD_DAYS} after init; MSWEP pool spans the FULL "
                  f"month {month:02d} -- windows are not identical for this lead",
                  flush=True)

        # ---- 6. QDM on ALL days ------------------------------------------
        raw = np.concatenate([a.ravel() for a in tgt])
        cens = np.concatenate(tgt_censored)
        corrected_cens = qdm_point(m_train, o_train, cens)

        # ---- 7. re-zero below the trace threshold ------------------------
        corrected = require_zero(corrected_cens, TRACE_MM)

        # ---- 8. summary ---------------------------------------------------
        print(f"  target raw : mean={raw.mean():.2f} max={raw.max():.2f} "
              f"dry_days={100*(raw == 0).mean():.1f}%", flush=True)
        print(f"  corrected  : mean={corrected.mean():.2f} "
              f"max={corrected.max():.2f} "
              f"dry_days={100*(corrected == 0).mean():.1f}%", flush=True)
        print(f"  obs (train): mean={o_train.mean():.2f} "
              f"dry_days={100*(o_train < TRACE_MM).mean():.1f}%", flush=True)

        assert corrected.ndim == 1 and raw.ndim == 1, (corrected.shape, raw.shape)

        # ---- 9. write per-point output ------------------------------------
        out_ds = xr.Dataset(
            {
                "tp_raw": ("sample", raw.astype("float32")),
                "tp_qdm": ("sample", corrected.astype("float32")),
            },
            coords={"sample": np.arange(raw.size)},
            attrs={
                "lat": str(LAT_POINT), "lon": str(LON_POINT),
                "month": str(month), "lead": str(k),
                "trace_mm": str(TRACE_MM),
                "threshold_mm": str(THRESHOLD_MM),
                "dry_day_treatment": "censored below trace; zeros -> U(0,trace) "
                                     "before correction, re-zeroed after",
                "train": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
                "members_kept": str(N_MEMBERS_KEEP),
                "skip_lead_days": str(SKIP_LEAD_DAYS),
                "window_note": (
                    f"model trained and corrected on days > {SKIP_LEAD_DAYS} "
                    f"after init; MSWEP training pool over the FULL month {month:02d}"
                    if first_lead else
                    f"model and MSWEP span the FULL month {month:02d}"
                ),
            },
        )
        p = OUT_DIR / f"qdm_tr{TRACE_MM}mm_{N_QUANTILES}qtls_point_mon{month:02d}_init{INIT_MONTH:02d}_{VERIF_YEARS[0]}-{VERIF_YEARS[1]}_lat{LAT_POINT}_lon{LON_POINT}.nc"
        out_ds.to_netcdf(p)
        print(f"  wrote {p}", flush=True)

        # ---- 10. demonstrative plots -------------------------------------
        # plotting must not kill the run if matplotlib misbehaves
        try:
            for fp in make_plots(month, k, m_train, o_train, raw, corrected):
                print(f"  plot -> {fp}", flush=True)
        except Exception as exc:
            print(f"  plot skipped: {exc}", flush=True)

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()

