"""
Probabilistic skill metrics for a seasonal SPEI forecast.

Separate from the deterministic metrics so the deterministic scores can be
produced (and re-read) without any probabilistic work, and vice versa.

Design (as described by the user, not inferred from the headers):
  forecast : dims (number=25, init_year=33, lat=159, lon=90)
             init month = October
             SPEI evaluated over months 1-3 after init (Nov, Dec, Jan)
             one AGGREGATED value per member/init_year -> verification is
             against the END month of the window (January for Oct init).
  obs      : SPEI_Mod-Hargreaves_3m_1993-2026_Madagascar
             dims (time, lat, lon), monthly, 3-month SPEI.

Metrics implemented here:
  - CRPS  continuous ranked probability score, energy form
  - CRPSS CRPS skill score vs the pooled hindcast ensemble
  - RPS   ranked probability score over the three SPEI terciles
  - RPSS  RPS skill score vs the pooled hindcast ensemble
  - rank histogram flatness (calibration summary)
  - mean tercile probabilities
  - year-block bootstrap CI on the domain-mean CRPSS and RPSS

Terciles: SPEI is standardised, so the below / normal / above-normal
categories are the fixed breakpoints -0.4307 and +0.4307 (the 1/3 and 2/3
quantiles of a standard normal), NOT quantiles estimated from the sample.

Reference: the pooled hindcast ensemble (all members, all init years),
which is the standard climatological benchmark and keeps the comparison
inside one standardisation. Against the observed climatology the skill
scores would not be interpretable if the two fields were fitted on
different samples.

Plots live in a third file, plot_spei_scores.py.

Output: gridded scores -> NetCDF (lat, lon), one variable per metric.
"""

import numpy as np
import xarray as xr

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INIT_MONTH   = 10          # October
LEADS        = (1, 2, 3)   # aggregate = Nov, Dec, Jan
LEAD_OFFSET  = 1           # lead n = init + n months  -> lead 1 = November

DIR_FCST = f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/monthly/init_{INIT_MONTH}/SPEI/"
PATH_FCST = DIR_FCST +\
        (f"SPEI_Mod-Hargreaves_lead123_init{INIT_MONTH}_1993-2025_Madagascar_"
         "res_ERA5-Land_qdm_ERA5-Land_MSWEP.nc")
DIR_OBS = "/ec/res4/scratch/ecme4047/ERA5-Land/SPEI/monthly/"
PATH_OBS  = DIR_OBS + "SPEI_Mod-Hargreaves_3m_1993-2026_Madagascar.nc"

# Aggregated field verifies against the last month of the window
# (January for an October init), which is init month + 3.
VERIFY_MONTH_OFFSET = 3

PATH_OUT_NC = DIR_FCST + \
        (f"SPEI_Mod-Hargreaves_probabilistic_scores_lead123_"
         f"init{INIT_MONTH}_1993-2025_Madagascar.nc")
WRITE_CSV   = False          # also write the flattened grid as CSV
WRITE_NC    = True           # gridded scores -> NetCDF

# tercile breakpoints: the 1/3 and 2/3 quantiles of a standard normal.
# SPEI is standardised, so these are fixed, not estimated from the sample.
TERCILE_EDGES = (-0.4307272992954575, 0.4307272992954575)

# bootstrap for the domain-mean skill scores
N_BOOTSTRAP     = 1000
BOOTSTRAP_ALPHA = 0.05       # 95% interval


# ----------------------------------------------------------------------
# LEAD / MONTH BOOKKEEPING
# ----------------------------------------------------------------------
def window_months(init_year):
    """[(year, month), ...] for each lead of the Oct-init window."""
    return [((init_year + (INIT_MONTH - 1 + L * LEAD_OFFSET) // 12),
             (INIT_MONTH - 1 + L * LEAD_OFFSET) % 12 + 1) for L in LEADS]


def verify_month(init_year):
    """Observation month the aggregated window is verified against."""
    m = INIT_MONTH - 1 + VERIFY_MONTH_OFFSET
    return np.datetime64(f"{init_year + m // 12}-{m % 12 + 1:02d}-01")


def month_index(dates):
    """Map a datetime64 array to YYYY-MM integers for fast lookup."""
    d = np.asarray(dates, dtype="datetime64[M]")
    y = d.astype("datetime64[Y]").astype(int) + 1970
    m = d.astype(int) % 12 + 1
    return y * 100 + m


def window_fully_observed(init_year, obs_months):
    """True when EVERY month of the window exists in the observation record."""
    return all(y * 100 + m in obs_months for y, m in window_months(init_year))


# ----------------------------------------------------------------------
# PROBABILISTIC SCORES
# ----------------------------------------------------------------------
def crps_ensemble(ens, y):
    """CRPS of an ensemble against observations, energy form.

        CRPS = E|X - y| - 0.5 * E|X - X'|

    ens : (member, case, lat, lon)
    y   : (case, lat, lon), or broadcastable to that shape
    """
    ens = np.asarray(ens)
    n = ens.shape[0]

    term1 = np.abs(ens - y).mean(axis=0)

    tot = np.zeros_like(term1, dtype=float)
    for i in range(n):
        tot += np.abs(ens - ens[i:i + 1, ...]).sum(axis=0)
    term2 = 0.5 * tot / (n * n)

    return term1 - term2


def rps_categories(probs, obs_cat):
    """Ranked probability score for ordered categories.

        RPS = sum_k (CDF_fcst_k - CDF_obs_k)^2

    probs   : (category, case, lat, lon), probabilities summing to 1
    obs_cat : (case, lat, lon), observed category index
    """
    probs = np.asarray(probs)
    K = probs.shape[0]

    cdf_f = np.cumsum(probs, axis=0)

    cdf_o = np.zeros_like(cdf_f)
    for k in range(K):
        cdf_o[k] = (obs_cat > k).astype(float)

    return ((cdf_f - cdf_o) ** 2).sum(axis=0)


def tercile_probs(f):
    """Forecast probabilities of the three SPEI terciles.

    f : (number, time, lat, lon)  ->  (3, time, lat, lon)
    """
    lo, hi = TERCILE_EDGES
    v = np.asarray(f.transpose("time", "number", "lat", "lon").values)

    p_below = (v < lo).mean(axis=1)
    p_above = (v > hi).mean(axis=1)
    p_normal = 1.0 - p_below - p_above

    return np.stack([p_below, p_normal, p_above], axis=0)


def tercile_index(o):
    """Observed tercile category: 0 below, 1 normal, 2 above."""
    lo, hi = TERCILE_EDGES
    v = np.asarray(o.transpose("time", "lat", "lon").values)
    return np.digitize(v, [lo, hi]).astype(float)


def rank_flatness(fv, ov):
    """L1 departure of the member-rank histogram from flat, per grid point.

    fv : (time, member, lat, lon)   ov : (time, lat, lon)

    Each observation is ranked among the members. A well-calibrated
    ensemble gives each rank equally often, so the score is the L1
    distance from a uniform histogram divided by the maximum possible:
    0 is perfectly flat, 1 is the worst possible shape.
    """
    T, M, Y, X = fv.shape
    counts = np.zeros((M + 1, Y, X), dtype=float)

    for t in range(T):
        ens = fv[t]                                   # (M, Y, X)
        obs = ov[t]                                   # (Y, X)
        valid = np.isfinite(obs)
        # rank of the observation among the members: 0 .. M
        rank = np.full((Y, X), -1, dtype=int)
        rank[valid] = (ens[:, valid] < obs[valid]).sum(axis=0)
        for r in range(M + 1):
            counts[r] += (rank == r)

    total = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = counts / total
    ideal = 1.0 / (M + 1)
    # maximum L1 distance from uniform is 2*(1 - ideal)
    l1 = np.abs(frac - ideal).sum(axis=0)
    flat = l1 / (2.0 * (1.0 - ideal))
    return np.where(total > 0, flat, np.nan)


def probabilistic_scores(f, o):
    """CRPS, CRPSS and the tercile RPS / RPSS, per grid point.

    f : (number, time, lat, lon)   o : (time, lat, lon)

    The reference for both skill scores is the POOLED HINDCAST ENSEMBLE
    (all members, all init years).
    """
    fv = np.asarray(f.transpose("time", "number", "lat", "lon").values)
    ov = np.asarray(o.transpose("time", "lat", "lon").values)
    # fv : (time, member, lat, lon)   ov : (time, lat, lon)

    # ---- CRPS of the forecast ------------------------------------------
    crps_fc = crps_ensemble(np.moveaxis(fv, 1, 0), ov)   # (time, lat, lon)

    # ---- CRPS of the reference: the pooled ensemble --------------------
    pooled = fv.reshape(-1, fv.shape[2], fv.shape[3])    # (T*M, lat, lon)
    y_flat = np.repeat(ov, fv.shape[1], axis=0)          # (T*M, lat, lon)
    crps_ref = crps_ensemble(pooled, y_flat)

    # np.nanmean on an all-NaN slice (a masked grid point) warns once per
    # point and returns NaN; use the quiet helper instead.
    crpss = 1.0 - _colmean(crps_fc) / _colmean(crps_ref)

    # ---- tercile RPS ----------------------------------------------------
    probs = tercile_probs(f)                             # (3, T, lat, lon)
    cat = tercile_index(o)                               # (T, lat, lon)
    rps_fc = rps_categories(probs, cat)

    lo, hi = TERCILE_EDGES
    ref_probs = np.stack([
        (pooled < lo).mean(axis=0) * np.ones_like(cat),
        ((pooled >= lo) & (pooled <= hi)).mean(axis=0) * np.ones_like(cat),
        (pooled > hi).mean(axis=0) * np.ones_like(cat),
    ], axis=0)
    rps_ref = rps_categories(ref_probs, cat)

    rpss = 1.0 - np.nanmean(rps_fc, axis=0) / np.nanmean(rps_ref, axis=0)

    coords = {"lat": f.lat.values, "lon": f.lon.values}
    d2 = ("lat", "lon")

    return xr.Dataset({
        "crps":  xr.DataArray(_colmean(crps_fc), coords=coords, dims=d2),
        "crpss": xr.DataArray(crpss, coords=coords, dims=d2),
        "rps":   xr.DataArray(np.nanmean(rps_fc, axis=0),
                              coords=coords, dims=d2),
        "rpss":  xr.DataArray(rpss, coords=coords, dims=d2),
        "rank_hist_flatness": xr.DataArray(rank_flatness(fv, ov),
                                           coords=coords, dims=d2),
        "p_below":  xr.DataArray(np.nanmean(probs[0], axis=0),
                                 coords=coords, dims=d2),
        "p_normal": xr.DataArray(np.nanmean(probs[1], axis=0),
                                 coords=coords, dims=d2),
        "p_above":  xr.DataArray(np.nanmean(probs[2], axis=0),
                                 coords=coords, dims=d2),
    })


def _bootstrap_draws(T, n_boot, seed):
    """Resampled year indices, one row per bootstrap replicate.

    Drawn up-front in the parent process so every worker uses the same
    reproducible set of draws, whatever order they complete in.
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(0, T, size=T) for _ in range(n_boot)]


def _colmean(a):
    """Mean over axis 0, quiet on all-NaN slices.

    np.nanmean warns ("Mean of empty slice") once for every masked grid
    point and returns NaN. The result is the same either way; this just
    keeps the console readable when the field is land-masked.
    """
    a = np.asarray(a, dtype=float)
    m = np.isfinite(a)
    counts = m.sum(axis=0)
    total = np.where(m, a, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, total / counts, np.nan)


def _safe_mean(a):
    """Mean over the finite values, or NaN when there are none.

    Used instead of np.nanmean so that an all-NaN slice (a land-masked
    grid point, or a bootstrap draw that hit only masked years) returns
    NaN quietly rather than emitting "Mean of empty slice" once per grid
    point per replicate.
    """
    a = np.asarray(a)
    m = np.isfinite(a)
    if not m.any():
        return float("nan")
    return float(a[m].mean())


def _bootstrap_one(payload):
    """Domain-mean CRPSS and RPSS for one resampled year set.

    Takes plain arrays rather than xarray objects so the payload pickles
    cheaply when it crosses a process boundary.
    """
    fv, ov, pick = payload
    try:
        sb = _score_arrays(fv[pick], ov[pick])
        return (_safe_mean(sb["crpss"]), _safe_mean(sb["rpss"]))
    except Exception:                                     # noqa: BLE001
        return (float("nan"), float("nan"))


def bootstrap_ci(f, o, n_boot=N_BOOTSTRAP, alpha=BOOTSTRAP_ALPHA, seed=0,
                 n_jobs=None, backend="threads"):
    """Year-block bootstrap CI for the domain-mean CRPSS and RPSS.

    Resamples whole verification years with replacement and recomputes
    the scores. Blocking by year is essential: grid points are spatially
    correlated, so a point-wise bootstrap gives absurdly narrow
    intervals.

    Parallelised with joblib when it is available. The work is dominated
    by numpy array arithmetic, which releases the GIL, so the default is
    the thread backend: it shares memory instead of pickling the arrays
    to every worker, and it is the faster of the two here. Pass
    backend="loky" to use separate processes instead.

    n_jobs=None lets joblib choose (-1, i.e. all cores, for joblib's
    default). Without joblib installed this falls back to a plain serial
    loop and still works.
    """
    T = f.sizes["time"]
    draws = _bootstrap_draws(T, n_boot, seed)

    # arrays are shipped once, then sliced per replicate
    fv = np.asarray(f.transpose("time", "number", "lat", "lon").values)
    ov = np.asarray(o.transpose("time", "lat", "lon").values)
    payloads = [(fv, ov, pick) for pick in draws]

    results = None
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_bootstrap_one)(p) for p in payloads
        )
    except ImportError:
        results = None

    if results is None:
        print("  joblib not available - running the bootstrap serially")
        results = [_bootstrap_one(p) for p in payloads]

    arr = np.asarray(results, dtype=float)          # (n_boot, 2)
    out = {}
    for i, name in enumerate(("crpss", "rpss")):
        a = arr[:, i]
        a = a[np.isfinite(a)]
        if a.size:
            out[name] = (float(np.percentile(a, 100 * alpha / 2)),
                         float(np.percentile(a, 100 * (1 - alpha / 2))))
        else:
            out[name] = (float("nan"), float("nan"))
    return out


def _score_arrays(fv, ov):
    """probabilistic_scores on plain arrays, for bootstrap workers.

    fv : (time, member, lat, lon)   ov : (time, lat, lon)

    Returns a dict of 2-D array-valued scores; the spatial reduction is
    left to the caller.
    """
    crps_fc = crps_ensemble(np.moveaxis(fv, 1, 0), ov)

    pooled = fv.reshape(-1, fv.shape[2], fv.shape[3])
    y_flat = np.repeat(ov, fv.shape[1], axis=0)
    crps_ref = crps_ensemble(pooled, y_flat)
    # np.nanmean on an all-NaN slice (a masked grid point) warns once per
    # point and returns NaN; use the quiet helper instead.
    crpss = 1.0 - _colmean(crps_fc) / _colmean(crps_ref)

    lo, hi = TERCILE_EDGES
    probs = np.stack([
        (fv < lo).mean(axis=1),
        1.0 - (fv < lo).mean(axis=1) - (fv > hi).mean(axis=1),
        (fv > hi).mean(axis=1),
    ], axis=0)                                            # (3, time, lat, lon)
    cat = np.digitize(ov, [lo, hi]).astype(float)         # (time, lat, lon)

    rps_fc = rps_categories(probs, cat)

    ref_probs = np.stack([
        (pooled < lo).mean(axis=0) * np.ones_like(cat),
        (((pooled >= lo) & (pooled <= hi)).mean(axis=0)
         * np.ones_like(cat)),
        (pooled > hi).mean(axis=0) * np.ones_like(cat),
    ], axis=0)
    rps_ref = rps_categories(ref_probs, cat)

    rpss = 1.0 - np.nanmean(rps_fc, axis=0) / np.nanmean(rps_ref, axis=0)

    return {"crpss": crpss, "rpss": rpss}


# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------
def build_output(ps, usable, n_eff):
    """Attach CF-style metadata and provenance coordinates."""
    out = ps.copy()

    out = out.assign_coords(
        n_cases=n_eff,
        n_members=int(ps.sizes.get("number", 0)) or 25,
        init_month=INIT_MONTH,
        first_init_year=usable[0],
        last_init_year=usable[-1],
        verify_month_offset=VERIFY_MONTH_OFFSET,
        tercile_edge_low=TERCILE_EDGES[0],
        tercile_edge_high=TERCILE_EDGES[1],
    )

    meta = {
        "crps":  ("Continuous ranked probability score", "-"),
        "crpss": ("CRPS skill score vs the pooled ensemble", "1"),
        "rps":   ("Ranked probability score over SPEI terciles", "-"),
        "rpss":  ("RPS skill score vs the pooled ensemble", "1"),
        "rank_hist_flatness": (
            "L1 departure of the rank histogram from flat (0 = flat)", "1"),
        "p_below":  ("Mean forecast probability of the below-normal tercile", "1"),
        "p_normal": ("Mean forecast probability of the normal tercile", "1"),
        "p_above":  ("Mean forecast probability of the above-normal tercile", "1"),
    }
    for name, (long_name, units) in meta.items():
        if name in out:
            out[name].attrs.update(long_name=long_name, units=units)
            out[name].attrs["_FillValue"] = np.float32(np.nan)

    out["lat"].attrs.update(units="degrees_north", standard_name="latitude",
                            axis="Y")
    out["lon"].attrs.update(units="degrees_east", standard_name="longitude",
                            axis="X")
    out.attrs.update(
        title="Probabilistic SPEI forecast skill scores",
        forecast=PATH_FCST,
        observation=PATH_OBS,
        init_month=f"{INIT_MONTH:02d}",
        forecast_window="leads 1-3 (Nov, Dec, Jan)",
        verification_month="init + 3 months (January)",
        scoring_period=f"{usable[0]}-{usable[-1]} ({n_eff} init years)",
        ensemble_members=str(int(ps.sizes.get("number", 25))),
        tercile_edges=f"{TERCILE_EDGES[0]:.4f}, {TERCILE_EDGES[1]:.4f}",
        reference_note=("Skill scores use the pooled hindcast ensemble as "
                        "reference, which keeps the comparison inside one "
                        "standardisation"),
        conventions="CF-1.8",
    )
    return out


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    obs = xr.open_dataset(PATH_OBS)["SPEI"]
    fcst = xr.open_dataset(PATH_FCST)["SPEI"]

    print("obs  sizes :", dict(obs.sizes))
    print("fcst sizes :", dict(fcst.sizes))

    if "lead" in fcst.dims:
        raise SystemExit(
            "Forecast carries a 'lead' dimension - it holds separate leads, "
            "not one aggregated window. Average over 'lead' first."
        )

    # ---- calendar ------------------------------------------------------
    obs_months = set(month_index(obs.time.values).tolist())
    print(f"obs covers {min(obs_months)//100}-{max(obs_months)//100}, "
          f"{len(obs_months)} distinct months")

    yrs = [int(y) for y in fcst.init_year.values]
    print(f"forecast init years: {yrs[0]}-{yrs[-1]} ({len(yrs)})")
    print("window for init {}: {}".format(
        yrs[0], " ".join(f"{y}-{m:02d}" for y, m in window_months(yrs[0]))))
    print(f"verifies against {verify_month(yrs[0])} (end of window)")

    usable = [y for y in yrs if window_fully_observed(y, obs_months)]
    dropped = [y for y in yrs if y not in usable]
    print(f"init years with the FULL window observed: {len(usable)}/{len(yrs)}")
    if dropped:
        print(f"  dropped (window runs past the obs record): {dropped}")
    n_eff = len(usable)

    # ---- build aligned arrays ------------------------------------------
    idx = [yrs.index(y) for y in usable]
    tgt = np.array([verify_month(y) for y in usable])

    f = fcst.isel(init_year=idx).assign_coords(time=("init_year", tgt))
    f = f.swap_dims({"init_year": "time"}).transpose(
        "number", "time", "lat", "lon")

    o = obs.sel(time=tgt).transpose("time", "lat", "lon")

    got = month_index(o.time.values)
    want = np.array([int(str(t)[:4]) * 100 + int(str(t)[5:7]) for t in tgt])
    print(f"obs months pulled, first/last: {got[0]} / {got[-1]}"
          f"  {'OK' if np.array_equal(got, want) else 'MISMATCH'}")

    print(f"\nscoring {f.sizes['time']} init years x {f.sizes['number']} members"
          f" on {f.sizes['lat']}x{f.sizes['lon']} grid")
    print(f"forecast NaN fraction: {float(np.isnan(f.values).mean()):.4f}")

    # ---- scores ---------------------------------------------------------
    ps = probabilistic_scores(f, o)

    pkeys = ["crps", "crpss", "rps", "rpss", "rank_hist_flatness"]
    print("\n=== Domain-mean probabilistic scores ===")
    print(ps[pkeys].mean(["lat", "lon"]).to_pandas().round(3).to_string())

    print("\nskill > 0 at this fraction of grid points:")
    print(f"  crpss : {(ps['crpss'].values > 0).mean():.3f}")
    print(f"  rpss  : {(ps['rpss'].values > 0).mean():.3f}")

    print("\n=== Bootstrap CI on the domain mean (year blocks) ===")
    ci = bootstrap_ci(f, o)
    for name in ("crpss", "rpss"):
        lo, hi = ci[name]
        print(f"  {name}: 95% CI [{lo:+.3f}, {hi:+.3f}]")

    # ---- gridded output -------------------------------------------------
    out = build_output(ps, usable, n_eff)

    if WRITE_NC:
        out.to_netcdf(PATH_OUT_NC)
        print(f"\nwrote {PATH_OUT_NC}")
        print("  variables:", ", ".join(out.data_vars))
        print(f"  dims     : {dict(out.sizes)}")

    if WRITE_CSV:
        df = out.to_dataframe().reset_index().dropna(subset=["crpss"])
        csv_name = ("SPEI_Mod-Hargreaves_probabilistic_scores_lead123_"
                    f"init{INIT_MONTH}_1993-2025_Madagascar.csv")
        df.to_csv(csv_name, index=False)
        print(f"wrote {csv_name} ({len(df)} rows)")

    return out


if __name__ == "__main__":
    main()

