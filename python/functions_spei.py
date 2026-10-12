import calendar
import numpy as np
import xarray as xr
import pandas as pd
from scipy.special import gamma
from scipy.stats import norm
import os
from multiprocessing import Pool
import socket


# ---------------------------------------------------------------- utilities
def get_scratch_path():
    """Return the scratch root directory based on the current cluster.

    Dispatches on the machine hostname. Returns the path as a string, or
    None if the hostname is not recognised (the caller must handle it —
    note that the original `return print(...)` silently returned None).
    """
    hostname = socket.gethostname()
    if hostname == "ophfe1":
        return "/home/PERSONALE/alice.portal2/scratch/"
    elif "ac6" in hostname:
        return "/ec/res4/scratch/ecme4047/"
    return None


def boxes_african_countries(name_country):
    """Return the lon/lat bounding box for a supported country name.

    Bounds are in degrees; the box is given as lon_min/lon_max/lat_min/
    lat_max and is consumed by `subset_box`. Raises KeyError for an
    unsupported country.
    """
    boxes = {
        'ghana':      {'lon_min': -4, 'lon_max': 2,  'lat_min': 4,   'lat_max': 12},
        'madagascar': {'lon_min': 42, 'lon_max': 51, 'lat_min': -27, 'lat_max': -11},
    }
    return boxes[name_country.lower()]


def subset_box(da, box):
    """Subset a DataArray to a geographic box, handling both coord namings.

    Accepts 'longitude'/'latitude' or 'lon'/'lat'. The array is sorted by
    longitude first so that the slice is always increasing; latitude is
    sliced in descending order (lat_max -> lat_min) as is conventional for
    atmospheric fields. The result inherits the original coordinate names.
    """
    lon_name = "longitude" if "longitude" in da.dims or "longitude" in da.coords else "lon"
    lat_name = "latitude" if "latitude" in da.dims or "latitude" in da.coords else "lat"

    da = da.sortby(lon_name)
    return da.sel(
        {lat_name: slice(box['lat_max'], box['lat_min']),
         lon_name: slice(box['lon_min'], box['lon_max'])}
    )


def standardize_latlon(da):
    """Rename 'latitude'/'longitude' coords to the short 'lat'/'lon'.

    Applied after `subset_box` so all downstream code can assume the short
    names. Returns the renamed DataArray (a no-op if already short).
    """
    rename_dict = {}
    if "latitude" in da.dims or "latitude" in da.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in da.dims or "longitude" in da.coords:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        da = da.rename(rename_dict)
    return da


# ---------------------------------------------------------------- extraterrestrial Ra
def _ra_daily(lat_rad, doys):
    """Daily extraterrestrial radiation Ra in MJ m-2 day-1, vectorised.

    Implements FAO-56 Eq. 21. lat_rad is (L,) in radians and doys is (D,)
    day-of-year, giving a (D, L) result. The solar constant is 0.0820
    MJ m-2 min-1, `dr` is the inverse relative Earth-Sun distance and
    `delta` the solar declination. The sunset hour angle `ws` is clipped to
    [-1, 1] to keep arccos defined through polar day/night.
    """
    doys = np.asarray(doys, dtype=float)
    dr    = 1 + 0.033 * np.cos(2 * np.pi * doys / 365)[:, None]
    delta = 0.409 * np.sin(2 * np.pi * doys / 365 - 1.39)[:, None]
    sin_lat, cos_lat = np.sin(lat_rad)[None, :], np.cos(lat_rad)[None, :]

    ws = np.arccos(np.clip(-np.tan(lat_rad)[None, :] * np.tan(delta), -1, 1))  # (D, L)
    Gsc = 0.0820
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * sin_lat * np.sin(delta) + cos_lat * np.cos(delta) * np.sin(ws)
    )
    return Ra  # (D, L)


def _ra_monthly_cached(times, lat_deg, _cache={}):
    """Monthly mean Ra (MJ m-2 day-1) for each timestamp, memoised per (year, month).

    Ra varies with day-of-year and latitude, so for a monthly series we take
    the mean daily Ra over the calendar days of each month. `_cache` keys on
    (year, month) and is safe to share across calls because Ra for a given
    month and latitude is identical in every call. Returns (T, L).
    """
    lat_rad = np.radians(np.asarray(lat_deg, dtype=float))
    times = pd.DatetimeIndex(times)
    out = np.empty((len(times), len(lat_rad)), dtype=float)

    for i, t in enumerate(times):
        key = (t.year, t.month)
        if key not in _cache:
            y, m = key
            n_days = calendar.monthrange(y, m)[1]
            first_doy = sum(calendar.monthrange(y, mm)[1] for mm in range(1, m)) + 1
            doys = np.arange(first_doy, first_doy + n_days)
            _cache[key] = _ra_daily(lat_rad, doys).mean(axis=0)
        out[i] = _cache[key]
    return out


def _ra_dataarray(ds, time_dim='time', lat_dim='lat'):
    """Wrap `_ra_monthly_cached` output as a DataArray aligned to `ds`.

    Carries the time and latitude coordinates from `ds` so the result can be
    multiplied directly with the temperature variables in the ET0 functions.
    """
    Ra = _ra_monthly_cached(ds[time_dim].values, ds[lat_dim].values)
    return xr.DataArray(
        Ra, dims=[time_dim, lat_dim],
        coords={time_dim: ds[time_dim], lat_dim: ds[lat_dim]},
    )


# ---------------------------------------------------------------- Hargreaves
def hargreaves_modified(ds, Ra=None, tmin_var='tmin', tmax_var='tmax',
                        precip_var='precip', time_dim='time', lat_dim='lat'):
    """Modified Hargreaves ET0 (Droogers & Allen 2002), in mm/day.

    Uses the precipitation-corrected temperature range,
        Td_adj = (Tmax - Tmin) - 0.0123 * P,
    floored at zero, with coefficients 0.0013 and 0.76 and the +17.0 offset.
    Requires tmin, tmax and precip in `ds` (units: degC and mm/day).
    Ra is (T, L) in MJ m-2 day-1 via `_ra_dataarray`; 0.408 converts to
    mm/day equivalent.
    """
    if Ra is None:
        Ra = _ra_dataarray(ds, time_dim, lat_dim)
    tmin, tmax, precip = ds[tmin_var], ds[tmax_var], ds[precip_var]

    tmean  = (tmax + tmin) / 2
    td_adj = (tmax - tmin) - 0.0123 * precip
    td_safe = xr.where(td_adj < 0, 0, td_adj)
    Ra_mm  = 0.408 * Ra

    ET0 = 0.0013 * (tmean + 17.0) * (td_safe ** 0.76) * Ra_mm
    ET0.name = 'ET0'
    ET0.attrs['units'] = 'mm/day'
    return ET0


def hargreaves(ds, Ra=None, tmin_var='tmin', tmax_var='tmax',
               time_dim='time', lat_dim='dim'):
    """Classic Hargreaves-Samani ET0, in mm/day.

    The original formulation: 0.0023 * (Tmean + 17.8) * sqrt(Tmax - Tmin)
    * Ra_mm. Not called anywhere in this module — kept as a reference
    implementation for comparison against `hargreaves_modified`. Note the
    default `lat_dim` differs from the modified version's.
    """
    if Ra is None:
        Ra = _ra_dataarray(ds, time_dim, lat_dim)
    tmin, tmax = ds[tmin_var], ds[tmax_var]

    tmean = (tmax + tmin) / 2
    td    = tmax - tmin
    Ra_mm = 0.408 * Ra

    ET0 = 0.0023 * (tmean + 17.8) * (td ** 0.5) * Ra_mm
    ET0.name = 'ET0'
    ET0.attrs['units'] = 'mm/day'
    return ET0


# ---------------------------------------------------------------- spei
def _fit_loglogistic_pwm(series):
    """Fit a 3-parameter log-logistic by unbiased probability-weighted moments.

    Follows the parameterisation used by the SPEI R package
    (Vicente-Serrano et al. 2010, Appendix). PWM are computed with the
    unbiased plotting positions (i-1)/(n-1) and (i-1)(i-2)/((n-1)(n-2)),
    then inverted to shape `beta`, location `loc` and scale `scale`:

        beta  = (2*w1 - w0) / (6*w1 - w0 - 6*w2)
        gg    = gamma(1 + 1/beta) * gamma(1 - 1/beta)
        scale = (w0 - 2*w1) * beta / gg
        loc   = w0 - scale * gg

    Returns (nan, nan, nan) when the sample is too small, or a moment is
    degenerate.
    """
    vals = np.asarray(series, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size < 3:
        return np.nan, np.nan, np.nan

    vals = np.sort(vals)
    n = vals.size
    i = np.arange(1, n + 1, dtype=float)

    w0 = np.mean(vals)
    w1 = np.sum(((i - 1) / (n - 1)) * vals) / n
    w2 = np.sum(((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * vals) / n)

    denom = 6.0 * w1 - w0 - 6.0 * w2
    if not np.isfinite(denom) or np.isclose(denom, 0.0):
        return np.nan, np.nan, np.nan

    beta = (2.0 * w1 - w0) / denom
    if not np.isfinite(beta):
        return np.nan, np.nan, np.nan

    gg = gamma(1.0 + 1.0 / beta) * gamma(1.0 - 1.0 / beta)
    if not np.isfinite(gg) or np.isclose(gg, 0.0):
        return np.nan, np.nan, np.nan

    scale = ((w0 - 2.0 * w1) * beta) / gg
    loc = w0 - scale * gg

    if not np.isfinite(scale) or not np.isfinite(loc):
        return np.nan, np.nan, np.nan

    return beta, loc, scale


def _loglogistic_cdf(x, beta, loc, scale):
    """Log-logistic CDF F(x) = 1 / (1 + (scale / (x - loc))**-beta).

    x may be any shape; a matching array of probabilities is returned with
    NaN where x is not finite.

    """
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)

    valid = np.isfinite(x)
    if np.any(valid):
        z = (scale / (x[valid] - loc)) ** beta
        out[valid] = z / (1.0 + z)

    return out


def _spei_1d(values, months, cal_mask):
    """Compute SPEI for a single grid point.

    `values` is the accumulated water balance (1-D, time ordered), `months`
    the corresponding month numbers and `cal_mask` a boolean selecting the
    calibration period. A separate log-logistic is fitted for each calendar
    month on the calibration sample only, removing the seasonal cycle; that
    month's fitted parameters are then applied to every year of that month
    across the whole record.

    Probabilities are transformed with the standard normal quantile, exactly
    as the R package does — a fitted CDF of 0.5 maps to SPEI 0. Months whose
    calibration sample is too small to fit are left as NaN.
    """
    out = np.full(values.shape, np.nan, dtype=float)

    for m in range(1, 13):
        idx = (months == m)
        idx_cal = idx & cal_mask

        cal_vals = values[idx_cal]
        cal_vals = cal_vals[np.isfinite(cal_vals)]

        if cal_vals.size < 8:
            continue

        beta, loc, scale = _fit_loglogistic_pwm(cal_vals)
        if np.isnan(beta):
            continue

        vals = values[idx]
        good = np.isfinite(vals)
        if not np.any(good):
            continue

        probs = np.full(vals.shape, np.nan, dtype=float)
        probs[good] = _loglogistic_cdf(vals[good], beta, loc, scale)

        out_month = np.full(vals.shape, np.nan, dtype=float)
        out_month[good] = norm.ppf(probs[good])
        out[idx] = out_month

    return out


def _fit_loglogistic_pwm_diagnostic(series):
    """Diagnostic wrapper around the PWM fit, reporting the failure reason.

    Returns a dict with `ok` plus, on failure, a `reason` of
    'too_few_values', 'bad_denom', 'bad_beta', 'bad_gamma_product',
    'bad_scale' or 'bad_loc'. On success it also carries `n` and every
    intermediate moment (w0/w1/w2, denom, beta, g1/g2, gg, loc, scale) so a
    failed month can be inspected without re-deriving the arithmetic.
    """
    vals = np.asarray(series, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size < 3:
        return {
            "ok": False, "reason": "too_few_values",
            "beta": np.nan, "loc": np.nan, "scale": np.nan,
        }

    vals = np.sort(vals)
    n = vals.size
    i = np.arange(1, n + 1, dtype=float)

    w0 = np.mean(vals)
    w1 = np.sum(((i - 1) / (n - 1)) * vals) / n
    w2 = np.sum(((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * vals) / n)

    denom = 6.0 * w1 - w0 - 6.0 * w2
    if not np.isfinite(denom) or np.isclose(denom, 0.0):
        return {
            "ok": False, "reason": "bad_denom",
            "w0": w0, "w1": w1, "w2": w2, "denom": denom,
            "beta": np.nan, "loc": np.nan, "scale": np.nan,
        }

    beta = (2.0 * w1 - w0) / denom
    if not np.isfinite(beta):
        return {
            "ok": False, "reason": "bad_beta",
            "w0": w0, "w1": w1, "w2": w2, "denom": denom,
            "beta": beta, "loc": np.nan, "scale": np.nan,
        }

    g1 = gamma(1.0 + 1.0 / beta)
    g2 = gamma(1.0 - 1.0 / beta)
    gg = g1 * g2

    if not np.isfinite(gg) or np.isclose(gg, 0.0):
        return {
            "ok": False, "reason": "bad_gamma_product",
            "w0": w0, "w1": w1, "w2": w2, "denom": denom,
            "beta": beta, "g1": g1, "g2": g2, "gg": gg,
            "loc": np.nan, "scale": np.nan,
        }

    scale = ((w0 - 2.0 * w1) * beta) / gg
    loc = w0 - scale * gg

    if not np.isfinite(scale):
        return {
            "ok": False, "reason": "bad_scale",
            "w0": w0, "w1": w1, "w2": w2, "denom": denom,
            "beta": beta, "g1": g1, "g2": g2, "gg": gg,
            "loc": loc, "scale": scale,
        }

    if not np.isfinite(loc):
        return {
            "ok": False, "reason": "bad_loc",
            "w0": w0, "w1": w1, "w2": w2, "denom": denom,
            "beta": beta, "g1": g1, "g2": g2, "gg": gg,
            "loc": loc, "scale": scale,
        }

    return {
        "ok": True, "reason": "ok",
        "n": n,
        "w0": w0, "w1": w1, "w2": w2, "denom": denom,
        "beta": beta, "g1": g1, "g2": g2, "gg": gg,
        "loc": loc, "scale": scale,
    }


def _spei_point_worker(args):
    """Multiprocessing entry point: run `_spei_1d` for one grid point.

    Takes a single tuple so it can be dispatched via `Pool.imap_unordered`,
    which only passes one argument. Returns (index, float32 array) so the
    parent can scatter results back without pickling the whole field. Points
    with fewer than `min_valid` finite values are returned all-NaN early,
    skipping the per-month fits entirely.
    """
    j, vals, months, cal_mask, min_valid = args

    if np.isfinite(vals).sum() < min_valid:
        return j, np.full(vals.shape, np.nan, dtype=np.float32)

    out = _spei_1d(vals, months, cal_mask).astype(np.float32)
    return j, out


def compute_spei(balance, scale, cal_start, cal_end,
                 time_dim='time', lat_dim='lat', lon_dim='lon',
                 n_workers=None, chunk_size=200):
    """Compute the SPEI field at a given accumulation scale.

    `balance` is the climatic water balance (P - ET0), monthly. The series is
    accumulated over a `scale`-month window with min_periods=scale, so the
    leading scale-1 steps are NaN rather than partial sums. A log-logistic is
    fitted per grid point and per calendar month on the `cal_start`-`cal_end`
    window, then applied to the whole record, and the resulting probability
    is transformed to a standard normal score.

    Space is stacked into a single `point` dimension before the loop so the
    per-point work parallelises cleanly; points with any finite value are
    kept and the rest are filled back with NaN at the end. `n_workers`
    defaults to SLURM_CPUS_PER_TASK (1 if unset), and runs serially in that
    case.

    Returns a DataArray named 'SPEI' with the original space restored,
    carrying scale, calibration window and units in its attrs.
    """
    # Rolling accumulation of climatic water balance
    accum = balance.rolling({time_dim: scale}, min_periods=scale).sum()

    months = accum[time_dim].dt.month.values
    cal_mask = (
        (accum[time_dim] >= np.datetime64(cal_start)) &
        (accum[time_dim] <= np.datetime64(cal_end))
    ).values

    # Stack space into one dimension
    accum_stacked = accum.stack(point=(lat_dim, lon_dim))
    arr = accum_stacked.values  # shape: (time, point)

    valid_points = np.isfinite(arr).any(axis=0)
    arr_valid = arr[:, valid_points]

    out_valid = np.full(arr_valid.shape, np.nan, dtype=np.float32)

    min_valid = max(8, scale + 6)

    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    if n_workers <= 1:
        for k in range(arr_valid.shape[1]):
            vals = arr_valid[:, k]
            if np.isfinite(vals).sum() < min_valid:
                continue
            out_valid[:, k] = _spei_1d(vals, months, cal_mask)
    else:
        tasks = [
            (k, arr_valid[:, k], months, cal_mask, min_valid)
            for k in range(arr_valid.shape[1])
        ]
        with Pool(processes=n_workers) as pool:
            for k, out in pool.imap_unordered(_spei_point_worker, tasks, chunksize=chunk_size):
                out_valid[:, k] = out

    out = np.full(arr.shape, np.nan, dtype=np.float32)
    out[:, valid_points] = out_valid

    spei = xr.DataArray(
        out,
        dims=accum_stacked.dims,
        coords=accum_stacked.coords,
        name="SPEI",
    ).unstack("point")

    spei.attrs["units"] = "-"
    spei.attrs["scale_months"] = scale
    spei.attrs["calibration_start"] = cal_start
    spei.attrs["calibration_end"] = cal_end

    return spei

