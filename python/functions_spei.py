import calendar
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import fisk, norm
from scipy.special import gamma
import os
from multiprocessing import Pool



def boxes_african_countries(name_country):
    boxes = {
        'ghana':     {'lon_min': -4,  'lon_max': 2,  'lat_min': 4,   'lat_max': 12},
        'madagascar': {'lon_min': 42, 'lon_max': 51, 'lat_min': -27, 'lat_max': -11},
    }
    return boxes[name_country.lower()]


def subset_box(da, box):
    da = da.sortby('longitude')
    return da.sel(
        latitude=slice(box['lat_max'], box['lat_min']),
        longitude=slice(box['lon_min'], box['lon_max']),
    )


# ---------------------------------------------------------------- extraterrestrial Ra
def _ra_daily(lat_rad, doys):
    """Daily Ra (MJ/m²/day), fully vectorized.

    lat_rad: (L,) radians ; doys: (D,) day-of-year → (D, L).
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
    """Monthly Ra (MJ/m²/day), cached per (year, month).

    times: array of datetime-like (T,) ; lat_deg: (L,) → (T, L).
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
    Ra = _ra_monthly_cached(ds[time_dim].values, ds[lat_dim].values)
    return xr.DataArray(
        Ra, dims=[time_dim, lat_dim],
        coords={time_dim: ds[time_dim], lat_dim: ds[lat_dim]},
    )


# ---------------------------------------------------------------- Hargreaves
def hargreaves_modified(ds, Ra=None, tmin_var='tmin', tmax_var='tmax',
                        precip_var='precip', time_dim='time', lat_dim='lat'):
    if Ra is None:
        Ra = _ra_dataarray(ds, time_dim, lat_dim)
    tmin, tmax, precip = ds[tmin_var], ds[tmax_var], ds[precip_var]

    tmean  = (tmax + tmin) / 2
    td_adj = (tmax - tmin) - 0.0123 * precip
    Ra_mm  = 0.408 * Ra

    ET0 = 0.0013 * (tmean + 17.0) * (td_adj ** 0.76) * Ra_mm
    ET0 = ET0.where(td_adj > 0).clip(min=0)
    ET0.name = 'ET0'
    ET0.attrs['units'] = 'mm/day'
    return ET0


def hargreaves(ds, Ra=None, tmin_var='tmin', tmax_var='tmax',
               time_dim='time', lat_dim='lat'):
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

    if not np.isfinite(scale) or not np.isfinite(loc) or scale <= 0.0:
        return np.nan, np.nan, np.nan

    return beta, loc, scale


def _loglogistic_cdf(x, beta, loc, scale):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)

    finite = np.isfinite(x)
    out[finite & (x <= loc)] = 0.0

    valid = finite & (x > loc)
    if np.any(valid):
        z = ((x[valid] - loc) / scale) ** beta
        out[valid] = z / (1.0 + z)

    return out


def _spei_1d(values, months, cal_mask):
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
        probs[good] = np.clip(probs[good], 1e-8, 1.0 - 1e-8)

        out_month = np.full(vals.shape, np.nan, dtype=float)
        out_month[good] = norm.ppf(probs[good])
        out[idx] = out_month

    return out


def _spei_point_worker(args):
    j, vals, months, cal_mask, min_valid = args

    if np.isfinite(vals).sum() < min_valid:
        return j, np.full(vals.shape, np.nan, dtype=np.float32)

    out = _spei_1d(vals, months, cal_mask).astype(np.float32)
    return j, out


def compute_spei(balance, scale, cal_start, cal_end,
                 time_dim='time', lat_dim='lat', lon_dim='lon',
                 n_workers=None, chunk_size=200):
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


