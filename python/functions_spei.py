import calendar
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import fisk, norm


def boxes_african_countries(name_country):
    boxes = {
        'ghana':     {'lon_min': -4,  'lon_max': 2,  'lat_min': 4,   'lat_max': 12},
        'madagascar': {'lon_min': 42, 'lon_max': 51, 'lat_min': -27, 'lat_max': -11},
    }
    return boxes[name_country.lower()]


def subset_box(da, box):
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
def _fit_fisk(series):
    vals = np.asarray(series, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size < 8:
        return np.nan, np.nan, np.nan

    try:
        c, loc, scale = fisk.fit(vals)
        if not np.isfinite(c) or not np.isfinite(loc) or not np.isfinite(scale):
            return np.nan, np.nan, np.nan
        if c <= 0 or scale <= 0:
            return np.nan, np.nan, np.nan
        return c, loc, scale
    except Exception:
        return np.nan, np.nan, np.nan


def _spei_1d(values, months, cal_mask):
    out = np.full(values.shape, np.nan, dtype=float)

    for m in range(1, 13):
        idx = (months == m)
        idx_cal = idx & cal_mask

        c, loc, scale = _fit_fisk(values[idx_cal])
        if np.isnan(c):
            continue

        vals = values[idx]
        good = np.isfinite(vals)
        if not np.any(good):
            continue

        probs = np.full(vals.shape, np.nan, dtype=float)
        probs[good] = fisk.cdf(vals[good], c, loc=loc, scale=scale)
        probs = np.clip(probs, 1e-8, 1 - 1e-8)

        out[idx] = norm.ppf(probs)

    return out


def compute_spei(balance, scale=1, cal_start="1993-01-01", cal_end="2020-12-31", time_dim='time'):
    # --- rolling accumulation ---
    accum = balance.rolling({time_dim: scale}, min_periods=scale).sum()

    months = accum[time_dim].dt.month.values
    cal_mask = (
        (accum[time_dim] >= np.datetime64(cal_start)) &
        (accum[time_dim] <= np.datetime64(cal_end))
    ).values

    arr = accum.values
    nt, ny, nx = arr.shape
    arr2 = arr.reshape(nt, ny * nx)

    out2 = np.full(arr2.shape, np.nan, dtype=np.float32)

    for j in range(arr2.shape[1]):
        out2[:, j] = _spei_1d(arr2[:, j], months, cal_mask)

    out = out2.reshape(nt, ny, nx)

    spei = xr.DataArray(
        out,
        dims=accum.dims,
        coords=accum.coords,
        name=f"SPEI_{scale:02d}",
    )
    spei.attrs['units'] = '-'
    spei.attrs['scale_months'] = scale
    spei.attrs['calibration_start'] = cal_start
    spei.attrs['calibration_end'] = cal_end

    return spei

