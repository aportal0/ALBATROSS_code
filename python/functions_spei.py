import calendar
import numpy as np
import xarray as xr
import pandas as pd


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


def _ra_dataarray(ds, time_dim='time', lat_dim='latitude'):
    Ra = _ra_monthly_cached(ds[time_dim].values, ds[lat_dim].values)
    return xr.DataArray(
        Ra, dims=[time_dim, lat_dim],
        coords={time_dim: ds[time_dim], lat_dim: ds[lat_dim]},
    )


# ---------------------------------------------------------------- Hargreaves
def hargreaves_modified(ds, Ra=None, tmin_var='tmin', tmax_var='tmax',
                        precip_var='precip', time_dim='time', lat_dim='latitude'):
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
               time_dim='time', lat_dim='latitude'):
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

