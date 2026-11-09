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

    base = scale / (x - loc) 
    valid = np.isfinite(x) & (base>=0)
    if np.any(valid):
        z = base[valid] ** (-beta)
        out[valid] = 1.0 / (1.0 + z)

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


# ==========================================================================
# I/O, regridding and ET0/SPEI pipeline helpers
# ==========================================================================
#
# Added to keep the driver script thin. Nothing here changes the behaviour of
# the ET0 / SPEI routines above; it only wraps them.
#
# Requires `xesmf` for the regridding helper. The import is deferred to the
# call site so this module still imports on a machine without xesmf.
# ==========================================================================

KELVIN_OFFSET = 273.15

# Visible name of each ET0 formulation -> the function above that computes it.
ET0_FUNCS = {
    "Hargreaves": "hargreaves",
    "Mod-Hargreaves": "hargreaves_modified",
}


def resolve_paths(country, scratch_root=None):
    """All input/output locations for the ET0 + SPEI pipeline.

    `scratch_root` overrides `get_scratch_path()` for testing or for running
    off-cluster. Raises RuntimeError when no root can be determined.
    """
    d = scratch_root if scratch_root is not None else get_scratch_path()
    if d is None:
        raise RuntimeError(
            "get_scratch_path() returned None: this hostname is not "
            "recognised. Pass scratch_root explicitly or run on the cluster."
        )
    root = f"{d}MSWEP/MSWEP_V316_test/Past/"
    return {
        "t2m": f"{d}ERA5-Land/t2m/monthly/",
        "mswep": f"{root}Monthly/",
        "et0": f"{d}ERA5-Land/ET0/monthly/",
        "precip": f"{root}Monthly/regridded_ERA5-Land/",
        "spei": f"{d}ERA5-Land/SPEI/monthly/",
        "weights": f"{root}Daily/regridded_ERA5-Land/"
                   f"weights_mswep_to_era5land_{country}.nc",
        "mask": f"{d}ERA5-Land/land_mask_ERA5-Land.nc",
    }


def to_celsius(da, sensor):
    """Convert a temperature field to degC if it looks like Kelvin.

    Checks the `units` attribute first; samples a single value only when the
    attribute is missing, and only that one value (not the whole grid).
    """
    units = (sensor.attrs.get("units") or "").strip().lower()
    if units in {"k", "kelvin", "degk", "degrees_k"}:
        return da - KELVIN_OFFSET
    if units and ("c" in units or "celsius" in units):
        return da
    probe = float(np.atleast_1d(sensor.isel({sensor.dims[0]: 0}).values)[0])
    return da - KELVIN_OFFSET if probe > 100.0 else da


def standardize_dims(ds):
    """Rename ERA5-Land's verbose dims to the short names used downstream.

    `standardize_latlon` above handles lat/lon only; this also maps
    `valid_time` to `time`.
    """
    rename = {}
    for old, new in (("valid_time", "time"),
                     ("latitude", "lat"),
                     ("longitude", "lon")):
        if old in ds.dims or old in ds.coords:
            rename[old] = new
    return ds.rename(rename) if rename else ds


def grid_shape(da):
    """(n_lat, n_lon) for either naming convention."""
    lat = "latitude" if "latitude" in da.dims else "lat"
    lon = "longitude" if "longitude" in da.dims else "lon"
    return da.sizes[lat], da.sizes[lon]


def get_regridder(src, target, weights_path=None, force=False):
    """Build a conservative regridder, caching weights on disk.

    xesmf reuses cached weights via `filename=` + `reuse_weights=True`, and
    raises on a grid mismatch rather than silently mis-regridding.
    """
    try:
        import xesmf as xe
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "xesmf is required for the regridding stage but is not importable."
        ) from exc

    reuse = bool(weights_path) and os.path.exists(weights_path) and not force
    if reuse:
        return xe.Regridder(src, target, method="conservative", periodic=False,
                            filename=weights_path, reuse_weights=True)

    regridder = xe.Regridder(src, target, method="conservative",
                             periodic=False, reuse_weights=False)
    if weights_path:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        regridder.to_netcdf(weights_path)
    return regridder


def build_et0_and_precip(paths, country, years, methods,
                         force_regrid=False, verbose=True):
    """Loop the years; return (et0, precip) as multi-year datasets.

    ET0 is returned in mm/day (as the functions above produce it); precip as
    mm/month. Dims are time/lat/lon throughout.

    Precip is regridded to the ERA5-Land grid of the first year, and the
    regridder is rebuilt whenever a year's source grid differs, rather than
    silently applying mismatched weights.
    """
    box = boxes_african_countries(country)
    y0, y1 = years
    years_iter = range(y0, y1 + 1)

    with xr.open_dataset(f"{paths['t2m']}t2m_maximum_monthly_{y0}.nc",
                         chunks={"valid_time": 12}) as ds_t:
        target = standardize_dims(
            subset_box(ds_t["t2m"].rename({"valid_time": "time"}), box)
        )

    regridder = None
    regridder_shape = None

    et0_parts = {m: [] for m in methods}
    precip_parts = []

    for year in years_iter:
        with xr.open_dataset(f"{paths['t2m']}t2m_minimum_monthly_{year}.nc",
                             chunks={"valid_time": 12}) as ds_tmin, \
             xr.open_dataset(f"{paths['t2m']}t2m_maximum_monthly_{year}.nc",
                             chunks={"valid_time": 12}) as ds_tmax, \
             xr.open_dataset(f"{paths['mswep']}{year}.nc",
                             chunks={"time": 12}) as ds_p:

            tmin = to_celsius(ds_tmin["t2m"], ds_tmin["t2m"])
            tmax = to_celsius(ds_tmax["t2m"], ds_tmin["t2m"])  # same units
            precip = ds_p["precipitation"]

            tmin_c = standardize_latlon(subset_box(
                tmin.rename({"valid_time": "time"}), box))
            tmax_c = standardize_latlon(subset_box(
                tmax.rename({"valid_time": "time"}), box))

            src_shape = grid_shape(precip)
            if regridder is None or src_shape != regridder_shape:
                regridder = get_regridder(precip, target, paths.get("weights"),
                                          force_regrid)
                regridder_shape = src_shape

            precip_c = (regridder(precip)
                        .resample(time="1MS").sum()
                        .rename("precipitation"))
            precip_c = standardize_latlon(precip_c)
            precip_c.attrs["units"] = "mm/month"

            tmin_m = tmin_c.resample(time="1MS").mean()
            tmax_m = tmax_c.resample(time="1MS").mean()

            # mm/day equivalent of the monthly total: hargreaves_modified
            # needs it for its 0.0123 * P correction. Feeding the monthly
            # total instead biases ET0 (about 13% for a tropical month).
            n_days = xr.DataArray(tmin_m["time"].dt.days_in_month,
                                  coords={"time": tmin_m["time"]}, dims=["time"])
            precip_day = (precip_c / n_days).transpose("time", "lat", "lon")

            ds_monthly = xr.Dataset({"tmin": tmin_m, "tmax": tmax_m,
                                     "precip": precip_day})
            Ra = _ra_dataarray(ds_monthly)

            for method in methods:
                fn = globals()[ET0_FUNCS[method]]
                et0_parts[method].append(fn(ds_monthly, Ra=Ra))

            precip_parts.append(precip_c)
            if verbose:
                print(f"[ET0] {year} done", flush=True)

    del regridder

    et0 = xr.Dataset({m: xr.concat(parts, dim="time")
                      for m, parts in et0_parts.items()})
    for m in methods:
        et0[m].attrs["units"] = "mm/day"

    precip_ds = xr.Dataset(
        {"precipitation": xr.concat(precip_parts, dim="time")})
    precip_ds["precipitation"].attrs["units"] = "mm/month"
    return et0, precip_ds


def land_fraction(paths, country, ref):
    """Land-sea mask interpolated onto the grid of `ref`."""
    with xr.open_dataset(paths["mask"]) as ds_mask:
        lsm = standardize_latlon(
            subset_box(ds_mask["lsm"], boxes_african_countries(country)))
        lsm = lsm.load()  # close the file before interpolating
    return lsm.interp(lat=ref.lat, lon=ref.lon, method="nearest")


def water_balance(precip, et0_daily, lsm=None):
    """P - ET0 in mm/month.

    ET0 is mm/day, so it is multiplied by days-in-month; precip is already a
    monthly total. With `lsm` given, the result is masked to land.
    """
    days = xr.DataArray(et0_daily["time"].dt.days_in_month,
                        coords={"time": et0_daily["time"]}, dims=["time"])
    pet = (et0_daily * days).rename("pet")
    pet.attrs["units"] = "mm/month"

    precip, pet = xr.align(precip.rename("precip"), pet, join="inner")
    balance = (precip - pet).rename("balance")
    balance.attrs["units"] = "mm/month"
    balance.attrs["long_name"] = "climatic water balance (P - ET0)"

    if lsm is not None:
        balance = balance.where(lsm > 0.5)
    return balance


def write_spei(balance, out_dir, method, years, scales, country,
               verbose=True):
    """Compute SPEI at each scale and write it, plus the balance, to `out_dir`.

    Returns the dict of {scale: surname} written.
    """
    y0, y1 = years
    cal_start, cal_end = f"{y0}-01-01", f"{y1}-12-31"
    os.makedirs(out_dir, exist_ok=True)
    written = {}

    for scale in scales:
        spei = compute_spei(balance, scale=scale,
                            cal_start=cal_start, cal_end=cal_end)
        name = f"SPEI_{method}_{scale}m_{y0}-{y1}_{country}.nc"
        spei.to_netcdf(os.path.join(out_dir, name))
        written[scale] = name
        if verbose:
            print(f"[SPEI] saved scale {scale}", flush=True)

    balance.to_netcdf(os.path.join(
        out_dir, f"water-balance_{method}_monthly_{y0}-{y1}_{country}.nc"))
    return written

