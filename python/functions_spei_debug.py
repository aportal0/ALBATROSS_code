import calendar
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import fisk, norm
from scipy.special import gamma
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt


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
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import norm


def inspect_loglogistic_fit(series):
    vals = np.asarray(series, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.sort(vals)

    n = vals.size
    i = np.arange(1, n + 1, dtype=float)

    w0 = np.mean(vals)
    w1 = np.sum(((i - 1) / (n - 1)) * vals) / n
    w2 = np.sum(((i - 1) * (i - 2) / ((n - 1) * (n - 2))) * vals) / n

    denom = 6.0 * w1 - w0 - 6.0 * w2
    beta = (2.0 * w1 - w0) / denom if np.isfinite(denom) and not np.isclose(denom, 0.0) else np.nan

    g1 = gamma(1.0 + 1.0 / beta) if np.isfinite(beta) else np.nan
    g2 = gamma(1.0 - 1.0 / beta) if np.isfinite(beta) else np.nan
    gg = g1 * g2 if np.isfinite(g1) and np.isfinite(g2) else np.nan

    scale = ((w0 - 2.0 * w1) * beta) / gg if np.isfinite(beta) and np.isfinite(gg) and not np.isclose(gg, 0.0) else np.nan
    loc = w0 - scale * gg if np.isfinite(scale) and np.isfinite(gg) else np.nan

    print("n      =", n)
    print("min    =", vals.min())
    print("max    =", vals.max())
    print("mean   =", vals.mean())
    print("w0     =", w0)
    print("w1     =", w1)
    print("w2     =", w2)
    print("denom  =", denom)
    print("beta   =", beta)
    print("g1     =", g1)
    print("g2     =", g2)
    print("gg     =", gg)
    print("scale  =", scale)
    print("loc    =", loc)

    return {
        "n": n, "w0": w0, "w1": w1, "w2": w2,
        "denom": denom, "beta": beta,
        "g1": g1, "g2": g2, "gg": gg,
        "scale": scale, "loc": loc
    }

    
def fit_loglogistic_pwm_debug(series):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 3:
        return None

    x = np.sort(x)
    n = x.size
    i = np.arange(1, n + 1, dtype=float)

    w0 = np.mean(x)
    w1 = np.sum(((i - 1) / (n - 1)) * x) / n
    w2 = np.sum(((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n)

    denom = 6.0 * w1 - w0 - 6.0 * w2
    if not np.isfinite(denom) or np.isclose(denom, 0.0):
        return None

    beta = (2.0 * w1 - w0) / denom
    if not np.isfinite(beta):
        return None

    g1 = gamma(1.0 + 1.0 / beta)
    g2 = gamma(1.0 - 1.0 / beta)
    gg = g1 * g2

    if not np.isfinite(gg) or np.isclose(gg, 0.0):
        return None

    scale = ((w0 - 2.0 * w1) * beta) / gg
    loc = w0 - scale * gg

    if not np.isfinite(scale) or not np.isfinite(loc) or scale <= 0.0:
        return None

    return {
        "n": n,
        "sorted_values": x,
        "w0": w0,
        "w1": w1,
        "w2": w2,
        "denom": denom,
        "beta": beta,
        "loc": loc,
        "scale": scale,
        "gg": gg,
    }



def loglogistic_cdf(x, beta, loc, scale):
    x = np.asarray(x, dtype=float)
    p = np.full(x.shape, np.nan, dtype=float)

    finite = np.isfinite(x)
    p[finite & (x <= loc)] = 0.0

    valid = finite & (x > loc)
    if np.any(valid):
        z = ((x[valid] - loc) / scale) ** beta
        p[valid] = z / (1.0 + z)

    return p


def empirical_plotting_positions(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = x.size
    ranks = np.arange(1, n + 1, dtype=float)

    # Common empirical plotting position for diagnostics
    p = (ranks - 0.35) / n
    return x, p


def rolling_water_balance(balance_1d, scale):
    s = pd.Series(balance_1d)
    return s.rolling(window=scale, min_periods=scale).sum().to_numpy()


def extract_month_sample(values, dates, month, cal_start, cal_end):
    dates = pd.to_datetime(dates)
    values = np.asarray(values, dtype=float)

    mask_month = dates.month == month
    mask_cal = (dates >= pd.Timestamp(cal_start)) & (dates <= pd.Timestamp(cal_end))

    sample_all = values[mask_month]
    sample_cal = values[mask_month & mask_cal]

    return {
        "all_values": sample_all,
        "cal_values": sample_cal,
        "dates_all": dates[mask_month],
        "dates_cal": dates[mask_month & mask_cal],
    }


def spei_from_fit(values, beta, loc, scale, eps=1e-8):
    p = loglogistic_cdf(values, beta, loc, scale)
    good = np.isfinite(p)
    out = np.full(np.shape(p), np.nan, dtype=float)
    p2 = p.copy()
    p2[good] = np.clip(p2[good], eps, 1.0 - eps)
    out[good] = norm.ppf(p2[good])
    return p, out


def monthwise_spei_diagnostic(values, dates, month, cal_start, cal_end):
    sample = extract_month_sample(values, dates, month, cal_start, cal_end)
    fit = fit_loglogistic_pwm_debug(sample["cal_values"])
    print("Fit: ",fit)
    if fit is None:
        return {"fit": None, "sample": sample}

    p_cal, z_cal = spei_from_fit(sample["cal_values"], fit["beta"], fit["loc"], fit["scale"])
    p_all, z_all = spei_from_fit(sample["all_values"], fit["beta"], fit["loc"], fit["scale"])

    return {
        "sample": sample,
        "fit": fit,
        "cdf_cal": p_cal,
        "spei_cal": z_cal,
        "cdf_all": p_all,
        "spei_all": z_all,
        "cal_mean": np.nanmean(z_cal),
        "cal_std": np.nanstd(z_cal, ddof=1) if np.sum(np.isfinite(z_cal)) > 1 else np.nan,
        "all_mean": np.nanmean(z_all),
        "all_std": np.nanstd(z_all, ddof=1) if np.sum(np.isfinite(z_all)) > 1 else np.nan,
    }


def plot_fit_diagnostic(cal_values, fit, output_path="fit_diagnostic.png"):
    x_emp, p_emp = empirical_plotting_positions(cal_values)
    x_grid = np.linspace(np.nanmin(x_emp), np.nanmax(x_emp), 400)
    p_fit = loglogistic_cdf(x_grid, fit["beta"], fit["loc"], fit["scale"])

    plt.figure(figsize=(6, 4))
    plt.scatter(x_emp, p_emp, s=20, label="Empirical")
    plt.plot(x_grid, p_fit, color="red", label="Fitted log-logistic")
    plt.xlabel("Accumulated water balance")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_spei_histogram(z_cal, month=None, output_path="spei_histogram.png"):
    z_cal = np.asarray(z_cal, dtype=float)
    z_cal = z_cal[np.isfinite(z_cal)]

    if z_cal.size == 0:
        raise ValueError("No finite SPEI values to plot.")

    # Bins: <= -2.0, then 0.5 steps, then >= 2.0
    bins = np.array([
        -np.inf, -2.0, -1.5, -1.0, -0.5,
         0.0,   0.5,  1.0,  1.5,  2.0, np.inf
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(z_cal, bins=bins, density=True, alpha=0.7, edgecolor="black")

    ax.axvline(np.nanmean(z_cal), color="red", linestyle="--",
               label=f"mean={np.nanmean(z_cal):.2f}")
    ax.axvline(0.0, color="black", linestyle=":")

    ax.set_title(f"SPEI calibration distribution month={month}")
    ax.set_xlabel("SPEI")
    ax.set_ylabel("Density")

    # Label the outer bins as inclusive tails
    ax.set_xticks([-2.5,-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xticklabels(["-inf", "<=-2.0", "-1.5", "-1.0", "-0.5",
                        "0.0", "0.5", "1.0", "1.5", ">=2.0", "+inf"])

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

