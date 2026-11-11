"""
Deterministic skill metrics for a seasonal SPEI forecast.

Design (as described by the user, not inferred from the headers):
  forecast : SPEI_Mod-Hargreaves_lead123_init10_1993-2025_Madagascar_res_...
             dims (number=25, init_year=33, lat=159, lon=90)
             init month = October
             SPEI evaluated over months 1-3 after init
             (lead n = init + n months)
             one AGGREGATED value per member/init_year -> verification is
             against the CENTRE month, December of the init year.
  obs      : SPEI_Mod-Hargreaves_3m_1993-2026_Madagascar
             dims (time, lat, lon), monthly, 3-month SPEI
             SPEI fit over the same years as the forecast.

Usable init years: the Nov-Jan window crosses into init_year+1, so the last
init whose window is fully inside the observation record is dropped
automatically (see `window_fully_observed`).

Metrics implemented here (deterministic only):
  - ensemble-mean error
  - RMSE, reported against the 1.0 no-skill reference for a standardised index
  - MAE
  - Pearson correlation and Spearman rank correlation, per grid point
  - ensemble spread
  - spread/RMSE ratio (calibration summary)

Output: the gridded scores are written to a NetCDF file (lat, lon) with one
variable per metric, plus a companion CSV of the flattened grid if requested.

Run:  python3 scratch/det_metrics.py
"""

import os
import numpy as np
import xarray as xr
from scipy import stats

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
START_YEAR = 1993
END_YEAR   = 2025

INIT_MONTH   = 10          # October
INIT_YEAR    = 1993        # first init year in the file
INIT_END_YEAR = 2025       # last init year in the file
LEADS        = (1, 2, 3)   # aggregate = Nov, Dec, Jan
LEAD_OFFSET  = 1           # lead n = init + n months  -> lead 1 = November

DIR_FCST = f"/ec/res4/scratch/ecme4047/C3S_seasonal/ecmwf51/monthly/init_{INIT_MONTH}/SPEI/"
PATH_FCST = DIR_FCST +\
        (f"SPEI_Mod-Hargreaves_lead123_init{INIT_MONTH}_1993-2025_Madagascar_"
         "res_ERA5-Land_qdm_ERA5-Land_MSWEP.nc")
DIR_OBS = "/ec/res4/scratch/ecme4047/ERA5-Land/SPEI/monthly/"
PATH_OBS  = DIR_OBS + "SPEI_Mod-Hargreaves_3m_1993-2026_Madagascar.nc"

# Aggregated field verifies against the last month of the window (January for Oct init),
# which is init month + 3.
VERIFY_MONTH_OFFSET = 3

PATH_OUT_NC  = DIR_FCST + \
        f"SPEI_Mod-Hargreaves_deterministic_scores_lead123_init{INIT_MONTH}_1993-2025_Madagascar.nc"

# figure filename stem, following the same convention as the NetCDF output
FIG_STEM = (f"SPEI_Mod-Hargreaves_{{score}}_lead123_init{INIT_MONTH}"
            f"_1993-2025_Madagascar")
WRITE_CSV    = False          # also write the flattened grid as CSV
WRITE_NC     = True          # gridded scores -> NetCDF

# figures
FIG_DIR    = "/ec/res4/scratch/ecme4047/figures/SPEI"
WRITE_FIGS = True

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
    """True when EVERY month of the window exists in the observation record.

    The window runs into init_year+1 (January), so the final inits are
    only partly covered by an observation record that ends in December.
    """
    return all(y * 100 + m in obs_months for y, m in window_months(init_year))


# ----------------------------------------------------------------------
# DETERMINISTIC SCORES
# ----------------------------------------------------------------------
def deterministic_scores(f, o, dim="time"):
    """Scores for ensemble f vs obs o, reduced over `dim`, per grid point.

    f : (number, time, lat, lon)
    o : (time, lat, lon)
    """
    fm = f.mean("number")
    err = fm - o                       # ensemble-mean error, per case

    # ---- correlation: Pearson and Spearman, NaN-safe ------------------
    m_corr = xr.corr(fm, o, dim=dim)

    def _spearman(a, b):
        a, b = np.asarray(a), np.asarray(b)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return np.nan
        return stats.spearmanr(a[m], b[m]).statistic

    rho = xr.apply_ufunc(
        _spearman, fm, o,
        input_core_dims=[[dim], [dim]], output_core_dims=[[]],
        vectorize=True, dask="allowed",
    )

    # ---- significance threshold for n cases ---------------------------
    # (the caller computes the critical r from the actual case count)

    return xr.Dataset({
        "eme":    err.mean(dim),
        "rmse":   np.sqrt((err ** 2).mean(dim)),
        "mae":    np.abs(err).mean(dim),
        "corr":   m_corr,
        "rho":    rho,
        "spread": f.std("number").mean(dim),
        "n_cases": np.isfinite(err).sum(dim),
    })


# ----------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.prop_cycle": matplotlib.cycler(
            color=["#2F5D9E", "#E07B39", "#3A9D5D", "#8E5EA2", "#6B6B6B"]),
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt


def _map(ax, da, title, cmap, label, vmin=None, vmax=None,
         diverging=False):
    """Draw one gridded field as a map with a colourbar."""
    kw = {}
    if diverging:
        lim = np.nanmax(np.abs(da.values))
        kw = dict(vmin=-lim, vmax=lim)
    elif vmin is not None:
        kw = dict(vmin=vmin, vmax=vmax)
    m = ax.pcolormesh(da.lon.values, da.lat.values, da.values,
                      cmap=cmap, shading="nearest", **kw)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    cb = ax.figure.colorbar(m, ax=ax, orientation="vertical",
                            fraction=0.032, pad=0.02)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return m


def plot_scores(sc, ratio, n_eff, r_crit, y0, y1):
    """Write the score maps and summary panels to FIG_DIR."""
    plt = _setup_mpl()
    os.makedirs(FIG_DIR, exist_ok=True)

    period = f"{y0}-{y1} ({n_eff} init years)"
    src_f = PATH_FCST.split("/")[-1]
    src_o = PATH_OBS.split("/")[-1]

    # ---------------- RMSE map -----------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    r = float(np.nanmean(sc["rmse"].values))
    _map(ax, sc["rmse"],
         f"RMSE of the ensemble-mean SPEI\ninit {INIT_MONTH:02d}, {period}",
         "binary", "RMSE (SPEI units)")
    fig.supxlabel(f"Domain mean {r:.2f}; no-skill reference 1.0. "
                  f"Source: {src_f}", x=0.01, ha="left", fontsize=8,
                  color="#777")
    fig.savefig(f"{FIG_DIR}/{FIG_STEM.format(score='rmse')}.png", dpi=200)
    plt.close(fig)

    # ---------------- correlation map ----------------------------------
    # Cells whose correlation exceeds the p=0.05 critical value are
    # outlined on the same map, so significance is visible without a
    # second plot.
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    corr = sc["corr"]
    sig = (corr > r_crit)
    frac = float(np.nanmean(sig.values))
    _map(ax, corr,
         f"Temporal correlation of ensemble-mean SPEI\n init {INIT_MONTH:02d}, "
         f"{period}",
         "RdBu_r", "Pearson r", diverging=True)

    # Mark the significant cells. Hatching is unusable at this grid
    # resolution: the cells are ~0.1 deg, far below one pixel at the
    # figure size, so hatch lines collapse into a solid tint. Draw a
    # stippled mask instead, one small dot per significant cell.
    from matplotlib.lines import Line2D
    lon2d, lat2d = np.meshgrid(corr.lon.values, corr.lat.values)
    msk = np.isfinite(corr.values) & sig.values
    ax.scatter(lon2d[msk], lat2d[msk], s=.6, c="#101010",
               marker="o", linewidths=0, zorder=4)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none",
                              markersize=2.5, color="#101010",
                              label=f"significant at p<0.05 (r > {r_crit:.2f})")],
              loc="lower left", fontsize=8, frameon=True, framealpha=0.9,
              borderpad=0.4)

    fig.savefig(f"{FIG_DIR}/{FIG_STEM.format(score='corr')}.png", dpi=200)
    plt.close(fig)

    # ---------------- spread / RMSE map --------------------------------
    fig, ax = plt.subplots(figsize=(5, 6), layout="constrained")
    rr = float(np.nanmean(ratio.values))
    _map(ax, ratio,
         f"Ensemble spread divided by RMSE\ninit {INIT_MONTH:02d}, {period}",
         "PuOr_r", "spread / RMSE (1 = calibrated)", vmin=0.8, vmax=1.2)
    fig.savefig(f"{FIG_DIR}/{FIG_STEM.format(score='spread_rmse')}.png", dpi=200)
    plt.close(fig)

    # ---------------- distributions ------------------------------------
    # sc has no spread_rmse_ratio (that variable is added later, in
    # build_output), so the third panel uses `ratio` directly.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, values, name, colour, xlabel in (
        (axes[0], sc["rmse"], "rmse", "#2F5D9E", "RMSE (SPEI units)"),
        (axes[1], sc["corr"], "corr", "#E07B39", "Pearson r"),
        (axes[2], ratio, "spread_rmse_ratio", "#3A9D5D", "spread / RMSE"),
    ):
        v = values.values.ravel()
        v = v[np.isfinite(v)]
        if name == "spread_rmse_ratio":
            v = v[v > 0]
        if not v.size:
            ax.set_visible(False)
            continue
        ax.hist(v, bins=30, color=colour, edgecolor="white", linewidth=0.4)
        ax.axvline(float(np.nanmedian(v)), color="#222", lw=1.4, ls="--",
                   label=f"median {np.nanmedian(v):.2f}")
        if name == "rmse":
            ax.axvline(1.0, color="#B33", lw=1.4, ls=":", label="no-skill 1.0")
        if name == "corr":
            ax.axvline(r_crit, color="#B33", lw=1.4, ls=":",
                       label=f"p=0.05 r = {r_crit:.2f}")
        if name == "spread_rmse_ratio":
            ax.axvline(1.0, color="#B33", lw=1.4, ls=":", label="calibrated 1.0")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Grid points")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Distribution of SPEI skill scores across "
                 f"{sc['rmse'].size} grid points\ninit {INIT_MONTH:02d}, "
                 f"{period}")
    fig.savefig(f"{FIG_DIR}/{FIG_STEM.format(score='distributions')}.png",
                dpi=200)
    plt.close(fig)


    print(f"\nwrote 4 figures to {FIG_DIR}/")
    for s in ("rmse", "corr", "spread_rmse", "distributions"):
        print(f"  {FIG_STEM.format(score=s)}.png")


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

    # ---- calendar -----------------------------------------------------
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

    # ---- build aligned arrays -----------------------------------------
    idx = [yrs.index(y) for y in usable]
    tgt = np.array([verify_month(y) for y in usable])

    print(fcst)
    f = fcst.isel(init_year=idx).assign_coords(time=("init_year", tgt))
    f = f.swap_dims({"init_year": "time"}).transpose(
        "number", "time", "lat", "lon")

    o = obs.sel(time=tgt).transpose("time", "lat", "lon")

    # the observation month actually pulled, as a guard against a
    # silent calendar misalignment
    got = month_index(o.time.values)
    want = np.array([int(str(t)[:4]) * 100 + int(str(t)[5:7]) for t in tgt])
    print(f"obs months pulled, first/last: {got[0]} / {got[-1]}"
          f"  {'OK' if np.array_equal(got, want) else 'MISMATCH'}")

    print(f"\nscoring {f.sizes['time']} init years x {f.sizes['number']} members"
          f" on {f.sizes['lat']}x{f.sizes['lon']} grid")
    print(f"forecast NaN fraction: {float(np.isnan(f.values).mean()):.4f}")

    # ---- scores --------------------------------------------------------
    sc = deterministic_scores(f, o)

    # spread-rmse ratio, computed only where both are finite and rmse>0
    ratio = (sc["spread"] / sc["rmse"]).where(sc["rmse"] > 1e-6)

    keys = ["eme", "rmse", "mae", "corr", "rho", "spread"]
    print("\n=== Domain-mean deterministic scores ===")
    summary = sc[keys].mean(["lat", "lon"]).to_pandas()
    summary["spread/rmse"] = float(ratio.mean(["lat", "lon"]))
    summary["rmse/1.0 (no-skill ref)"] = float(
        sc["rmse"].mean(["lat", "lon"]))
    print(summary.round(3).to_string())

    # ---- spatial distribution of correlation --------------------------
    n_pts = sc["corr"].size
    print(f"\n=== Spatial distribution (n_points = {n_pts}) ===")
    for name in ("corr", "rho"):
        v = sc[name].values.ravel()
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        p = {q: np.percentile(v, q) for q in (5, 25, 50, 75, 95)}
        print(f"{name:>5}: p5={p[5]:+.2f} p25={p[25]:+.2f} "
              f"median={p[50]:+.2f} p75={p[75]:+.2f} p95={p[95]:+.2f}")
        print(f"       fraction > 0    : {(v > 0).mean():.3f}")

    # significance: use the critical r for the number of cases
    n_eff = int(np.nanmax(sc["n_cases"].values))
    tcrit = stats.t.ppf(0.975, n_eff - 2)
    r_crit = tcrit / np.sqrt(tcrit ** 2 + n_eff - 2)
    frac_sig = float((sc["corr"].values > r_crit).mean())
    print(f"\nn_cases = {n_eff}; critical r at p=0.05 (two-sided) = {r_crit:.3f}")
    print(f"fraction of points with corr > r_crit : {frac_sig:.3f}")

    # ---- rmse against the standardised-index reference ----------------
    rmse_mean = float(sc["rmse"].mean(["lat", "lon"]))
    print(f"\nmean RMSE = {rmse_mean:.3f} -> "
          f"{'better' if rmse_mean < 1 else 'worse'} than the "
          f"no-skill reference of 1.0 for a standardised index")

    # ---- gridded output: NetCDF ----------------------------------------
    out = build_output(sc, usable, n_eff, r_crit, ratio)

    if WRITE_NC:
        out.to_netcdf(PATH_OUT_NC)
        print(f"\nwrote {PATH_OUT_NC}")
        print("  variables:", ", ".join(out.data_vars))
        print(f"  dims     : {dict(out.sizes)}")

    # ---- optional CSV --------------------------------------------------
    if WRITE_CSV:
        # flat one-row-per-grid-point table, scores only (no scalars repeated)
        df = sc.to_dataframe().reset_index().dropna(subset=["rmse"])
        df["spread_rmse_ratio"] = (sc["spread"] / sc["rmse"]).values.ravel()
        df = df[np.isfinite(df["spread_rmse_ratio"])]
        df["init_month"] = INIT_MONTH
        df["verify_month_offset"] = VERIFY_MONTH_OFFSET
        df["n_cases"] = n_eff
        df["r_crit_p05"] = r_crit
        csv_name = ("SPEI_Mod-Hargreaves_deterministic_scores_lead123_"
                    f"init{INIT_MONTH}_1993-2025_Madagascar.csv")
        df.to_csv(csv_name, index=False)
        print(f"wrote {csv_name} ({len(df)} rows)")

    # ---- figures --------------------------------------------------------
    if WRITE_FIGS:
        try:
            plot_scores(sc, ratio, n_eff, r_crit, usable[0], usable[-1])
        except Exception as e:                                # noqa: BLE001
            print(f"plotting failed: {e}")

    return out


def build_output(sc, usable, n_eff, r_crit, ratio):
    """Assemble the gridded score file with CF-style metadata."""
    out = sc.copy()

    # spread/RMSE ratio only makes sense where RMSE is non-zero
    out["spread_rmse_ratio"] = ratio
    out["corr_r_crit_p05"] = xr.full_like(sc["corr"], r_crit, dtype="float64")

    # scalar (non-gridded) provenance as coordinates, so they survive the
    # round-trip as attributes the user can read back
    out = out.assign_coords(
        n_cases=n_eff,
        r_crit_p05=r_crit,
        n_members=int(sc.sizes.get("number", 0)) or 25,
        init_month=INIT_MONTH,
        first_init_year=usable[0],
        last_init_year=usable[-1],
        verify_month_offset=VERIFY_MONTH_OFFSET,
    )

    # ---- variable metadata (CF-ish) ------------------------------------
    meta = {
        "eme":   ("Ensemble-mean error (forecast mean minus observed)", "-"),
        "rmse":  ("Root mean square error of the ensemble mean", "-"),
        "mae":   ("Mean absolute error of the ensemble mean", "-"),
        "corr":  ("Pearson correlation, ensemble mean vs observed, over time", "1"),
        "rho":   ("Spearman rank correlation, ensemble mean vs observed", "1"),
        "spread": ("Mean ensemble standard deviation across members", "-"),
        "spread_rmse_ratio": ("Ensemble spread divided by RMSE (1 = calibrated)", "1"),
        "corr_r_crit_p05": ("Critical Pearson r at p=0.05, two-sided", "1"),
        "n_cases": ("Number of init years with both forecast and obs present", "1"),
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
        title="Deterministic SPEI forecast skill scores",
        forecast=PATH_FCST,
        observation=PATH_OBS,
        init_month=f"{INIT_MONTH:02d}",
        forecast_window="leads 1-3",
        verification_month=" init + 3 months",
        scoring_period=f"{usable[0]}-{usable[-1]} ({n_eff} init years)",
        ensemble_members=str(int(sc.sizes.get("number", 25))),
        reference_note=("SPEI is standardised, so RMSE ~1.0 is the no-skill "
                        "reference for a 3-month index"),
        conventions="CF-1.8",
    )
    return out


if __name__ == "__main__":
    main()

