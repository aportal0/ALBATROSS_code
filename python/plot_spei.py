#!/usr/bin/env python3
"""
Plot ET0 and SPEI from monthly NetCDF files, plus SPEI forecasts.

- ET0 is plotted for the selected month/year.
- SPEI is assumed to be already accumulated on WINDOW months.
- SPEI forecasts come from a hindcast file (C3S seasonal, ECMWF 51) that holds
  one lead time per file, with an ensemble ("number") and an initialization-year
  coordinate ("init_year"). The ensemble mean or a single ensemble member can be
  plotted.
- SOURCE picks what is plotted: "obs" (ERA5-Land observations) or
  "forecast" (C3S seasonal). Both routes go through plot_spei_source().
- The script writes:
    1) ET0 figure
    2) SPEI figure
    3) combined ET0 + SPEI figure
    4) one SPEI forecast figure, ensemble mean or single member
    5) one SPEI forecast figure with the ensemble mean and single members
"""

import calendar
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator


# =============================================================================
# Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Source: observations or seasonal forecast
# -----------------------------------------------------------------------------
# "obs"      -> the monthly observation file (PATH_SPEI)
# "forecast" -> the C3S seasonal forecast file (PATH_FORECAST)
SOURCE = "obs"

# Both are also reachable directly, whatever SOURCE says:
#   plot_spei_source(source="forecast", window=3, year=2022)
#   plot_spei_source(source="obs", window=3, month=12, year=2022)


# -----------------------------------------------------------------------------
# Obs configuration
# -----------------------------------------------------------------------------
VAR_ET0 = "ET0"
VAR_SPEI = "SPEI"      # change if your variable is named differently
VAR_PR = "precipitation"
VAR_WB = "balance"

METHOD = "MH"             # "H" or "MH"
MONTH = 1
WINDOWS = [3]
YEAR = 2022              # year (int) or "clim" (str)
COUNTRY = "Madagascar"

PATH_ET0 = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/ET0/monthly/")
PATH_SPEI = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/SPEI/monthly/")
PATH_PR = Path("/home/alice/Desktop/UniBo/data/MSWEP/MSWEP_V316_test/Past/Monthly/regridded_ERA5-Land/")
PATH_WB = PATH_SPEI
PATH_OUT = Path("/home/alice/Desktop/UniBo/figures/SPEI/")


# -----------------------------------------------------------------------------
# Forecast configuration
# -----------------------------------------------------------------------------
# The method tag used inside the forecast file name ("Mod-Hargreaves",
# "Hargreaves", ...). Kept separate from METHOD_INFO labels because the file
# says "Mod-Hargreaves" while the label says "Modified Hargreaves".
FORECAST_METHOD_TAG = "Mod-Hargreaves"
# Short model tag for the output file names, e.g. ..._ensmean_ecmwf51_init102024_.
FORECAST_MODEL_TAG = "ecmwf51"
FORECAST_LEADS_TAG = "123"      # the "lead123" part of the file name

FORECAST_Y0 = 1993
FORECAST_Y1 = 2025

FORECAST_YEAR = 1999            # initialization year to plot
FORECAST_INIT_MONTH = 10        # "init10" -> October initialization

# Paths
PATH_FORECAST = Path(f"/home/alice/Desktop/UniBo/data/C3S_seasonal/{FORECAST_MODEL_TAG}/monthly/init_{FORECAST_INIT_MONTH}/SPEI/")

# File template. Placeholders: {method} {leads} {init} {y0} {y1} {country}
FILE_FORECAST = (
    "SPEI_{method}_lead{leads}_init{init}_{y0}-{y1}_{country}"
    "_res_ERA5-Land_qdm_ERA5-Land_MSWEP.nc"
)

# "mean"  -> ensemble mean
# "member" -> one ensemble member (FORECAST_MEMBER)
FORECAST_MODE = "mean"
FORECAST_MEMBER = 0

# The forecast window: SPEI accumulated over N months. The file name does not
# carry it, so it is set here for the labels. Set to None to leave the label
# generic ("SPEI forecast"). Keep it in step with the file you point at.
FORECAST_WINDOW = 3

# Colour scale for the ENSEMBLE MEAN figures: 0.2-wide bands from -0.9 to
# +0.9, so the mean's weak anomalies stay readable instead of sitting in one
# neutral cell. Values outside the range use the over/under colours. The
# observation figures and the individual members keep the full +/-2.5 scale.
FORECAST_COMPARE_SCALE = True

# Print the file structure once (dims, coords, variables) before plotting.
FORECAST_DEBUG = True

# The comparison figure: the ensemble mean plus these members, side by side.
# Drawn by main_forecast() unless FORECAST_COMPARE is off.
FORECAST_COMPARE = True
FORECAST_COMPARE_MEMBERS = [0, 1, 2, 3]
FORECAST_COMPARE_INCLUDE_MEAN = True


# =============================================================================
# Constants
# =============================================================================
MONTHS = list(calendar.month_abbr)[1:]

METHOD_INFO = {
    "H": {
        "label": "Hargreaves",
        "file_ET0": "ET0_Hargreaves_monthly_1993-2026_{country}.nc",
        "file_SPEI": "SPEI_Hargreaves_{window}m_1993-2026_{country}.nc",
        "file_precipitation": "precip_monthly_1993-2026_{country}.nc",
        "file_balance": "water-balance_Hargreaves_monthly_1993-2026_{country}.nc",
    },
    "MH": {
        "label": "Modified Hargreaves",
        "file_ET0": "ET0_Mod-Hargreaves_monthly_1993-2026_{country}.nc",
        "file_SPEI": "SPEI_Mod-Hargreaves_{window}m_1993-2026_{country}.nc",
        "file_precipitation": "precip_monthly_1993-2026_{country}.nc",
        "file_balance": "water-balance_Mod-Hargreaves_monthly_1993-2026_{country}.nc",
    },
}

# Dimension / coordinate names in the forecast file.
MEMBER_DIM = "number"        # ensemble-member dimension
INIT_YEAR_DIM = "init_year"  # initialization-year coordinate

# Set to the name of a lead dimension ONLY if your file ever holds several leads
# at once. With "lead123" files there is no lead dimension and this stays None.
LEAD_DIM = None


# =============================================================================
# File builders
# =============================================================================
def build_input_file(var, country, method, window=None):
    if method not in METHOD_INFO:
        raise ValueError(f"Unknown method: {method}. Use one of {list(METHOD_INFO)}")
    if var==VAR_ET0:
        return PATH_ET0 / METHOD_INFO[method][f"file_{var}"].format(
                country=country, window=window
        )
    elif var==VAR_SPEI:
        return PATH_SPEI / METHOD_INFO[method][f"file_{var}"].format(
                country=country, window=window
        )
    elif var == VAR_PR:
        return PATH_PR / METHOD_INFO[method][f"file_{var}"].format(
            country=country, window=window
        )
    elif var == VAR_WB:
        return PATH_WB / METHOD_INFO[method][f"file_{var}"].format(
            country=country, window=window
        )
    else:
        raise ValueError(f"Unknown variable: {var}")


def build_output_file(var, country, method, window, year, month):
    year_label = str(year)
    if var in [VAR_ET0, VAR_SPEI, VAR_WB]:
        return PATH_OUT / f"{var}_{method}_{window}m_{year_label}{month:02d}_{country}.png"
    elif var == VAR_PR:
        return PATH_OUT / f"{var}_{window}m_{year_label}{month:02d}_{country}.png"


def build_output_file_combo(country, method, window, year, month):
    year_label = str(year)
    return PATH_OUT / f"ET0-SPEI-PR-WB_{method}_{window}m_{year_label}{month:02d}_{country}.png"


def build_forecast_file(
    country=COUNTRY,
    method_tag=FORECAST_METHOD_TAG,
    leads_tag=FORECAST_LEADS_TAG,
    init_month=FORECAST_INIT_MONTH,
    y0=FORECAST_Y0,
    y1=FORECAST_Y1,
    path=PATH_FORECAST,
):
    """Path of the SPEI forecast file, e.g.
    SPEI_Mod-Hargreaves_lead123_init10_1993-2025_Madagascar_..._MSWEP.nc
    """
    name = FILE_FORECAST.format(
        method=method_tag,
        leads=leads_tag,
        init=f"{init_month:02d}",
        y0=y0,
        y1=y1,
        country=country,
    )
    return path / name


def build_output_file_obs(window, year, month, country=COUNTRY, method=METHOD):
    """Output path of a single-window observation figure."""
    return PATH_OUT / f"SPEI_{method}_{window}m_{year}{month:02d}_{country}.png"


def build_output_file_forecast(country, method, year, mode, member,
                               window=None, init_month=None,
                               model_tag=FORECAST_MODEL_TAG):
    """Output path of a forecast figure, e.g.
    SPEI_MH_3m_lead123_ensmean_ecmwf51_init102024_Madagascar.png
    """
    window_tag = f"{window}m" if window else "Nm"
    mode_tag = "ensmean" if mode == "mean" else f"member{member}"
    return PATH_OUT / (
        f"SPEI_{method}_{window_tag}_lead{FORECAST_LEADS_TAG}_{mode_tag}_"
        f"{model_tag}_init{init_month:02d}{year}_{country}.png"
    )


def build_output_file_forecast_comparison(country, method, year,
                                          window=None, init_month=None,
                                          model_tag=FORECAST_MODEL_TAG):
    """Output path of the mean-vs-members figure."""
    window_tag = f"{window}m" if window else "Nm"
    return PATH_OUT / (
        f"SPEI_{method}_{window_tag}_lead{FORECAST_LEADS_TAG}_"
        f"ensmean-members_{model_tag}_init{init_month:02d}{year}_{country}.png"
    )


# =============================================================================
# Loaders
# =============================================================================
def load_var(var, country, method, window):
    file_path = build_input_file(var, country, method, window)
    ds = xr.open_dataset(file_path)
    da = ds[var]
    if var==VAR_ET0:
        da = convert_day_to_month(da)
    return da


def convert_day_to_month(da, dim_time="time"):
    days = xr.DataArray(
        da[dim_time].dt.days_in_month,
        coords={dim_time: da[dim_time]},
        dims=[dim_time]
    )
    return da * days


def load_spei_forecast(file_path=None, var=VAR_SPEI, debug=FORECAST_DEBUG):
    """Open the forecast file and return the SPEI DataArray.

    If the variable is not called SPEI, the only data variable is taken. The
    dataset is kept open through `da` so the file is not closed early.
    """
    file_path = Path(file_path) if file_path is not None else build_forecast_file()
    if not file_path.exists():
        raise FileNotFoundError(
            f"Forecast file not found:\n  {file_path}\n"
            "Check PATH_FORECAST and the FILE_FORECAST template "
            "(method tag, lead tag, init month, year range, country)."
        )

    ds = xr.open_dataset(file_path)

    if debug:
        print_forecast_structure(ds, file_path)

    if var in ds.data_vars:
        da = ds[var]
    else:
        data_vars = list(ds.data_vars)
        if len(data_vars) != 1:
            raise KeyError(
                f"Variable '{var}' not in {file_path.name}. "
                f"Available variables: {data_vars}. "
                "Set VAR_SPEI to the right one, or set it to the only variable."
            )
        if debug:
            print(f"  note   : '{var}' not found, using '{data_vars[0]}'")
        da = ds[data_vars[0]]

    return da


# =============================================================================
# Forecast dimension handling
# =============================================================================
def print_forecast_structure(ds, file_path=None):
    """Print dimensions, coordinates and variables of the forecast file."""
    print("=" * 72)
    if file_path is not None:
        print(f"Forecast file: {Path(file_path).name}")
    print(f"  variables : {list(ds.data_vars)}")
    print(f"  coords    : {list(ds.coords)}")
    for name, size in ds.sizes.items():
        dim = ds[name] if name in ds.coords else None
        if dim is not None and dim.ndim == 1:
            values = np.asarray(dim.values)
            if np.issubdtype(values.dtype, np.datetime64):
                preview = [
                    np.datetime_as_string(v, unit="D") for v in values[:3]
                ] + (["..."] if values.size > 3 else [])
                preview += [np.datetime_as_string(values[-1], unit="D")] if values.size > 3 else []
            else:
                preview = list(values[:6]) + (["..."] if values.size > 6 else [])
            print(f"  dim {name:<10}: size={size:<5} values={preview}")
        else:
            print(f"  dim {name:<10}: size={size}")

    for name in ds.coords:
        coord = ds[name]
        if coord.ndim == 1 and name not in ds.sizes:
            values = np.asarray(coord.values)
            print(f"  coord {name:<8}: size={coord.size} values={list(values[:6])}"
                  + (" ..." if coord.size > 6 else ""))
    print("=" * 72)


def get_init_years(da):
    """Initialization years along INIT_YEAR_DIM, as integers."""
    values = np.asarray(da[INIT_YEAR_DIM].values).reshape(-1)
    if np.issubdtype(values.dtype, np.datetime64):
        return da[INIT_YEAR_DIM].dt.year.values.reshape(-1).astype(int)
    return values.astype(int)


def select_init_year(da, year):
    """Pick the field for initialization year `year` along INIT_YEAR_DIM.

    Returns (field, year_label). Raises ValueError when the year is absent,
    listing the years the file actually holds.
    """
    if INIT_YEAR_DIM not in da.dims:
        raise ValueError(
            f"Initialization coordinate '{INIT_YEAR_DIM}' not found in "
            f"dims={list(da.dims)}. Set INIT_YEAR_DIM to the right name."
        )

    years = get_init_years(da)
    mask = years == int(year)

    if not mask.any():
        raise ValueError(
            f"No initialization found for year={year} along '{INIT_YEAR_DIM}'.\n"
            f"Available years: {sorted(set(int(y) for y in years))}.\n"
            "Set FORECAST_YEAR accordingly."
        )

    index = int(np.flatnonzero(mask)[0])
    if mask.sum() > 1:
        print(
            f"  warning: {int(mask.sum())} entries for {year} along "
            f"'{INIT_YEAR_DIM}'; using the first one."
        )
    return da.isel({INIT_YEAR_DIM: index}), str(int(year))


# =============================================================================
# Time selectors
# =============================================================================
def select_month_year(da, month, year):
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    selected = da.where(da.time.dt.month == month, drop=True)
    if is_climatology_year(year):
        if selected.sizes["time"] == 0:
            raise ValueError(f"No data found for climatology of {MONTHS[month - 1]}.")
        result = selected.mean(dim="time", skipna=True)
        time_stamp = selected.time.values[0]
        return result, time_stamp
    selected = selected.where(selected.time.dt.year == year, drop=True)
    if selected.sizes["time"] == 0:
        raise ValueError(f"No data found for {MONTHS[month - 1]} {year}.")
    return selected.isel(time=0), selected.time[0].values


def is_climatology_year(year):
    return isinstance(year, str) and year.lower() == "clim"


def window_mean_or_sum_ending_month(da, month, window, year, l_stats):
    if window < 1:
        raise ValueError("window must be >= 1")
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    if window == 1:
        selected = da.where(da.time.dt.month == month, drop=True)
    else:
        if l_stats.lower() == "mean":
            rolling_stats = da.rolling(time=window, center=False, min_periods=window).mean()
        elif l_stats.lower() == "sum":
            rolling_stats = da.rolling(time=window, center=False, min_periods=window).sum()
        else:
            raise ValueError("stats must be 'mean' or 'sum'")
        selected = rolling_stats.where(rolling_stats.time.dt.month == month, drop=True)
    if is_climatology_year(year):
        if selected.sizes["time"] == 0:
            raise ValueError(
                f"No data found for climatology with a full {window}-month window ending in "
                f"{MONTHS[month - 1]}."
            )
        result = selected.mean(dim="time", skipna=True)
        time_stamp = selected.time.values[0]
        return result, time_stamp
    selected = selected.where(selected.time.dt.year == year, drop=True)
    if selected.sizes["time"] == 0:
        raise ValueError(
            f"No data with a full {window}-month window ending in "
            f"{MONTHS[month - 1]} {year}."
        )
    return selected.isel(time=0), selected.time[0].values


# =============================================================================
# Forecast selectors
# =============================================================================
def select_spei_forecast(da, year=FORECAST_YEAR, mode=FORECAST_MODE, member=FORECAST_MEMBER):
    """Extract the forecast field for one initialization year.

    The file holds a single lead time, so nothing is selected along a lead
    dimension: the field already is the one the lead in the file name refers to.

    Parameters
    ----------
    da : xarray.DataArray
        SPEI forecast with an ensemble dimension ('number') and an
        initialization-year coordinate ('init_year').
    year : int
        Which initialization year to use.
    mode : {"mean", "member"}
        Ensemble mean or a single member.
    member : int
        Ensemble-member value (as in the 'number' coordinate), used when
        mode == "member".

    Returns
    -------
    (result, info) where result is a 2-D DataArray and info a dict with
    'init_label' and 'mode_label'.
    """
    mode = str(mode).lower()
    if mode not in ("mean", "member"):
        raise ValueError("mode must be 'mean' or 'member'")

    selected, init_label = select_init_year(da, year)

    # --- a lead dimension, only if this file happens to have one --------
    if LEAD_DIM is not None and LEAD_DIM in selected.dims:
        selected = selected.isel({LEAD_DIM: 0})

    # --- ensemble -------------------------------------------------------
    if MEMBER_DIM not in selected.dims:
        raise ValueError(
            f"Ensemble dimension '{MEMBER_DIM}' not found in "
            f"dims={list(selected.dims)}. Set MEMBER_DIM to the right name."
        )

    mode_label = "ensemble mean"
    if mode == "member":
        selected = select_member(selected, member)
        mode_label = f"member {member}"
    else:
        selected = selected.mean(dim=MEMBER_DIM, skipna=True)

    if selected.ndim != 2:
        raise ValueError(
            f"Expected a 2-D field after selection, got dims={list(selected.dims)}. "
            "Check the forecast file structure."
        )

    info = {
        "init_label": init_label,
        "init_year": int(init_label) if str(init_label).isdigit() else year,
        "init_month": resolve_init_month(),
        "mode_label": mode_label,
    }
    return selected, info


def select_member(selected, member):
    """Pick one ensemble member from the 'number' coordinate."""
    coord = selected[MEMBER_DIM]
    values = np.asarray(coord.values)
    if not np.isin(member, values):
        raise ValueError(
            f"Member {member} not found along '{MEMBER_DIM}' "
            f"(size={coord.size}, values={list(values[:10])})."
        )
    return selected.sel({MEMBER_DIM: member})


# =============================================================================
# Titles
# =============================================================================
def build_period_label(month, year):
    if is_climatology_year(year):
        return f"{MONTHS[month - 1]} climatology"
    return f"{MONTHS[month - 1]} {year}"


def build_title_et0(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = build_period_label(month, year)
    window_label = "single month" if window == 1 else f"{window}-month mean"
    return f"ET0 - {method_label}\n{window_label}, {period}"


def build_title_spei(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = build_period_label(month, year)
    return f"SPEI-{window} - {method_label}\n{period}"


def build_title_pr(month, window, year):
    period = build_period_label(month, year)
    window_label = "single month" if window == 1 else f"{window}-month mean"
    return f"Precipitation\n {window_label}, {period}"


def build_title_wb(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = build_period_label(month, year)
    window_label = "single month" if window == 1 else f"{window}-month mean"
    return f"Water balance - {method_label}\n{window_label}, {period}"


def build_title_combo(method, month, window, year):
    method_label = METHOD_INFO[method]["label"]
    period = f"{MONTHS[month - 1]} {year}"
    window_label = "single month" if window == 1 else f"{window} months"
    return f"ET0, SPEI-{window}, P, WB - {method_label}\n{window_label}, {period}"


def spei_label(window=None):
    """Label for the SPEI variable, e.g. 'SPEI-3' or plain 'SPEI'."""
    return f"SPEI-{window}" if window else "SPEI"


def build_lead_label(leads_tag=FORECAST_LEADS_TAG):
    """Lead part of a title, e.g. 'lead 1,2,3' or 'lead 3'."""
    if not leads_tag:
        return "lead unknown"
    months = str(leads_tag)
    if len(months) > 1:
        return f"lead {', '.join(months)}"
    return f"lead {months}"


def build_title_spei_source(source, method, window=None, mode_label=None,
                            init_label=None, month=None, year=None):
    """Title shared by the observation and the forecast SPEI map."""
    method_label = METHOD_INFO[method]["label"]
    if str(source).lower() == "forecast":
        return (
            f"{spei_label(window)} forecast - {method_label}\n"
            f"{init_label}, {build_lead_label()}\n"
            f"{mode_label}"
        )
    return (
        f"{spei_label(window)} - {method_label}\n"
        f"{build_period_label(month, year)}, observations"
    )


def resolve_init_month(init_month=None):
    """Which month the forecast file is initialized in.

    Explicit argument first, then the init_<month> folder the file lives in,
    then FORECAST_INIT_MONTH.
    """
    if init_month is not None:
        return int(init_month)
    for part in reversed(PATH_FORECAST.parts):
        if part.startswith("init_"):
            tail = part[len("init_"):]
            if tail.isdigit() and 1 <= int(tail) <= 12:
                return int(tail)
    return int(FORECAST_INIT_MONTH)


def build_init_label(year, init_month=None, style="long"):
    """Initialization label, e.g. 'init Oct 2022' or 'init102022'."""
    month = resolve_init_month(init_month)
    if style == "tag":
        return f"init{month:02d}{year}"
    return f"init {MONTHS[month - 1]} {year}"


def build_init_sentence(year, init_month=None):
    """Standalone initialization line for a title."""
    month = resolve_init_month(init_month)
    return f"initialized {month:02d}/{year}"


def build_title_spei_forecast(method, info, window=None):
    """Title for a SPEI forecast map."""
    method_label = METHOD_INFO[method]["label"]
    return (
        f"{spei_label(window)} forecast - {method_label}\n"
        f"{build_init_sentence(info['init_year'], info['init_month'])}, "
        f"{build_lead_label()}\n"
        f"{info['mode_label']}"
    )


# =============================================================================
# Color settings
# =============================================================================
def get_color_settings_et0():
    vmin, vmax = 0, 200
    levels = MaxNLocator(nbins=10).tick_values(vmin, vmax)

    cmap = plt.colormaps["YlOrRd"].copy()
    cmap.set_over("#3e0000")

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


def get_color_settings_spei():
    levels = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5])

    colors = [
        "#b30000",  # extreme dry - deep red
        "#ff7f0e",  # bright orange
        "#ffb347",  # yellow-orange
        "#fff7bc",  # pale yellow
        "#f7f7f7",  # near normal
        "#d9f0d3",  # light green
        "#78c679",  # green
        "#238b45",  # deep green
        "#08306b"   # deper blue
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    # cmap = plt.colormaps["RdBu"].copy()
    # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


def get_color_settings_spei_comparison(nbands=9):
    """SPEI colours on the narrow ensemble-mean scale.

    The bands are 0.2 wide and sit strictly inside 0, which cannot be a band
    edge, so the palette is split in two: reds below zero, greens and blues
    above it, with no band crossing zero.
    """
    edges = np.array([-1., -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1.])

    colors = [
        "#b30000",  # -1 .. -0.8  extreme dry
        "#ff7f0e",  # -0.8 .. -0.6
        "#ffb347",  # -0.6 .. -0.4
        "#fff7bc",  # -0.4 .. -0.2
        "#f7f7f7",  # -0.2 .. +0.2  near normal
        "#d9f0d3",  # +0.2 .. +0.4
        "#78c679",  # +0.4 .. +0.6
        "#238b45",  # +0.6 .. +0.8
        "#08306b"   # +0.8 .. +1.  wettest  
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(edges, ncolors=cmap.N, clip=False)
    return cmap, norm


def get_color_settings_pr():
    vmin, vmax = 0, 500
    levels = MaxNLocator(nbins=10).tick_values(vmin, vmax)
    cmap = plt.colormaps["Blues"].copy()
    cmap.set_over("k")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


def get_color_settings_wb():
    levels = np.array([-300, -250, -200, -150, -100, -50, -25, 25, 50, 100, 150, 200, 250, 300])
    cmap = plt.colormaps["BrBG"].copy()
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm


# =============================================================================
# Plot helpers
# =============================================================================
def _get_lon_lat_vals(da):
    lon_name = "lon" if "lon" in da.coords else "longitude"
    lat_name = "lat" if "lat" in da.coords else "latitude"

    lon = np.asarray(da[lon_name].values)
    lat = np.asarray(da[lat_name].values)
    vals = np.squeeze(np.asarray(da.values))

    return lon, lat, vals


def _format_map(ax, title):
    ax.set_title(title)
    ax.coastlines(resolution="50m", linewidth=0.6)

    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="0.5",
        alpha=0.5,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False


# =============================================================================
# Plot single-variable figures
# =============================================================================
def plot_et0(da, method, month, window, year, outpath):
    title = build_title_et0(method, month, window, year)

    lon, lat, vals = _get_lon_lat_vals(da)
    cmap, norm = get_color_settings_et0()

    fig, ax = plt.subplots(
        figsize=(4.4, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im = ax.pcolormesh(
        lon,
        lat,
        vals,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )

    _format_map(ax, title)

    fig.colorbar(
        im,
        ax=ax,
        label=f"ET0 ({window}-month mean, mm)",
        shrink=0.8,
        extend="max",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_spei(da, method, month, window, year, outpath):
    title = build_title_spei(method, month, window, year)

    lon, lat, vals = _get_lon_lat_vals(da)
    cmap, norm = get_color_settings_spei()

    fig, ax = plt.subplots(
        figsize=(4.4, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im = ax.pcolormesh(
        lon,
        lat,
        vals,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )

    _format_map(ax, title)

    fig.colorbar(
        im,
        ax=ax,
        label=f"SPEI-{window} (-)",
        shrink=0.8,
        extend="both",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_pr(da, month, window, year, outpath):
    title = build_title_pr(month, window, year)

    lon, lat, vals = _get_lon_lat_vals(da)
    cmap, norm = get_color_settings_pr()

    fig, ax = plt.subplots(
        figsize=(4.4, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im = ax.pcolormesh(
        lon,
        lat,
        vals,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )

    _format_map(ax, title)

    fig.colorbar(
        im,
        ax=ax,
        label=f"P ({window}-month mean, mm)",
        shrink=0.8,
        extend="max",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_wb(da, method, month, window, year, outpath):
    title = build_title_wb(method, month, window, year)

    lon, lat, vals = _get_lon_lat_vals(da)
    cmap, norm = get_color_settings_wb()

    fig, ax = plt.subplots(
        figsize=(4.4, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im = ax.pcolormesh(
        lon,
        lat,
        vals,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )

    _format_map(ax, title)

    fig.colorbar(
        im,
        ax=ax,
        label=f"P - ET0 ({window}-month mean, mm)",
        shrink=0.8,
        extend="both",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# =============================================================================
# Plot SPEI forecast
# =============================================================================
def plot_spei_forecast(da, method, info, outpath, window=None):
    """One map: the SPEI forecast, ensemble mean or a single member.

    The ensemble mean is drawn on the narrow scale (FORECAST_COMPARE_SCALE),
    a single member on the full one.
    """
    title = build_title_spei_forecast(method, info, window)

    narrow = info["mode_label"] == "ensemble mean" and FORECAST_COMPARE_SCALE
    cmap, norm = (
        get_color_settings_spei_comparison() if narrow else get_color_settings_spei()
    )

    lon, lat, vals = _get_lon_lat_vals(da)

    fig, ax = plt.subplots(
        figsize=(4.4, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im = ax.pcolormesh(
        lon,
        lat,
        vals,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )

    _format_map(ax, title)

    # Same bar style for the mean and for a member: identical except for the
    # scale it carries.
    fig.colorbar(
        im,
        ax=ax,
        label=f"{spei_label(window)} (-)" + ("\nensemble mean" if narrow else ""),
        shrink=0.8,
        extend="both",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_spei_source(
    source=None,
    window=FORECAST_WINDOW,
    month=None,
    year=None,
    mode=FORECAST_MODE,
    member=FORECAST_MEMBER,
    outpath=None,
    file_path=None,
):
    """Plot SPEI from either the observations or the seasonal forecast.

    source : "obs" | "forecast" | None
        None falls back to SOURCE in the configuration block.
    window : int
        SPEI accumulation window, used in the labels and the output name.
    month, year : int
        Observations only: which month of which year to map. They default to
        MONTH and YEAR; a "clim" year gives the month's climatology.
    mode, member : str, int
        Forecast only: ensemble mean or a single member.
    outpath : Path or None
        Defaults to the same name the matching writer uses.

    Returns the output path. The observations are plotted as stored in the
    file, with no extra rolling, and the printed ending date says which
    accumulation the field is.
    """
    source = (SOURCE if source is None else source).lower()
    if source not in ("obs", "forecast"):
        raise ValueError("source must be 'obs' or 'forecast'")

    if source == "forecast":
        da = load_spei_forecast(file_path=file_path)
        field, info = select_spei_forecast(
            da,
            year=FORECAST_YEAR if year is None else year,
            mode=mode,
            member=member,
        )
        title = build_title_spei_source(
            "forecast", METHOD, window=window,
            mode_label=info["mode_label"], init_label=info["init_label"],
        )
        if outpath is None:
            outpath = build_output_file_forecast(
                COUNTRY, METHOD, info["init_year"], mode, member, window,
                init_month=info["init_month"],
            )
        print_summary_forecast(outpath, info, field, window=window)
    else:
        month = MONTH if month is None else month
        year = YEAR if year is None else year
        da = load_var(VAR_SPEI, COUNTRY, METHOD, window)
        field, time_stamp = select_month_year(da, month, year)
        title = build_title_spei_source(
            "obs", METHOD, window=window, month=month, year=year
        )
        if outpath is None:
            outpath = build_output_file_obs(window, year, month)
        print_summary(outpath, f"SPEI-{window}", time_stamp, field, month, year)

    lon, lat, vals = _get_lon_lat_vals(field)
    cmap, norm = get_color_settings_spei()

    fig, ax = plt.subplots(
        figsize=(4.4, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im = ax.pcolormesh(
        lon,
        lat,
        vals,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )

    _format_map(ax, title)

    fig.colorbar(
        im,
        ax=ax,
        label=f"{spei_label(window)} (-)",
        shrink=0.8,
        extend="both",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    if source == "forecast":
        da.close() if hasattr(da, "close") else None
    return outpath


def collect_forecast_panels(da, members, include_mean=True, year=FORECAST_YEAR):
    """One field per panel: ('mean'|member value, field)."""
    panels = []
    if include_mean:
        field, _ = select_spei_forecast(da, year=year, mode="mean")
        panels.append(("mean", field))
    for member in members:
        field, _ = select_spei_forecast(da, year=year, mode="member", member=member)
        panels.append((member, field))
    return panels


def forecast_panel_title(member, mean_label="Ensemble mean"):
    return mean_label if member == "mean" else f"Member {member}"


def plot_spei_forecast_comparison(
    da,
    method,
    year,
    members=None,
    include_mean=True,
    outpath=None,
    window=None,
    wspace=0.6,
    bar_width=0.022,
    bar_pad=0.03,
):
    """Ensemble mean and/or single members side by side.

    `members` defaults to FORECAST_COMPARE_MEMBERS, which is what
    FORECAST_COMPARE_MEAN is added to when it is on.

    The ensemble mean is drawn on the narrow scale and each member on the full
    one, so the figure carries up to two scales and each scale gets its own
    colourbar, built the same way for both: for the mean the arrow marks values
    above +0.9, for a member above +2.5.

    Every panel keeps the same width whether or not it carries a bar: the bars
    are drawn in axes of their own, placed in the gutter to the right of the
    panel they describe. Panels therefore stay the same shape and line up in a
    row, which narrowing one of them in place would not do.
    """
    if members is None:
        members = list(FORECAST_COMPARE_MEMBERS)
    members = list(members)
    if not members:
        raise ValueError("No ensemble members selected for the summary figure.")

    panels = collect_forecast_panels(da, members, include_mean=include_mean, year=year)
    if not panels:
        raise ValueError(
            "Nothing to plot: give at least one member or set include_mean=True."
        )

    # One scale per bar: the mean can have its own, the members share the other.
    scales = [
        ("ensemble mean", get_color_settings_spei_comparison()),
        ("members", get_color_settings_spei()),
    ]

    groups = {}
    for member, field in panels:
        is_mean = member == "mean"
        index = 0 if (is_mean and FORECAST_COMPARE_SCALE) else 1
        groups.setdefault(index, []).append((member, field))

    ncols = len(panels)
    panel_width = 4.0
    panel_height = 6.0
    bar_height = 0.02          # bar thickness, as a fraction of figure height
    bar_span = 0.12            # how far the ends overhang their panel, per side

    # The bars go underneath the panels, so the maps keep the full height and
    # the figure is taller by the strip the bars need.
    fig_height = panel_height * (1.0 + bar_height) + 1.1
    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(panel_width * ncols, fig_height),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.atleast_1d(axes)

    for ax, (member, field) in zip(axes, panels):
        is_mean = member == "mean"
        cmap, norm = scales[0 if (is_mean and FORECAST_COMPARE_SCALE) else 1][1]

        lon, lat, vals = _get_lon_lat_vals(field)
        ax.pcolormesh(
            lon,
            lat,
            vals,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
        )
        _format_map(ax, f"{spei_label(window)} - {forecast_panel_title(member)}")

    # One bar per scale, drawn the same way for the mean and for the members:
    # horizontal, same length, same thickness, each in axes of its own in the
    # strip below the first panel that uses it. Its ticks and label sit below
    # the bar, never on top of a map.
    fig.canvas.draw()
    fig.subplots_adjust(bottom=0.28)
    for index in (0, 1):
        group = groups.get(index)
        if not group:
            continue
        label, (cmap, norm) = scales[index]
        target = axes[panels.index(group[0])]
        position = target.get_position()

        bar_ax = fig.add_axes([0.0, 0.0, 0.1, bar_height])
        bar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=bar_ax,
            orientation="horizontal",
            extend="both",
        )
        bar.outline.set_visible(False)

        # A full bar thickness plus a margin below the panel, so neither the
        # bar nor its tick labels reach the map above it. In width the bar
        # overhangs the panel by bar_span per side, so it reads wider than the
        # map it belongs to without thinning the map or crowding a neighbour.
        margin = 0.09
        bar_ax.set_position(
            [
                position.x0 - position.width * bar_span,
                position.y0 - bar_height - margin,
                position.width * (1.0 + 2.0 * bar_span),
                bar_height,
            ]
        )
        bar.set_label(f"{spei_label(window)} (-), {label}")

    fig.suptitle(
        f"{spei_label(window)} forecast - {METHOD_INFO[method]['label']}\n"
        f"{build_init_sentence(year)}, {build_lead_label()}"
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Plot combined figure
# =============================================================================
def plot_et0_spei_pr_wb(et0, spei, pr, wb, method, month, window, year, outpath):
    title = build_title_combo(method, month, window, year)

    lon_et0, lat_et0, vals_et0 = _get_lon_lat_vals(et0)
    lon_spei, lat_spei, vals_spei = _get_lon_lat_vals(spei)
    lon_pr, lat_pr, vals_pr = _get_lon_lat_vals(pr)
    lon_wb, lat_wb, vals_wb = _get_lon_lat_vals(wb)

    cmap_et0, norm_et0 = get_color_settings_et0()
    cmap_spei, norm_spei = get_color_settings_spei()
    cmap_pr, norm_pr = get_color_settings_pr()
    cmap_wb, norm_wb = get_color_settings_wb()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    im0 = axes[0, 0].pcolormesh(
        lon_spei, lat_spei, vals_spei,
        transform=ccrs.PlateCarree(),
        cmap=cmap_spei, norm=norm_spei
    )
    _format_map(axes[0, 0], f"SPEI-{window}")

    im1 = axes[0, 1].pcolormesh(
        lon_et0, lat_et0, vals_et0,
        transform=ccrs.PlateCarree(),
        cmap=cmap_et0, norm=norm_et0
    )
    _format_map(axes[0, 1], f"ET0 ({window}-month mean)")

    im2 = axes[1, 0].pcolormesh(
        lon_pr, lat_pr, vals_pr,
        transform=ccrs.PlateCarree(),
        cmap=cmap_pr, norm=norm_pr
    )
    _format_map(axes[1, 0], f"P ({window}-month mean)")

    im3 = axes[1, 1].pcolormesh(
        lon_wb, lat_wb, vals_wb,
        transform=ccrs.PlateCarree(),
        cmap=cmap_wb, norm=norm_wb
    )
    _format_map(axes[1, 1], "WB = P - ET0")

    fig.suptitle(title)

    fig.colorbar(im0, ax=axes[0, 0], label=f"SPEI-{window} (-)", shrink=0.8, extend="both")
    fig.colorbar(im1, ax=axes[0, 1], label="ET0 (mm/month)", shrink=0.8, extend="max")
    fig.colorbar(im2, ax=axes[1, 0], label="P (mm/month)", shrink=0.8, extend="max")
    fig.colorbar(im3, ax=axes[1, 1], label="WB (mm/month)", shrink=0.8, extend="both")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# =============================================================================
# Summary
# =============================================================================
def print_summary(file_out, label, time_stamp, result, month=None, year=None):
    lon_name = "lon" if "lon" in result.coords else "longitude"
    lat_name = "lat" if "lat" in result.coords else "latitude"
    print(f"Wrote {file_out}")
    print(f"  field  : {label}")
    if month is not None and year is not None and is_climatology_year(year):
        print(f"  period : {MONTHS[month - 1]} climatology")
    else:
        print(f"  ending : {np.datetime_as_string(time_stamp, unit='D')}")
    print(
        f"  shape  : "
        f"{lat_name}={result.sizes[lat_name]}, {lon_name}={result.sizes[lon_name]}"
    )
    print(
        f"  range  : "
        f"{float(np.nanmin(result.values)):.2f} .. "
        f"{float(np.nanmax(result.values)):.2f}"
    )


def print_summary_forecast(file_out, info, result, window=None):
    lon_name = "lon" if "lon" in result.coords else "longitude"
    lat_name = "lat" if "lat" in result.coords else "latitude"
    print(f"Wrote {file_out}")
    print(f"  field  : {spei_label(window)} forecast, {info['mode_label']}")
    print(f"  init   : {MONTHS[info['init_month'] - 1]} {info['init_year']} "
          f"(init{info['init_month']:02d}/{info['init_year']})")
    print(f"  lead   : {build_lead_label()}")
    print(f"  shape  : {lat_name}={result.sizes[lat_name]}, {lon_name}={result.sizes[lon_name]}")
    print(
        f"  range  : "
        f"{float(np.nanmin(result.values)):.2f} .. "
        f"{float(np.nanmax(result.values)):.2f}"
    )


# =============================================================================
# Main
# =============================================================================
def main():
    """Everything in one run, driven by SOURCE.

    SOURCE = "obs"      -> the observation figures
    SOURCE = "forecast" -> the forecast figure, plus the member comparison
                           figure when FORECAST_COMPARE is on
    """
    if str(SOURCE).lower() == "forecast":
        main_forecast()
    else:
        main_observations()


def main_observations():
    PATH_OUT.mkdir(parents=True, exist_ok=True)

    for window in WINDOWS:
        file_out_et0 = build_output_file(VAR_ET0, COUNTRY, METHOD, window, YEAR, MONTH)
        file_out_spei = build_output_file(VAR_SPEI, COUNTRY, METHOD, window, YEAR, MONTH)
        file_out_pr = build_output_file(VAR_PR, COUNTRY, METHOD, window, YEAR, MONTH)
        file_out_wb = build_output_file(VAR_WB, COUNTRY, METHOD, window, YEAR, MONTH)
        file_out_combo = build_output_file_combo(COUNTRY, METHOD, window, YEAR, MONTH)

        da_et0 = load_var(VAR_ET0, COUNTRY, METHOD, window)
        da_spei = load_var(VAR_SPEI, COUNTRY, METHOD, window)
        da_pr = load_var(VAR_PR, COUNTRY, METHOD, window)
        da_wb = load_var(VAR_WB, COUNTRY, METHOD, window)

        result_et0, time_et0 = window_mean_or_sum_ending_month(da_et0, MONTH, window, YEAR, "mean")
        result_spei, time_spei = select_month_year(da_spei, MONTH, YEAR)
        result_pr, time_pr = window_mean_or_sum_ending_month(da_pr, MONTH, window, YEAR, "mean")
        result_wb, time_wb = window_mean_or_sum_ending_month(da_wb, MONTH, window, YEAR, "mean")

        plot_et0(result_et0, METHOD, MONTH, window, YEAR, file_out_et0)
        plot_spei(result_spei, METHOD, MONTH, window, YEAR, file_out_spei)
        plot_pr(result_pr, MONTH, window, YEAR, file_out_pr)
        plot_wb(result_wb, METHOD, MONTH, window, YEAR, file_out_wb)
        plot_et0_spei_pr_wb(result_et0, result_spei, result_pr, result_wb, METHOD, MONTH, window, YEAR, file_out_combo)

        print_summary(file_out_et0, f"ET0-{window}", time_et0, result_et0, MONTH, YEAR)
        print_summary(file_out_spei, f"SPEI-{window}", time_spei, result_spei, MONTH, YEAR)
        print_summary(file_out_pr, f"P-{window}", time_pr, result_pr, MONTH, YEAR)
        print_summary(file_out_wb, f"WB-{window}", time_wb, result_wb, MONTH, YEAR)
        print(f"Wrote {file_out_combo}")


def main_forecast(
    year=FORECAST_YEAR,
    mode=FORECAST_MODE,
    member=FORECAST_MEMBER,
    window=FORECAST_WINDOW,
    compare=None,
    members=None,
    file_path=None,
):
    """Plot the SPEI forecast for one initialization year.

    Writes one figure — the ensemble mean or a single member, depending on
    `mode` — and, when `compare` is on, a second figure with the ensemble mean
    and several members side by side. The lead time is fixed by the file
    (FORECAST_LEADS_TAG) and appears in the titles.
    """
    PATH_OUT.mkdir(parents=True, exist_ok=True)

    compare = FORECAST_COMPARE if compare is None else compare
    da = load_spei_forecast(file_path=file_path)

    # The chosen field: ensemble mean or one member.
    field, info = select_spei_forecast(da, year=year, mode=mode, member=member)
    init_month = resolve_init_month()
    outpath = build_output_file_forecast(
        COUNTRY, METHOD, year, mode, member, window, init_month=init_month
    )
    plot_spei_forecast(field, METHOD, info, outpath, window=window)
    print_summary_forecast(outpath, info, field, window=window)

    # Optional: ensemble mean against individual members.
    if compare:
        outpath_cmp = build_output_file_forecast_comparison(
            COUNTRY, METHOD, year, window, init_month=init_month
        )
        plot_spei_forecast_comparison(
            da,
            METHOD,
            year,
            members=members,
            include_mean=FORECAST_COMPARE_INCLUDE_MEAN,
            outpath=outpath_cmp,
            window=window,
        )
        print(f"Wrote {outpath_cmp}")
        print(
            "  field  : SPEI forecast, "
            + ("ensemble mean + " if FORECAST_COMPARE_INCLUDE_MEAN else "")
            + f"members {list(FORECAST_COMPARE_MEMBERS) if members is None else list(members)}"
        )

    da.close() if hasattr(da, "close") else None


if __name__ == "__main__":
    main()

