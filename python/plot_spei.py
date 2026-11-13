#!/usr/bin/env python3
"""
Plot ET0 and SPEI from monthly NetCDF files, plus SPEI forecasts.

- ET0 is plotted for the selected month/year.
- SPEI is assumed to be already accumulated on WINDOW months.
- SPEI forecasts come from a hindcast/forecast file with an
  initialization date and a lead time; the ensemble mean or a single
  ensemble member can be plotted.
- The script writes:
    1) ET0 figure
    2) SPEI figure
    3) combined ET0 + SPEI figure
    4) one SPEI forecast figure per lead time
    5) one SPEI forecast figure with all leads side by side
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
VAR_ET0 = "ET0"
VAR_SPEI = "SPEI"      # change if your variable is named differently
VAR_PR = "precipitation"
VAR_WB = "balance"

METHOD = "MH"             # "H" or "MH"
MONTH = 12
WINDOWS = [1,3,6,12]
YEAR = 2022             # year (int) or "clim" (str)
COUNTRY = "Madagascar"

PATH_ET0 = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/ET0/monthly/")
PATH_SPEI = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/SPEI/monthly/")
PATH_PR = PATH_ET0
PATH_WB = PATH_SPEI
PATH_OUT = Path("/home/alice/Desktop/UniBo/figures/SPEI/")

# -----------------------------------------------------------------------------
# Forecast configuration
# -----------------------------------------------------------------------------
PATH_FORECAST = Path("/home/alice/Desktop/UniBo/data/ERA5-Land/SPEI/forecast/")

# File template. Placeholders: {method} {leads} {init} {y0} {y1} {country}
FILE_FORECAST = (
    "SPEI_{method}_lead{leads}_init{init}_{y0}-{y1}_{country}"
    "_res_ERA5-Land_qdm_ERA5-Land_MSWEP.nc"
)

# The method tag used inside the forecast file name ("Mod-Hargreaves",
# "Hargreaves", ...). Kept separate from METHOD_INFO labels because the file
# says "Mod-Hargreaves" while the label says "Modified Hargreaves".
FORECAST_METHOD_TAG = "Mod-Hargreaves"
FORECAST_LEADS_TAG = "123"      # the "lead123" part of the file name

FORECAST_Y0 = 1993
FORECAST_Y1 = 2025

FORECAST_YEAR = 2022            # initialization year to plot
FORECAST_INIT_MONTH = 10        # "init10" -> October initialization
FORECAST_LEADS = [1, 2, 3]      # lead months to plot

# "mean"  -> ensemble mean
# "member" -> one ensemble member (FORECAST_MEMBER)
FORECAST_MODE = "mean"
FORECAST_MEMBER = 0

# The forecast window: SPEI accumulated over N months (the file suffix does not
# always carry it, so it is set here only for labels). Set to None to leave the
# label generic ("SPEI forecast").
FORECAST_WINDOW = 3

# Map each lead to the valid calendar month of the target field
# (valid = init month + lead). If False, the lead index is used as-is and the
# title shows only the lead.
FORECAST_USE_VALID_MONTH = True

# Print the file structure once (dims, coords, variables) before plotting.
FORECAST_DEBUG = True

# If the ensemble-member dimension cannot be found, still plot the ensemble
# mean instead of raising.
FORECAST_FALLBACK_TO_MEAN = True


# =============================================================================
# Constants
# =============================================================================
MONTHS = list(calendar.month_abbr)[1:]

METHOD_INFO = {
    "H": {
        "label": "Hargreaves",
        "file_ET0": "ET0_Hargreaves_monthly_1993-2024_{country}.nc",
        "file_SPEI": "SPEI_Hargreaves_{window}m_1993-2024_{country}.nc",
        "file_precipitation": "precip_monthly_1993-2024_{country}.nc",
        "file_balance": "water-balance_Hargreaves_monthly_1993-2024_{country}.nc",
    },
    "MH": {
        "label": "Modified Hargreaves",
        "file_ET0": "ET0_Mod-Hargreaves_monthly_1993-2024_{country}.nc",
        "file_SPEI": "SPEI_Mod-Hargreaves_{window}m_1993-2024_{country}.nc",
        "file_precipitation": "precip_monthly_1993-2024_{country}.nc",
        "file_balance": "water-balance_Mod-Hargreaves_monthly_1993-2024_{country}.nc",
    },
}

# Candidate names for the dimensions of a forecast file. The first one that is
# present in the dataset wins; add your own naming here if yours differs.
MEMBER_DIM_CANDIDATES = (
    "member", "members", "ensemble", "ens", "realization", "realizations",
    "number", "r", "m",
)
LEAD_DIM_CANDIDATES = (
    "lead", "lead_time", "leadtime", "forecast_time", "step", "lead_month",
    "lead_months", "l",
)
INIT_DIM_CANDIDATES = (
    "init", "init_time", "initialization", "initialization_time", "time",
    "forecast_reference_time", "reference_time", "s", "date",
)


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


def build_output_file_forecast(country, method, lead, year, init_month, mode, window=None):
    window_tag = f"SPEI{window}" if window else "SPEI"
    mode_tag = "ensmean" if mode == "mean" else "ensmem"
    return PATH_OUT / (
        f"{window_tag}-forecast_{method}_{mode_tag}_"
        f"init{init_month:02d}{year}_lead{lead}_{country}.png"
    )


def build_output_file_forecast_panels(country, method, year, init_month, mode, window=None):
    window_tag = f"SPEI{window}" if window else "SPEI"
    mode_tag = "ensmean" if mode == "mean" else "ensmem"
    return PATH_OUT / (
        f"{window_tag}-forecast_{method}_{mode_tag}_"
        f"init{init_month:02d}{year}_allleads_{country}.png"
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
def find_dim(da, candidates, kind="dimension"):
    """First dimension of `da` whose name matches one of `candidates`."""
    for name in candidates:
        if name in da.dims:
            return name
    return None


def dim_name_from_coords(ds, candidates):
    """Same as find_dim but for dataset coords/variables (1-D only)."""
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    return None


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


def get_init_dates(da, init_dim):
    """Datetime64 array of initialization dates along `init_dim`."""
    init = da[init_dim]
    values = np.asarray(init.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values
    # Non-datetime init axis (e.g. integer year): keep the raw values.
    return values


def get_init_index(da, init_dim, year, init_month=None):
    """Position of the initialization along `init_dim` for a year (and month).

    Returns (index, label). Raises ValueError when nothing matches, listing the
    initializations that are available.
    """
    values = get_init_dates(da, init_dim)

    if np.issubdtype(np.asarray(values).dtype, np.datetime64):
        times = da[init_dim].dt
        mask = times.year == year
        if init_month is not None:
            mask = mask & (times.month == init_month)
        mask = np.asarray(mask.values, dtype=bool)
    else:
        # Numeric init axis: assume it encodes the year.
        mask = np.asarray(values) == year

    if not mask.any():
        if np.issubdtype(np.asarray(values).dtype, np.datetime64):
            available = sorted(
                {f"{int(y)}-{int(m):02d}" for y, m in
                 zip(da[init_dim].dt.year.values, da[init_dim].dt.month.values)}
            )
        else:
            available = [str(v) for v in np.asarray(values)]
        raise ValueError(
            f"No initialization found for year={year}"
            + (f", month={init_month}" if init_month is not None else "")
            + f" along '{init_dim}'.\nAvailable initializations: {available}"
            + "\nSet FORECAST_YEAR / FORECAST_INIT_MONTH accordingly."
        )

    index = int(np.flatnonzero(mask)[0])
    label = format_init_label(da[init_dim].isel({init_dim: index}))
    return index, label


def format_init_label(init_value):
    """Human-readable initialization date, e.g. 'Oct 2022'."""
    values = np.asarray(init_value.values)
    if values.size and np.issubdtype(values.dtype, np.datetime64):
        month = int(init_value.dt.month.values.reshape(-1)[0])
        year = int(init_value.dt.year.values.reshape(-1)[0])
        return f"{MONTHS[month - 1]} {year}"
    return str(values.reshape(-1)[0])


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
def select_spei_forecast(
    da,
    year=FORECAST_YEAR,
    init_month=FORECAST_INIT_MONTH,
    lead=1,
    mode=FORECAST_MODE,
    member=FORECAST_MEMBER,
):
    """Extract the forecast field for one initialization year and lead.

    Parameters
    ----------
    da : xarray.DataArray
        SPEI forecast with an ensemble-member dim, a lead dim and an
        initialization dim.
    year, init_month : int
        Which initialization to use.
    lead : int
        Lead time in months (as written in the file, 1-based lead index).
    mode : {"mean", "member"}
        Ensemble mean or a single member.
    member : int
        Member index or member coordinate value, used when mode == "member".

    Returns
    -------
    (result, info) where result is a 2-D DataArray and info a dict with
    'init_label', 'lead', 'valid_label', 'mode_label'.
    """
    mode = str(mode).lower()
    if mode not in ("mean", "member"):
        raise ValueError("mode must be 'mean' or 'member'")

    init_dim = find_dim(da, INIT_DIM_CANDIDATES)
    if init_dim is None:
        raise ValueError(
            f"No initialization dimension found in dims={list(da.dims)}. "
            f"Add its name to INIT_DIM_CANDIDATES."
        )
    lead_dim = find_dim(da, LEAD_DIM_CANDIDATES)
    member_dim = find_dim(da, MEMBER_DIM_CANDIDATES)

    init_index, init_label = get_init_index(da, init_dim, year, init_month)
    selected = da.isel({init_dim: init_index})

    # --- lead -----------------------------------------------------------
    valid_label = None
    if lead_dim is not None:
        selected, valid_label = select_lead(selected, lead_dim, lead)
    elif lead != 1:
        raise ValueError(
            f"No lead dimension found in dims={list(da.dims)}, but lead={lead} "
            "was requested. Add its name to LEAD_DIM_CANDIDATES."
        )

    # --- ensemble -------------------------------------------------------
    mode_label = "ensemble mean"
    if mode == "member":
        if member_dim is None:
            if not FORECAST_FALLBACK_TO_MEAN:
                raise ValueError(
                    f"No ensemble-member dimension found in dims={list(da.dims)}. "
                    "Add its name to MEMBER_DIM_CANDIDATES, or set "
                    "FORECAST_MODE='mean'."
                )
            print(
                "  warning: no ensemble-member dimension found "
                f"(dims={list(da.dims)}); plotting the ensemble mean instead."
            )
        else:
            selected = select_member(selected, member_dim, member)
            mode_label = f"member {member}"
    if member_dim is not None and mode == "mean":
        selected = selected.mean(dim=member_dim, skipna=True)

    if selected.ndim != 2:
        raise ValueError(
            f"Expected a 2-D field after selection, got dims={list(selected.dims)}. "
            "Check FORECAST_LEADS / MEMBER_DIM_CANDIDATES and the file structure."
        )

    info = {
        "init_label": init_label,
        "lead": lead,
        "valid_label": valid_label,
        "mode_label": mode_label,
        "init_dim": init_dim,
        "lead_dim": lead_dim,
        "member_dim": member_dim,
    }
    return selected, info


def select_lead(selected, lead_dim, lead):
    """Pick one lead time. `lead` is matched on the coordinate when possible."""
    coord = selected[lead_dim]
    values = np.asarray(coord.values)
    valid_label = None

    if np.issubdtype(values.dtype, np.datetime64):
        index = int(np.argmin(np.abs(values - values[0])))
        # datetime lead axis: take the position matching `lead` (1-based)
        index = min(max(int(lead) - 1, 0), coord.size - 1)
        valid_label = np.datetime_as_string(values[index], unit="D")
    else:
        numeric = values.astype(float) if values.dtype.kind in "iuf" else None
        if numeric is not None and np.isin(float(lead), numeric):
            index = int(np.flatnonzero(numeric == float(lead))[0])
        else:
            index = min(max(int(lead) - 1, 0), coord.size - 1)

    return selected.isel({lead_dim: index}), valid_label


def select_member(selected, member_dim, member):
    """Pick one ensemble member, by coordinate value when it exists."""
    coord = selected[member_dim]
    values = np.asarray(coord.values)
    if np.isin(member, values):
        return selected.sel({member_dim: member})
    if values.dtype.kind in "iuf" and np.isin(float(member), values.astype(float)):
        index = int(np.flatnonzero(values.astype(float) == float(member))[0])
        return selected.isel({member_dim: index})
    if 0 <= int(member) < coord.size:
        return selected.isel({member_dim: int(member)})
    raise ValueError(
        f"Member {member} not found along '{member_dim}' "
        f"(size={coord.size}, values={list(values[:10])})."
    )


def valid_month_label(init_month, lead):
    """Calendar month reached by adding `lead` months to the init month."""
    month = (int(init_month) + int(lead) - 1) % 12 + 1
    return MONTHS[month - 1]


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


def build_title_spei_forecast(method, info, init_month, window=None):
    """Title for a single-lead SPEI forecast map."""
    method_label = METHOD_INFO[method]["label"]
    spei_label = f"SPEI-{window}" if window else "SPEI"
    lead = info["lead"]

    if info["valid_label"]:
        lead_label = f"lead {lead} (valid {info['valid_label']})"
    elif FORECAST_USE_VALID_MONTH:
        lead_label = f"lead {lead} ({valid_month_label(init_month, lead)})"
    else:
        lead_label = f"lead {lead}"

    return (
        f"{spei_label} forecast - {method_label}\n"
        f"init {info['init_label']}, {lead_label}\n"
        f"{info['mode_label']}"
    )


def build_title_spei_forecast_panels(method, init_label, mode_label, init_month, leads, window=None):
    method_label = METHOD_INFO[method]["label"]
    spei_label = f"SPEI-{window}" if window else "SPEI"
    return (
        f"{spei_label} forecast - {method_label}\n"
        f"init {init_label}, leads {', '.join(str(l) for l in leads)} "
        f"(from {MONTHS[init_month - 1]})\n"
        f"{mode_label}"
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
        "#08306b"   # deeper blue
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    # cmap = plt.colormaps["RdBu"].copy()
    # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
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
def plot_spei_forecast(da, method, info, init_month, outpath, window=None, lead=None):
    """One map: SPEI forecast for a single lead time."""
    lead = lead if lead is not None else info["lead"]
    title = build_title_spei_forecast(method, info, init_month, window)

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

    spei_label = f"SPEI-{window}" if window else "SPEI"
    fig.colorbar(
        im,
        ax=ax,
        label=f"{spei_label} (-)",
        shrink=0.8,
        extend="both",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_spei_forecast_panels(
    da,
    method,
    year,
    init_month,
    leads,
    mode=FORECAST_MODE,
    member=FORECAST_MEMBER,
    outpath=None,
    window=None,
):
    """All leads of one initialization side by side, sharing one colorbar."""
    leads = list(leads)
    if not leads:
        raise ValueError("leads must not be empty")

    ncols = len(leads)
    cmap, norm = get_color_settings_spei()
    spei_label = f"SPEI-{window}" if window else "SPEI"

    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(4.0 * ncols, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.atleast_1d(axes)

    info_last = None
    for ax, lead in zip(axes, leads):
        field, info = select_spei_forecast(
            da, year=year, init_month=init_month, lead=lead, mode=mode, member=member
        )
        info_last = info

        lon, lat, vals = _get_lon_lat_vals(field)
        im = ax.pcolormesh(
            lon,
            lat,
            vals,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
        )

        if info["valid_label"]:
            lead_label = f"lead {lead} (valid {info['valid_label']})"
        elif FORECAST_USE_VALID_MONTH:
            lead_label = f"lead {lead} ({valid_month_label(init_month, lead)})"
        else:
            lead_label = f"lead {lead}"
        _format_map(ax, f"{spei_label}\n{lead_label}")

    title = build_title_spei_forecast_panels(
        method,
        info_last["init_label"],
        info_last["mode_label"],
        init_month,
        leads,
        window=window,
    )
    fig.suptitle(title)

    fig.colorbar(
        im,
        ax=list(axes),
        label=f"{spei_label} (-)",
        shrink=0.8,
        extend="both",
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return info_last


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
    spei_label = f"SPEI-{window}" if window else "SPEI"
    print(f"Wrote {file_out}")
    print(f"  field  : {spei_label} forecast, {info['mode_label']}")
    print(f"  init   : {info['init_label']}")
    print(f"  lead   : {info['lead']}" + (f" (valid {info['valid_label']})" if info["valid_label"] else ""))
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
    init_month=FORECAST_INIT_MONTH,
    leads=FORECAST_LEADS,
    mode=FORECAST_MODE,
    member=FORECAST_MEMBER,
    window=FORECAST_WINDOW,
    file_path=None,
):
    """Plot the SPEI forecast for one initialization, one figure per lead plus
    one figure with every lead side by side."""
    PATH_OUT.mkdir(parents=True, exist_ok=True)

    da = load_spei_forecast(file_path=file_path)

    # One figure per lead.
    for lead in leads:
        field, info = select_spei_forecast(
            da, year=year, init_month=init_month, lead=lead, mode=mode, member=member
        )
        outpath = build_output_file_forecast(
            COUNTRY, METHOD, lead, year, init_month, mode, window
        )
        plot_spei_forecast(field, METHOD, info, init_month, outpath, window=window)
        print_summary_forecast(outpath, info, field, window=window)

    # One figure with all leads.
    outpath_panels = build_output_file_forecast_panels(
        COUNTRY, METHOD, year, init_month, mode, window
    )
    info_last = plot_spei_forecast_panels(
        da,
        METHOD,
        year,
        init_month,
        leads,
        mode=mode,
        member=member,
        outpath=outpath_panels,
        window=window,
    )
    print(f"Wrote {outpath_panels}")
    print(f"  field  : SPEI forecast, {info_last['mode_label']}, leads {leads}")

    da.close() if hasattr(da, "close") else None


if __name__ == "__main__":
    main()
    main_forecast()

