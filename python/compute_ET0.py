import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xesmf as xe

# Directories
dir_ERA5Land = "/home/PERSONALE/alice.portal2/scratch/ERA5-Land/t2m/monthly/"
dir_MSWEP = "/home/PERSONALE/alice.portal2/scratch/MSWEP/MSWEP_V316_test/Past/Monthly/"

# Input files
tmin_file = dir_ERA5Land + "t2m_minimum_monthly_1993.nc"
tmax_file = dir_ERA5Land + "t2m_maximum_monthly_1993.nc"
tmean_file = dir_ERA5Land + "t2m_mean_monthly_1993.nc"
precip_file = dir_MSWEP + "1993.nc"

# Load datasets
ds_tmin = xr.open_dataset(tmin_file)
ds_tmax = xr.open_dataset(tmax_file)
ds_tmean = xr.open_dataset(tmean_file)
ds_precip = xr.open_dataset(precip_file)

# Extract variables (change names according to your files)
tmin = ds_tmin["t2m"]
tmax = ds_tmax["t2m"]
tmean = ds_tmean["t2m"]
precip = ds_precip["precipitation"]

# Convert temperature from Kelvin to Celsius if needed
if tmin.mean() > 100:
    tmin = tmin - 273.15
    tmax = tmax - 273.15
    tmean = tmean - 273.15

# Boxes for single countries
ghana_box = {
        'lon_min': -4, 
        'lon_max': 2, 
        'lat_min': 4, 
        'lat_max': 12
        }
madagascar_box = {
        'lon_min': 42, 
        'lon_max': 51, 
        'lat_min': -27, 
        'lat_max': -11
        }
def subset_box(da, box):
    return da.sel(
        latitude=slice(box["lat_max"], box["lat_min"]),
        longitude=slice(box["lon_min"], box["lon_max"])
    )

# Select boxes 
tmean_mg = subset_box(tmean, madagascar_box)
tmin_mg  = subset_box(tmin, madagascar_box)
tmax_mg  = subset_box(tmax, madagascar_box)

# Regrid precipitation to temperature grid
target_grid = tmean_mg
print(target_grid)
print(precip)
regridder = xe.Regridder(
    precip,
    target_grid,
    method="conservative",
    periodic=False,
    reuse_weights=False
)
precip_mg = regridder(precip)

# Functions for computing Modified Hargreaves (Droogers & Allen 2002)
import numpy as np

def extraterrestrial_radiation_daily(lat_deg, day_of_year):
    """Ra giornaliera (MJ/m²/giorno)."""
    lat = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
    
    tan_product = np.clip(-np.tan(lat) * np.tan(delta), -1, 1)
    ws = np.arccos(tan_product)
    
    Gsc = 0.0820
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(lat) * np.sin(delta) + 
        np.cos(lat) * np.cos(delta) * np.sin(ws)
    )
    return Ra


def extraterrestrial_radiation_monthly(lat_deg, month, year=2001):
    """
    Ra media mensile (MJ/m²/giorno).
    Calcola Ra per ogni giorno del mese e fa la media.
    """
    import calendar
    
    days_in_month = calendar.monthrange(year, month)[1]
    
    # Primo giorno dell'anno per quel mese
    first_doy = sum(calendar.monthrange(year, m)[1] for m in range(1, month)) + 1
    
    # Calcola Ra per ogni giorno del mese
    doys = np.arange(first_doy, first_doy + days_in_month)
    Ra_daily = extraterrestrial_radiation_daily(lat_deg, doys)
    
    return np.mean(Ra_daily)


def hargreaves_modified_monthly(tmin, tmax, precip, lat_deg, month, year=2001):
    """
    ET0 Hargreaves modificato per dati mensili.
    
    Parametri:
    ----------
    tmin, tmax : float o array
        Temperature medie mensili min/max (°C)
    precip : float o array
        Precipitazione totale mensile (mm/mese)
    lat_deg : float
        Latitudine (gradi)
    month : int
        Mese (1-12)
    year : int
        Anno (per calcolare i giorni del mese)
        
    Ritorna:
    --------
    ET0 : mm/giorno (media giornaliera per quel mese)
    """
    Ra = extraterrestrial_radiation_monthly(lat_deg, month, year)
    Ra_mm = 0.408 * Ra  # conversione in mm/giorno
    
    tmean = (tmax + tmin) / 2
    td = tmax - tmin
    
    td_adj = td - 0.0123 * precip
    
    ET0 = 0.0013 * (tmean + 17.0) * (td_adj ** 0.76) * Ra_mm # in mm/giorno

    ET0 = np.where(td_adj > 0, ET0, np.nan)

    return np.maximum(ET0, 0)

# # Check
# print(tmin)
# print(tmax)
# print(tmean)
# print(precip)
# 
# 
# # Select January (all years averaged)
# jan_tmean = tmean.isel(valid_time=0)
# 
# # Plot
# fig = plt.figure(figsize=(10, 6))
# 
# ax = plt.axes(projection=ccrs.PlateCarree())
# 
# jan_tmean.plot(
#     ax=ax,
#     transform=ccrs.PlateCarree(),
#     cmap="RdYlBu_r",
#     cbar_kwargs={"label": "Mean temperature (°C)"}
# )
# 
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linewidth=0.5)
# 
# ax.set_title("January mean temperature")
# 
# plt.savefig(
#     "January_mean_temperature.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# 
# plt.close()
