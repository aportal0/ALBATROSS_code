# Functions for computing SPEI
import calendar
import numpy as np
import xarray as xr
import pandas as pd

# Boxes and positions (lon lat)
def boxes_african_countries(name_country):
    if name_country.lower()=="ghana":
        box = {
                'lon_min': -4, 
                'lon_max': 2, 
                'lat_min': 4, 
                'lat_max': 12
                }
    elif name_country.lower()=="madagascar":
        box = {
                'lon_min': 42, 
                'lon_max': 51, 
                'lat_min': -27, 
                'lat_max': -11
                }
    return box


def subset_box(da, box):
    return da.sel(
        latitude=slice(box["lat_max"], box["lat_min"]),
        longitude=slice(box["lon_min"], box["lon_max"])
    )


# Evapotranspiration ET0
# Functions for computing Modified Hargreaves (Droogers & Allen 2002)
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


def extraterrestrial_radiation_monthly(lat_deg, month, year):
    """
    Ra media mensile (MJ/m²/giorno).
    Calcola Ra per ogni giorno del mese e fa la media.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    
    # Primo giorno dell'anno per quel mese
    first_doy = sum(calendar.monthrange(year, m)[1] for m in range(1, month)) + 1
    
    # Calcola Ra per ogni giorno del mese
    doys = np.arange(first_doy, first_doy + days_in_month)
    Ra_daily = np.ones((len(doys), len(lat_deg))) * np.nan
    for i, doy in enumerate(doys):
        Ra_daily[i] = extraterrestrial_radiation_daily(lat_deg, doy)
    
    return np.mean(Ra_daily, axis=0)


def hargreaves_modified_vectorized(ds, tmin_var='tmin', tmax_var='tmax', precip_var='precip', time_dim='time', lat_dim='latitude'):
    """
    Versione vettorizzata su tutta la dimensione temporale.
    """
    tmin = ds[tmin_var]
    tmax = ds[tmax_var]
    precip = ds[precip_var]
    lat = ds[lat_dim]

    # Calcola Ra per ogni timestep
    Ra_list = []
    for t in ds[time_dim].values:
        ts = pd.Timestamp(t)
        Ra_t = extraterrestrial_radiation_monthly(lat.values, ts.month, ts.year)
        Ra_list.append(Ra_t)

    # Stack in array (time, lat) e converti in DataArray
    Ra = np.stack(Ra_list, axis=0)
    Ra = xr.DataArray(Ra, dims=[time_dim, lat_dim],
                      coords={time_dim: ds[time_dim], lat_dim: lat})
    Ra_mm = 0.408 * Ra

    # Calcolo vettorizzato
    tmean = (tmax + tmin) / 2
    td = tmax - tmin
    td_adj = td - 0.0123 * precip

    ET0 = 0.0013 * (tmean + 17.0) * (td_adj ** 0.76) * Ra_mm
    ET0 = ET0.where(td_adj > 0).clip(min=0)

    ET0.name = 'ET0'
    ET0.attrs['units'] = 'mm/day'

    return ET0


def hargreaves_vectorized(ds, tmin_var='tmin', tmax_var='tmax', time_dim='time', lat_dim='latitude'):
    """
    Versione vettorizzata su tutta la dimensione temporale.
    """
    tmin = ds[tmin_var]
    tmax = ds[tmax_var]
    lat = ds[lat_dim]

    # Calcola Ra per ogni timestep
    Ra_list = []
    for t in ds[time_dim].values:
        ts = pd.Timestamp(t)
        Ra_t = extraterrestrial_radiation_monthly(lat.values, ts.month, ts.year)
        Ra_list.append(Ra_t)

    # Stack in array (time, lat) e converti in DataArray
    Ra = np.stack(Ra_list, axis=0)
    Ra = xr.DataArray(Ra, dims=[time_dim, lat_dim],
                      coords={time_dim: ds[time_dim], lat_dim: lat})
    Ra_mm = 0.408 * Ra

    # Calcolo vettorizzato
    tmean = (tmax + tmin) / 2
    td = tmax - tmin

    ET0 = 0.0023 * (tmean + 17.8) * (td ** 0.5) * Ra_mm

    ET0.name = 'ET0'
    ET0.attrs['units'] = 'mm/day'

    return ET0

