'''
Docstring for eulerian_heat_budget.src.io

Responsibilities:

- Load datasets (ERA5, model data, etc.)
- Harmonize variable names into the canonical internal schema
- Enforce pressure units (Pa) and consistent coordinate names
- Return standardized xarray.Dataset objects
- Should not perform analysis calculations (no integrals, no budgets).

Contract requirement: `io.py` is where any renaming between external conventions (ERA5 variable names) and internal names must happen (e.g., surface pressure → `sp`).

'''

import xarray as xr
import numpy as np

from collections.abc import Mapping

from . import config

n_time:int = 24
n_lat :int = 32
n_lon :int = 32

DEFAULT_CHUNKS_3D1 = {
    "time": n_time,      # 1 day per chunk
    "level": -1,     # keep full vertical column together
    "latitude": n_lat,
    "longitude": n_lon,
}

DEFAULT_CHUNKS_2D1 = {
    "time": n_time,
    "latitude": n_lat,
    "longitude": n_lon,
}


def load_era5_T(filepath: str) -> xr.Dataset:
    ds_T = xr.open_dataset(filepath, chunks=DEFAULT_CHUNKS_3D1)
    ds_T = ds_T.rename({'latitude': 'lat', 'longitude': 'lon'}) 
    ds_T = ds_T.rename({'t': 'T'})  # rename temperature variable to 'T' for consistency

    #check units are in Kelvin
    if 'units' in ds_T['T'].attrs:
        if ds_T['T'].attrs['units'] in ['K', 'kelvin', 'Kelvin']:
            pass  # already in Kelvin
        elif ds_T['T'].attrs['units'] in ['C', 'celsius', 'Celsius']:
            ds_T['T'] = ds_T['T'] + 273.15  # convert to Kelvin
            ds_T['T'].attrs['units'] = 'K'
        else:
            raise ValueError(f"Unexpected temperature units: {ds_T['T'].attrs['units']}")
    else:
        raise ValueError("Temperature variable 'T' must have 'units' attribute.")

    return ds_T

def load_era5_u(filepath: str) -> xr.Dataset:
    ds_u = xr.open_dataset(filepath, chunks=DEFAULT_CHUNKS_3D1)
    ds_u = ds_u.rename({'latitude': 'lat', 'longitude': 'lon'})

    return ds_u #[m/s]

def load_era5_omega(filepath: str) -> xr.Dataset:
    ds_omega = xr.open_dataset(filepath, chunks=DEFAULT_CHUNKS_3D1)
    ds_omega = ds_omega.rename({'latitude': 'lat', 'longitude': 'lon'})

    return ds_omega #[Pa/s]

def load_era5_sp(filepath: str) -> xr.Dataset:
    ds_sp = xr.open_dataset(filepath, chunks=DEFAULT_CHUNKS_2D1)
    ds_sp = ds_sp.rename({'latitude': 'lat', 'longitude': 'lon'})

    return ds_sp #[Pa]

# def load_era5_zg(filepath: str) -> xr.Dataset:
#     ds_z = xr.open_dataset(filepath)
#     ds_z = ds_z.rename({'latitude': 'lat', 'longitude': 'lon'})

#     ds_zg = ds_z * config.g  # convert geopotential to geopotential height
#     ds_zg = ds_zg.rename({'z': 'zg'})  # rename variable to 'zg' for consistency

#     return ds_zg  #[m]

def load_era5_surface_u(filepath: str) -> xr.Dataset:
    ds_u = xr.open_dataset(filepath, chunks=DEFAULT_CHUNKS_3D1)
    ds_u = ds_u.rename({'latitude': 'lat', 'longitude': 'lon'})

    return ds_u #[m/s]

def load_era5_surface_T(filepath: str) -> xr.Dataset:
    ds_T = xr.open_dataset(filepath, chunks=DEFAULT_CHUNKS_3D1)
    ds_T = ds_T.rename({'latitude': 'lat', 'longitude': 'lon'})

    return ds_T #[K]


def load_era5_merge_dataset(ds_T, ds_u, ds_v, ds_w, ds_sp) -> xr.Dataset:
    # Merge all datasets into a single dataset
    merged = xr.merge([ds_T, ds_u, ds_v, ds_w, ds_sp], 
                      compat='identical') 

    #Ensure dimensions are in correct order
    expected_order = ['time', 'level', 'lat', 'lon']
    merged = merged.transpose(*expected_order)

    #Enforce lat/lon monotonic ascending
    if not merged['lat'].diff('lat').min() > 0:
        merged = merged.sortby('lat')
    if not merged['lon'].diff('lon').min() > 0:
        merged = merged.sortby('lon')

    #Enforce pressure coordinates are monotonic decreasing
    dlev = merged['level'].diff('level')
    if not ((dlev < 0).all() and merged['level'].to_index().is_unique):
        merged = merged.sortby('level', ascending=False)

    merged['level']                = merged['level'] * 100.0  # convert hPa to Pa
    merged['level'].attrs['units'] = 'Pa'  # ensure pressure levels are in Pa

    return merged


